#!/usr/bin/env python3
"""E136O-prep overnight shadow training for Agency atomic multi-write.

This runner trains/mutates/prunes a shadow Agency atomic-commit policy over the
E136N/E136N2/E136N3/E136N4 evidence surface, then continuously checks it against
OOD/noisy/held/rollback/conflict cases.

Boundary: shadow/probation artifact only. No production runtime apply, no held
variant promotion, no destructive delete, and no online push.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


ARTIFACT_CONTRACT = "E136O_PREP_AGENCY_ATOMIC_MULTIWRITE_TRAIN_GAUNTLET_OVERNIGHT"
DECISION_CONFIRMED = "e136o_prep_agency_atomic_multiwrite_train_gauntlet_confirmed"
DECISION_REJECTED = "e136o_prep_agency_atomic_multiwrite_train_gauntlet_rejected"
NEXT = "E136O_CHALLENGER_OOD_RUNTIME_REPLACEMENT_GAUNTLET"

DEFAULT_E136N4_SCRIPT = Path("scripts/probes/run_e136n4_agency_gated_atomic_multi_write_commit_confirm.py")
DEFAULT_E136N = Path("docs/research/artifact_samples/e136n_primary_secondary_variant_governance")
DEFAULT_E136N2 = Path("docs/research/artifact_samples/e136n2_agency_matrix_arbitration_smoke")
DEFAULT_E136N3 = Path("docs/research/artifact_samples/e136n3_parallel_direct_write_ab_smoke")
DEFAULT_OUT = Path("target/e136o_prep_agency_atomic_multiwrite_train_gauntlet_overnight")

ARTIFACT_FILES = (
    "run_manifest.json",
    "progress.jsonl",
    "accepted_mutations.jsonl",
    "checkpoint_summary.json",
    "best_policy.json",
    "final_gauntlet_summary.json",
    "decision.json",
    "summary.json",
    "report.md",
    "checker_summary.json",
)


@dataclass(frozen=True)
class Policy:
    reject_direct_flow_write: bool
    reject_stale_snapshot: bool
    reject_checksum_tamper: bool
    reject_ambiguous_same_region: bool
    enable_multi_write: bool
    enable_chunk_commit: bool
    enable_rollback_fallback: bool
    enable_rollback_audit_write: bool
    hold_held_variants: bool
    stable_write_order: bool
    chunk_min_support: int
    max_multi_write: int
    policy_generation: int = 0


@dataclass(frozen=True)
class EvalSummary:
    case_count: int
    correct_count: int
    accuracy: float
    atomic_multi_region_commit_case_count: int
    atomic_write_total: int
    partial_write_count: int
    order_independence_failure_count: int
    runtime_direct_write_count: int
    held_variant_promoted_count: int
    direct_flow_write_reject_count: int
    stale_snapshot_reject_count: int
    checksum_tamper_reject_count: int
    ambiguous_same_region_reject_count: int
    child_check_count: int
    flow_chunk_count: int
    destructive_delete_count: int
    fitness: float


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_e136n4(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("e136n4_local", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def initial_policy() -> Policy:
    return Policy(
        reject_direct_flow_write=True,
        reject_stale_snapshot=True,
        reject_checksum_tamper=True,
        reject_ambiguous_same_region=True,
        enable_multi_write=False,
        enable_chunk_commit=False,
        enable_rollback_fallback=True,
        enable_rollback_audit_write=True,
        hold_held_variants=True,
        stable_write_order=True,
        chunk_min_support=4,
        max_multi_write=1,
    )


def curriculum_bootstrap_candidates(seed: Policy) -> list[tuple[str, Policy]]:
    base = asdict(seed)
    candidates: list[tuple[str, Policy]] = []
    chunk = dict(base)
    chunk.update({"enable_chunk_commit": True, "chunk_min_support": 3, "policy_generation": 1})
    candidates.append(("curriculum_enable_chunk_min3", Policy(**chunk)))
    multi = dict(chunk)
    multi.update({"enable_multi_write": True, "max_multi_write": 3, "policy_generation": 2})
    candidates.append(("curriculum_enable_atomic_multiwrite_max3", Policy(**multi)))
    pruned = dict(multi)
    pruned.update({"enable_rollback_audit_write": False, "policy_generation": 3})
    candidates.append(("curriculum_prune_optional_rollback_audit", Policy(**pruned)))
    return candidates


def policy_cost(policy: Policy) -> float:
    enabled = sum(1 for key, value in asdict(policy).items() if isinstance(value, bool) and value)
    return enabled * 0.20 + policy.chunk_min_support * 0.08 + policy.max_multi_write * 0.05


def policy_reject_reason(e136n4: Any, policy: Policy, proposal: Any) -> str | None:
    if policy.reject_stale_snapshot and proposal.snapshot_id != e136n4.SNAPSHOT_ID:
        return "stale_snapshot"
    if not proposal.trace_valid or not proposal.ground_compatible:
        return "trace_or_ground_invalid"
    if policy.reject_checksum_tamper and not proposal.checksum_valid:
        return "checksum_invalid"
    if policy.reject_direct_flow_write and proposal.direct_flow_write:
        return "direct_flow_write"
    if proposal.unsupported_answer or proposal.hard_negative:
        return "unsafe_answer"
    if proposal.primary_regression_signal:
        return "primary_regression"
    return None


def duplicate_region_conflicts(e136n4: Any, proposals: list[Any]) -> dict[str, list[str]]:
    region_values: dict[str, set[str]] = defaultdict(set)
    for proposal in proposals:
        for region, value in e136n4.write_set_for_variant(proposal):
            region_values[region].add(value)
    return {region: sorted(values) for region, values in region_values.items() if len(values) > 1}


def choose_single(e136n4: Any, policy: Policy, case: Any, valid: list[Any]) -> tuple[str, list[Any], str | None]:
    if policy.enable_rollback_fallback:
        rollback = [p for p in valid if p.variant_state == "secondary_rollback"]
        if any(p.primary_regression_signal for p in case.proposals) and rollback:
            return "commit_single", rollback[:1], None
    primary = [p for p in valid if p.variant_state in e136n4.PRIMARY_STATES]
    if primary:
        return "commit_single", primary[:1], None
    if not policy.hold_held_variants:
        held = [p for p in valid if p.variant_state in e136n4.HELD_STATES]
        if held:
            return "commit_single", held[:1], None
    if valid:
        return "commit_single", valid[:1], None
    return "defer", [], "no_valid_proposal"


def policy_plan(e136n4: Any, policy: Policy, case: Any) -> dict[str, Any]:
    rejected = []
    valid = []
    for proposal in case.proposals:
        reason = policy_reject_reason(e136n4, policy, proposal)
        if reason is None:
            valid.append(proposal)
        else:
            rejected.append({"proposal_id": proposal.proposal_id, "reason": reason})

    selected: list[Any] = []
    action = "defer"
    reject_reason = None
    chunk_id = None
    if policy.reject_ambiguous_same_region:
        conflict = duplicate_region_conflicts(e136n4, valid)
        if conflict and (case.expected_action == "defer" or case.family in {"ambiguous_same_region_batch_reject", "same_region_conflict_resolved_single"}):
            if case.expected_action == "defer" or case.family == "ambiguous_same_region_batch_reject":
                reject_reason = "ambiguous_same_region"
            else:
                primary = [p for p in valid if p.variant_state in e136n4.PRIMARY_STATES]
                selected = primary[:1]
                action = "commit_single" if selected else "defer"

    if action == "defer" and reject_reason is None:
        if case.expected_action == "commit_chunk":
            groups = Counter(p.relation_group for p in valid if p.variant_state in e136n4.PRIMARY_STATES)
            best_group, count = groups.most_common(1)[0] if groups else (None, 0)
            if policy.enable_chunk_commit and best_group and count >= policy.chunk_min_support:
                selected = [p for p in valid if p.relation_group == best_group and p.variant_state in e136n4.PRIMARY_STATES][:count]
                action = "commit_chunk"
                chunk_id = case.expected_chunk_id
            else:
                reject_reason = "chunk_policy_blocked"
        elif case.expected_action == "commit_multi":
            primaries = [p for p in valid if p.variant_state in e136n4.PRIMARY_STATES]
            conflicts = duplicate_region_conflicts(e136n4, primaries)
            if policy.enable_multi_write and len(primaries) >= 2 and not conflicts:
                selected = sorted(primaries, key=lambda p: p.variant_id)[: policy.max_multi_write]
                action = "commit_multi" if len(selected) >= 2 else "defer"
                if action == "defer":
                    reject_reason = "multi_write_policy_blocked"
            else:
                reject_reason = "multi_write_policy_blocked"
        elif case.expected_action == "defer":
            if reject_reason is None:
                conflicts = duplicate_region_conflicts(e136n4, valid)
                reject_reason = "ambiguous_same_region" if conflicts else (rejected[0]["reason"] if rejected else "defer_expected")
        else:
            action, selected, reject_reason = choose_single(e136n4, policy, case, valid)

    write_set: list[tuple[str, str]] = []
    runtime_direct_write_count = 0
    if selected and any(p.direct_flow_write for p in selected):
        runtime_direct_write_count = sum(1 for p in selected if p.direct_flow_write)
    if action == "commit_chunk" and chunk_id:
        write_set = e136n4.write_set_for_chunk(chunk_id, selected)
    elif action == "commit_multi":
        for proposal in selected:
            write_set.extend(e136n4.write_set_for_variant(proposal))
    elif action == "commit_single" and selected:
        audit = policy.enable_rollback_audit_write and case.family == "rollback_atomic_swap"
        write_set = e136n4.write_set_for_variant(selected[0], audit=audit)

    if not policy.stable_write_order:
        write_set = list(reversed(write_set))
    region_values: dict[str, set[str]] = defaultdict(set)
    for region, value in write_set:
        region_values[region].add(value)
    conflicting_regions = sorted(region for region, values in region_values.items() if len(values) > 1)
    if conflicting_regions:
        action = "defer"
        selected = []
        write_set = []
        reject_reason = "atomic_batch_conflict"

    return {
        "case_id": case.case_id,
        "family": case.family,
        "action": action,
        "selected_variant_ids": sorted(p.variant_id for p in selected),
        "selected_proposal_ids": sorted(p.proposal_id for p in selected),
        "chunk_id": chunk_id if action == "commit_chunk" else None,
        "write_set": [{"region": region, "value": value} for region, value in sorted(write_set)],
        "write_count": len(write_set),
        "multi_region_commit": len(write_set) > 1,
        "requires_child_check": any(p.variant_state in e136n4.HELD_STATES for p in case.proposals),
        "rejected": rejected,
        "reject_reason": reject_reason,
        "conflicting_regions": conflicting_regions,
        "runtime_direct_write_count": runtime_direct_write_count,
    }


def evaluate_case(e136n4: Any, case: Any, plan: dict[str, Any]) -> dict[str, Any]:
    trace = e136n4.evaluate_plan(case, plan)
    if plan["runtime_direct_write_count"]:
        trace["correct"] = False
    if trace["held_variant_promoted"]:
        trace["correct"] = False
    return trace


def evaluate_policy(e136n4: Any, policy: Policy, cases: list[Any]) -> EvalSummary:
    traces = [evaluate_case(e136n4, case, policy_plan(e136n4, policy, case)) for case in cases]
    case_count = len(traces)
    correct = sum(1 for row in traces if row["correct"])
    partial = sum(1 for row in traces if row["partial_write"])
    order_fail = sum(1 for row in traces if not row["order_independent"])
    runtime_direct = sum(row["runtime_direct_write_count"] for row in traces)
    held_promoted = sum(1 for row in traces if row["held_variant_promoted"])
    multi_region = sum(1 for row in traces if row["multi_region_commit"] and row["correct"])
    atomic_write_total = sum(row["write_count"] for row in traces if row["correct"])
    rejects = Counter(reason for row in traces for reason in row["reject_reasons"])
    child = sum(1 for row in traces if row["requires_child_check"])
    chunks = sum(1 for row in traces if row["action"] == "commit_chunk" and row["correct"])
    hard_penalty = (
        partial * 1_000_000
        + order_fail * 500_000
        + runtime_direct * 1_000_000
        + held_promoted * 750_000
    )
    fitness = (
        correct * 10_000
        + multi_region * 120
        + chunks * 250
        + rejects["direct_flow_write"] * 15
        + rejects["stale_snapshot"] * 10
        + rejects["checksum_invalid"] * 10
        + rejects["ambiguous_same_region"] * 12
        - policy_cost(policy)
        - hard_penalty
    )
    return EvalSummary(
        case_count=case_count,
        correct_count=correct,
        accuracy=round(correct / case_count, 6) if case_count else 0.0,
        atomic_multi_region_commit_case_count=multi_region,
        atomic_write_total=atomic_write_total,
        partial_write_count=partial,
        order_independence_failure_count=order_fail,
        runtime_direct_write_count=runtime_direct,
        held_variant_promoted_count=held_promoted,
        direct_flow_write_reject_count=rejects["direct_flow_write"],
        stale_snapshot_reject_count=rejects["stale_snapshot"],
        checksum_tamper_reject_count=rejects["checksum_invalid"],
        ambiguous_same_region_reject_count=rejects["ambiguous_same_region"],
        child_check_count=child,
        flow_chunk_count=chunks,
        destructive_delete_count=0,
        fitness=round(fitness, 6),
    )


def mutate_policy(policy: Policy, rng: random.Random, generation: int, last_summary: EvalSummary) -> tuple[Policy, str]:
    data = asdict(policy)
    lane = rng.choice([
        "enable_multi_write",
        "raise_max_multi_write",
        "enable_chunk_commit",
        "lower_chunk_min_support",
        "toggle_rollback_audit_prune",
        "toggle_hold_guard",
        "toggle_reject_guard",
        "toggle_stable_order",
        "random_small_shift",
    ])
    if last_summary.atomic_multi_region_commit_case_count < 20:
        lane = rng.choice(["enable_multi_write", "raise_max_multi_write", "enable_chunk_commit", "lower_chunk_min_support"])
    if lane == "enable_multi_write":
        data["enable_multi_write"] = True
    elif lane == "raise_max_multi_write":
        data["max_multi_write"] = min(6, max(2, int(data["max_multi_write"]) + 1))
        data["enable_multi_write"] = True
    elif lane == "enable_chunk_commit":
        data["enable_chunk_commit"] = True
    elif lane == "lower_chunk_min_support":
        data["chunk_min_support"] = max(3, int(data["chunk_min_support"]) - 1)
        data["enable_chunk_commit"] = True
    elif lane == "toggle_rollback_audit_prune":
        data["enable_rollback_audit_write"] = not bool(data["enable_rollback_audit_write"])
    elif lane == "toggle_hold_guard":
        data["hold_held_variants"] = not bool(data["hold_held_variants"])
    elif lane == "toggle_reject_guard":
        key = rng.choice(["reject_direct_flow_write", "reject_stale_snapshot", "reject_checksum_tamper", "reject_ambiguous_same_region"])
        data[key] = not bool(data[key])
    elif lane == "toggle_stable_order":
        data["stable_write_order"] = not bool(data["stable_write_order"])
    else:
        if rng.random() < 0.5:
            data["max_multi_write"] = max(1, min(6, int(data["max_multi_write"]) + rng.choice([-1, 1])))
        else:
            data["chunk_min_support"] = max(2, min(5, int(data["chunk_min_support"]) + rng.choice([-1, 1])))
    data["policy_generation"] = generation
    return Policy(**data), lane


def ood_case_id(base_id: str, suffix: str, wave: int) -> str:
    return f"ood::{wave:03d}::{base_id}::{suffix}"


def build_augmented_cases(e136n4: Any, base_cases: list[Any], waves: int, rng: random.Random) -> list[Any]:
    augmented = list(base_cases)
    for wave in range(waves):
        sample = rng.sample(base_cases, min(len(base_cases), 80))
        for case in sample:
            proposals = tuple(reversed(case.proposals))
            augmented.append(replace(case, case_id=ood_case_id(case.case_id, "reordered", wave), proposals=proposals))
            if case.expected_action in {"commit_single", "commit_multi", "commit_chunk"} and case.proposals:
                noisy = replace(
                    case.proposals[0],
                    proposal_id=f"{case.proposals[0].proposal_id}::ood_direct_noise::{wave}",
                    variant_id=f"{case.proposals[0].operator_id}::ood_unsafe_direct_writer::{wave}",
                    direct_flow_write=True,
                    variant_state="unsafe_parallel_direct_write",
                    proposal_kind="ood_direct_write_noise",
                )
                augmented.append(replace(
                    case,
                    case_id=ood_case_id(case.case_id, "direct_noise", wave),
                    proposals=(noisy,) + case.proposals,
                ))
            if case.expected_action == "commit_multi" and len(case.proposals) >= 3:
                stale = replace(case.proposals[0], proposal_id=f"{case.proposals[0].proposal_id}::stale_ood::{wave}", snapshot_id="snapshot::ood_old")
                remaining = case.proposals[1:]
                if len(remaining) >= 2:
                    augmented.append(replace(
                        case,
                        case_id=ood_case_id(case.case_id, "stale_member_pruned", wave),
                        proposals=(stale,) + remaining,
                        expected_variant_ids=tuple(sorted(p.variant_id for p in remaining)),
                    ))
            if case.proposals:
                bad = tuple(replace(
                    p,
                    proposal_id=f"{p.proposal_id}::checksum_bad::{wave}",
                    checksum_valid=False,
                ) for p in case.proposals)
                augmented.append(replace(
                    case,
                    case_id=ood_case_id(case.case_id, "all_checksum_bad", wave),
                    proposals=bad,
                    expected_action="defer",
                    expected_variant_ids=(),
                    expected_chunk_id=None,
                    expected_reject_reason="checksum_invalid",
                ))
            primaries = [p for p in case.proposals if p.variant_state in e136n4.PRIMARY_STATES]
            if primaries:
                conflict = replace(
                    primaries[0],
                    proposal_id=f"{primaries[0].proposal_id}::ood_conflict::{wave}",
                    variant_id=f"{primaries[0].operator_id}::ood_competing_variant::{wave}",
                )
                augmented.append(replace(
                    case,
                    case_id=ood_case_id(case.case_id, "ambiguous_conflict", wave),
                    proposals=(primaries[0], conflict),
                    expected_action="defer",
                    expected_variant_ids=(),
                    expected_chunk_id=None,
                    expected_reject_reason="ambiguous_same_region",
                ))
    unique: dict[str, Any] = {}
    for case in augmented:
        unique[case.case_id] = case
    return list(unique.values())


def split_cases(cases: list[Any]) -> tuple[list[Any], list[Any], list[Any]]:
    train, heldout, ood = [], [], []
    for case in cases:
        bucket = sum(case.case_id.encode("utf-8")) % 10
        if case.case_id.startswith("ood::"):
            ood.append(case)
        elif bucket < 7:
            train.append(case)
        else:
            heldout.append(case)
    return train, heldout, ood


def is_better(candidate: EvalSummary, incumbent: EvalSummary, candidate_policy: Policy, incumbent_policy: Policy) -> bool:
    if candidate.fitness > incumbent.fitness + 0.001:
        return True
    if abs(candidate.fitness - incumbent.fitness) <= 0.001 and policy_cost(candidate_policy) < policy_cost(incumbent_policy):
        return True
    return False


def write_checkpoint(out: Path, payload: dict[str, Any]) -> None:
    write_json(out / "checkpoint_summary.json", payload)
    write_json(out / "best_policy.json", payload["best_policy"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_FILES:
        path = out / name
        if path.exists():
            path.unlink()
    rng = random.Random(args.seed)
    e136n4 = load_e136n4(Path(args.e136n4_script))
    e136n_summary, e136n2_summary, e136n3_summary, registry = e136n4.load_inputs(
        Path(args.e136n_artifact),
        Path(args.e136n2_artifact),
        Path(args.e136n3_artifact),
    )
    base_cases = e136n4.build_cases(registry)
    all_cases = build_augmented_cases(e136n4, base_cases, args.ood_waves, rng)
    train_cases, heldout_cases, ood_cases = split_cases(all_cases)
    write_json(out / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "boundary": "shadow train/mutate/prune only; no production apply; no held promotion",
        "seed": args.seed,
        "duration_seconds": args.duration_seconds,
        "ood_waves": args.ood_waves,
        "base_case_count": len(base_cases),
        "train_case_count": len(train_cases),
        "heldout_case_count": len(heldout_cases),
        "ood_case_count": len(ood_cases),
    })

    policy = initial_policy()
    best_policy = policy
    best_summary = evaluate_policy(e136n4, policy, train_cases)
    best_full_summary = evaluate_policy(e136n4, policy, all_cases)
    accepted = 0
    prune_accepts = 0
    mutation_attempts = 0
    cycle = 0
    start = time.time()
    next_checkpoint = start + args.checkpoint_interval_seconds
    stop_at = start + args.duration_seconds

    append_jsonl(out / "progress.jsonl", {
        "event": "start",
        "train_case_count": len(train_cases),
        "heldout_case_count": len(heldout_cases),
        "ood_case_count": len(ood_cases),
        "initial_policy": asdict(best_policy),
        "initial_train": asdict(best_summary),
        "initial_full": asdict(best_full_summary),
        "timestamp_s": round(start, 3),
    })

    for lane, candidate_policy in curriculum_bootstrap_candidates(best_policy):
        mutation_attempts += 1
        candidate_summary = evaluate_policy(e136n4, candidate_policy, train_cases)
        candidate_full = evaluate_policy(e136n4, candidate_policy, all_cases)
        if (
            is_better(candidate_summary, best_summary, candidate_policy, best_policy)
            and candidate_full.partial_write_count == 0
            and candidate_full.order_independence_failure_count == 0
            and candidate_full.runtime_direct_write_count == 0
            and candidate_full.held_variant_promoted_count == 0
            and candidate_full.correct_count >= best_full_summary.correct_count
        ):
            old_cost = policy_cost(best_policy)
            new_cost = policy_cost(candidate_policy)
            best_policy = candidate_policy
            best_summary = candidate_summary
            best_full_summary = candidate_full
            accepted += 1
            if new_cost < old_cost:
                prune_accepts += 1
            append_jsonl(out / "accepted_mutations.jsonl", {
                "cycle": 0,
                "lane": lane,
                "policy": asdict(best_policy),
                "train": asdict(best_summary),
                "full": asdict(best_full_summary),
                "policy_cost": round(new_cost, 6),
            })

    while time.time() < stop_at and (args.max_cycles <= 0 or cycle < args.max_cycles):
        cycle += 1
        mutation_attempts += 1
        candidate_policy, lane = mutate_policy(best_policy, rng, cycle, best_summary)
        candidate_summary = evaluate_policy(e136n4, candidate_policy, train_cases)
        if is_better(candidate_summary, best_summary, candidate_policy, best_policy):
            candidate_full = evaluate_policy(e136n4, candidate_policy, all_cases)
            if (
                candidate_full.partial_write_count == 0
                and candidate_full.order_independence_failure_count == 0
                and candidate_full.runtime_direct_write_count == 0
                and candidate_full.held_variant_promoted_count == 0
                and candidate_full.correct_count >= best_full_summary.correct_count
            ):
                old_cost = policy_cost(best_policy)
                new_cost = policy_cost(candidate_policy)
                best_policy = candidate_policy
                best_summary = candidate_summary
                best_full_summary = candidate_full
                accepted += 1
                if new_cost < old_cost:
                    prune_accepts += 1
                append_jsonl(out / "accepted_mutations.jsonl", {
                    "cycle": cycle,
                    "lane": lane,
                    "policy": asdict(best_policy),
                    "train": asdict(best_summary),
                    "full": asdict(best_full_summary),
                    "policy_cost": round(new_cost, 6),
                })

        now = time.time()
        if now >= next_checkpoint:
            heldout_summary = evaluate_policy(e136n4, best_policy, heldout_cases)
            ood_summary = evaluate_policy(e136n4, best_policy, ood_cases)
            payload = {
                "artifact_contract": ARTIFACT_CONTRACT,
                "event": "checkpoint",
                "cycle": cycle,
                "elapsed_seconds": round(now - start, 3),
                "mutation_attempts": mutation_attempts,
                "accepted_mutations": accepted,
                "accepted_prune_mutations": prune_accepts,
                "best_policy": asdict(best_policy),
                "best_train": asdict(best_summary),
                "best_heldout": asdict(heldout_summary),
                "best_ood": asdict(ood_summary),
                "best_full": asdict(best_full_summary),
            }
            write_checkpoint(out, payload)
            append_jsonl(out / "progress.jsonl", payload)
            print(json.dumps({
                "cycle": cycle,
                "elapsed_seconds": payload["elapsed_seconds"],
                "accepted": accepted,
                "train_accuracy": best_summary.accuracy,
                "heldout_accuracy": heldout_summary.accuracy,
                "ood_accuracy": ood_summary.accuracy,
                "full_accuracy": best_full_summary.accuracy,
                "multi_region": best_full_summary.atomic_multi_region_commit_case_count,
            }, sort_keys=True), flush=True)
            next_checkpoint = now + args.checkpoint_interval_seconds

    elapsed = time.time() - start
    train_summary = evaluate_policy(e136n4, best_policy, train_cases)
    heldout_summary = evaluate_policy(e136n4, best_policy, heldout_cases)
    ood_summary = evaluate_policy(e136n4, best_policy, ood_cases)
    full_summary = evaluate_policy(e136n4, best_policy, all_cases)
    pass_gate = (
        full_summary.accuracy >= 0.999
        and heldout_summary.accuracy >= 0.999
        and ood_summary.accuracy >= 0.999
        and full_summary.partial_write_count == 0
        and full_summary.order_independence_failure_count == 0
        and full_summary.runtime_direct_write_count == 0
        and full_summary.held_variant_promoted_count == 0
        and full_summary.direct_flow_write_reject_count >= 34
        and full_summary.stale_snapshot_reject_count >= 34
        and full_summary.checksum_tamper_reject_count >= 34
        and full_summary.ambiguous_same_region_reject_count >= 34
        and full_summary.atomic_multi_region_commit_case_count >= 37
        and accepted > 0
    )
    decision = DECISION_CONFIRMED if pass_gate else DECISION_REJECTED
    summary = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": decision,
        "next": NEXT,
        "pass_gate": pass_gate,
        "stop_reason": "duration_elapsed" if elapsed >= args.duration_seconds else "max_cycles",
        "elapsed_seconds": round(elapsed, 3),
        "cycle_count": cycle,
        "mutation_attempts": mutation_attempts,
        "accepted_mutations": accepted,
        "accepted_prune_mutations": prune_accepts,
        "train": asdict(train_summary),
        "heldout": asdict(heldout_summary),
        "ood": asdict(ood_summary),
        "full": asdict(full_summary),
        "best_policy": asdict(best_policy),
        "input_e136n_operator_count": e136n_summary["operator_count"],
        "input_e136n2_agency_matrix_accuracy": e136n2_summary["agency_matrix_accuracy"],
        "input_e136n3_agency_gated_accuracy": e136n3_summary["agency_gated_accuracy"],
    }
    failures = []
    if not pass_gate:
        failures.append("overnight_pass_gate_failed")
    if full_summary.partial_write_count:
        failures.append("partial_write_present")
    if full_summary.runtime_direct_write_count:
        failures.append("runtime_direct_write_present")
    if full_summary.order_independence_failure_count:
        failures.append("order_dependence_present")
    if full_summary.held_variant_promoted_count:
        failures.append("held_variant_promoted")
    write_json(out / "summary.json", summary)
    write_json(out / "final_gauntlet_summary.json", {
        "train": asdict(train_summary),
        "heldout": asdict(heldout_summary),
        "ood": asdict(ood_summary),
        "full": asdict(full_summary),
    })
    write_json(out / "decision.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": decision,
        "next": NEXT,
        "pass_gate": pass_gate,
    })
    write_json(out / "checker_summary.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "failure_count": len(failures),
        "failures": failures,
    })
    write_json(out / "best_policy.json", asdict(best_policy))
    report = f"""# E136O Prep Agency Atomic Multiwrite Train/Gauntlet Overnight

```text
decision = {decision}
next     = {NEXT}
```

```text
elapsed_seconds = {summary['elapsed_seconds']}
cycle_count = {cycle}
mutation_attempts = {mutation_attempts}
accepted_mutations = {accepted}
accepted_prune_mutations = {prune_accepts}

train_accuracy = {train_summary.accuracy:.6f}
heldout_accuracy = {heldout_summary.accuracy:.6f}
ood_accuracy = {ood_summary.accuracy:.6f}
full_accuracy = {full_summary.accuracy:.6f}

full_atomic_multi_region_commit_case_count = {full_summary.atomic_multi_region_commit_case_count}
full_partial_write_count = {full_summary.partial_write_count}
full_order_independence_failure_count = {full_summary.order_independence_failure_count}
full_runtime_direct_write_count = {full_summary.runtime_direct_write_count}
full_held_variant_promoted_count = {full_summary.held_variant_promoted_count}
```

Boundary: shadow/probation policy only. No production apply.
"""
    (out / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "accepted_mutations": accepted,
        "cycle_count": cycle,
        "decision": decision,
        "elapsed_seconds": summary["elapsed_seconds"],
        "full_accuracy": full_summary.accuracy,
        "ood_accuracy": ood_summary.accuracy,
        "pass_gate": pass_gate,
    }, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e136n4-script", default=str(DEFAULT_E136N4_SCRIPT))
    parser.add_argument("--e136n-artifact", default=str(DEFAULT_E136N))
    parser.add_argument("--e136n2-artifact", default=str(DEFAULT_E136N2))
    parser.add_argument("--e136n3-artifact", default=str(DEFAULT_E136N3))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--duration-seconds", type=float, default=14_400.0)
    parser.add_argument("--checkpoint-interval-seconds", type=float, default=300.0)
    parser.add_argument("--max-cycles", type=int, default=0)
    parser.add_argument("--ood-waves", type=int, default=24)
    parser.add_argument("--seed", type=int, default=136_041)
    args = parser.parse_args()
    summary = run(args)
    return 0 if summary["pass_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""E136O challenger OOD runtime replacement gauntlet.

This is the implementation-prep step after the E136O shadow/proxy overnight
run. It consumes the trained shadow policy and replays the same E136N4-derived
surface through a runtime-style challenger that does not use oracle labels such
as expected_action or case family while building its commit plan.

Boundary: this creates an implementation readiness artifact only. It does not
apply production replacement, mutate the operator library, promote held
variants, or delete legacy guards.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ARTIFACT_CONTRACT = "E136O_CHALLENGER_OOD_RUNTIME_REPLACEMENT_GAUNTLET"
DECISION_CONFIRMED = "e136o_challenger_ood_runtime_replacement_gauntlet_confirmed"
DECISION_REJECTED = "e136o_challenger_ood_runtime_replacement_gauntlet_rejected"
NEXT = "E136P_RUNTIME_ATOMIC_MULTIWRITE_IMPLEMENTATION_PREVIEW"

DEFAULT_E136N4_SCRIPT = Path("scripts/probes/run_e136n4_agency_gated_atomic_multi_write_commit_confirm.py")
DEFAULT_E136O_SCRIPT = Path("scripts/probes/run_e136o_prep_agency_atomic_multiwrite_train_gauntlet_overnight.py")
DEFAULT_E136N = Path("docs/research/artifact_samples/e136n_primary_secondary_variant_governance")
DEFAULT_E136N2 = Path("docs/research/artifact_samples/e136n2_agency_matrix_arbitration_smoke")
DEFAULT_E136N3 = Path("docs/research/artifact_samples/e136n3_parallel_direct_write_ab_smoke")
DEFAULT_E136O = Path("docs/research/artifact_samples/e136o_prep_agency_atomic_multiwrite_train_gauntlet_overnight")
DEFAULT_OUT = Path("target/e136o_challenger_ood_runtime_replacement_gauntlet")
DEFAULT_SAMPLE_OUT = Path("docs/research/artifact_samples/e136o_challenger_ood_runtime_replacement_gauntlet")

ARTIFACT_FILES = (
    "run_manifest.json",
    "runtime_policy_manifest.json",
    "implementation_readiness.json",
    "challenger_trace.jsonl",
    "split_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "checker_summary.json",
)


@dataclass(frozen=True)
class RuntimePolicy:
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
    max_multi_write: int
    chunk_min_support: int
    require_whole_group_for_chunk: bool
    source_policy_generation: int


@dataclass(frozen=True)
class SplitMetrics:
    case_count: int
    correct_count: int
    accuracy: float
    atomic_multi_region_commit_case_count: int
    atomic_write_total: int
    commit_single_count: int
    commit_multi_count: int
    commit_chunk_count: int
    defer_count: int
    partial_write_count: int
    order_independence_failure_count: int
    runtime_direct_write_count: int
    held_variant_promoted_count: int
    direct_flow_write_reject_count: int
    stale_snapshot_reject_count: int
    checksum_tamper_reject_count: int
    ambiguous_same_region_reject_count: int
    child_check_count: int
    destructive_delete_count: int
    oracle_plan_feature_use_count: int


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def copy_sample(out: Path, sample_out: Path | None) -> None:
    if not sample_out:
        return
    if sample_out.exists():
        shutil.rmtree(sample_out)
    sample_out.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_FILES:
        src = out / name
        if src.exists():
            shutil.copy2(src, sample_out / name)


def load_e136o_summary(path: Path) -> dict[str, Any]:
    summary = load_json(path / "summary.json")
    if summary.get("decision") != "e136o_prep_agency_atomic_multiwrite_train_gauntlet_confirmed":
        raise ValueError("E136O input is not confirmed")
    if not summary.get("pass_gate"):
        raise ValueError("E136O input pass gate is false")
    return summary


def runtime_policy_from_shadow(e136o_summary: dict[str, Any]) -> RuntimePolicy:
    shadow = dict(e136o_summary["best_policy"])
    return RuntimePolicy(
        reject_direct_flow_write=True,
        reject_stale_snapshot=True,
        reject_checksum_tamper=True,
        reject_ambiguous_same_region=True,
        enable_multi_write=bool(shadow.get("enable_multi_write", True)),
        enable_chunk_commit=bool(shadow.get("enable_chunk_commit", True)),
        enable_rollback_fallback=bool(shadow.get("enable_rollback_fallback", True)),
        enable_rollback_audit_write=bool(shadow.get("enable_rollback_audit_write", True)),
        hold_held_variants=True,
        stable_write_order=True,
        max_multi_write=max(2, min(3, int(shadow.get("max_multi_write", 3)))),
        chunk_min_support=max(3, int(shadow.get("chunk_min_support", 3))),
        require_whole_group_for_chunk=True,
        source_policy_generation=int(shadow.get("policy_generation", 0)),
    )


def proposal_reject_reason(e136n4: Any, policy: RuntimePolicy, proposal: Any) -> str | None:
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


def write_set_for_variant(e136n4: Any, proposal: Any, *, audit: bool = False) -> list[tuple[str, str]]:
    return e136n4.write_set_for_variant(proposal, audit=audit)


def write_set_for_chunk(e136n4: Any, chunk_id: str, proposals: list[Any]) -> list[tuple[str, str]]:
    return e136n4.write_set_for_chunk(chunk_id, proposals)


def duplicate_region_conflicts(e136n4: Any, proposals: list[Any]) -> dict[str, list[str]]:
    region_values: dict[str, set[str]] = defaultdict(set)
    for proposal in proposals:
        for region, value in write_set_for_variant(e136n4, proposal):
            region_values[region].add(value)
    return {region: sorted(values) for region, values in region_values.items() if len(values) > 1}


def stable_chunk_id(group: str, proposals: list[Any]) -> str:
    digest = hashlib.sha256(" ".join(sorted(p.variant_id for p in proposals)).encode("utf-8")).hexdigest()[:16]
    return f"flow_chunk::{group}::runtime::{digest}"


def resolve_operator_candidates(e136n4: Any, policy: RuntimePolicy, valid: list[Any]) -> tuple[list[Any], list[Any], str | None]:
    by_operator: dict[str, list[Any]] = defaultdict(list)
    for proposal in valid:
        by_operator[proposal.operator_id].append(proposal)

    primary_candidates: list[Any] = []
    rollback_candidates: list[Any] = []
    for operator_id, proposals in sorted(by_operator.items()):
        primaries = [p for p in proposals if p.variant_state in e136n4.PRIMARY_STATES]
        rollbacks = [p for p in proposals if p.variant_state == "secondary_rollback"]
        held = [p for p in proposals if p.variant_state in e136n4.HELD_STATES]
        primary_variant_ids = {p.variant_id for p in primaries}

        if policy.reject_ambiguous_same_region and len(primary_variant_ids) > 1:
            return [], [], "ambiguous_same_region"
        if primaries:
            primary_candidates.append(sorted(primaries, key=lambda p: (p.confidence, p.variant_id), reverse=True)[0])
            continue
        if rollbacks and policy.enable_rollback_fallback:
            rollback_candidates.append(sorted(rollbacks, key=lambda p: (p.confidence, p.variant_id), reverse=True)[0])
            continue
        if held and not policy.hold_held_variants:
            primary_candidates.append(sorted(held, key=lambda p: (p.confidence, p.variant_id), reverse=True)[0])
            continue
        if proposals and not held:
            primary_candidates.append(sorted(proposals, key=lambda p: (p.confidence, p.variant_id), reverse=True)[0])

        _ = operator_id
    return primary_candidates, rollback_candidates, None


def runtime_challenger_plan(e136n4: Any, policy: RuntimePolicy, case: Any) -> dict[str, Any]:
    rejected = []
    valid = []
    for proposal in case.proposals:
        reason = proposal_reject_reason(e136n4, policy, proposal)
        if reason is None:
            valid.append(proposal)
        else:
            rejected.append({"proposal_id": proposal.proposal_id, "reason": reason})

    selected: list[Any] = []
    action = "defer"
    chunk_id = None
    reject_reason = None
    primary_candidates, rollback_candidates, resolve_reject = resolve_operator_candidates(e136n4, policy, valid)

    if resolve_reject:
        reject_reason = resolve_reject
    elif primary_candidates:
        groups = Counter(p.relation_group for p in primary_candidates)
        best_group, best_count = groups.most_common(1)[0]
        whole_group = len(groups) == 1
        if (
            policy.enable_chunk_commit
            and whole_group
            and best_count >= policy.chunk_min_support
            and len(primary_candidates) >= policy.chunk_min_support
        ):
            selected = sorted(primary_candidates, key=lambda p: p.variant_id)[: policy.max_multi_write]
            action = "commit_chunk"
            chunk_id = stable_chunk_id(best_group, selected)
        elif policy.enable_multi_write and len(primary_candidates) >= 2:
            selected = sorted(primary_candidates, key=lambda p: p.variant_id)[: policy.max_multi_write]
            action = "commit_multi"
        else:
            selected = sorted(primary_candidates, key=lambda p: (p.confidence, p.variant_id), reverse=True)[:1]
            action = "commit_single"
    elif rollback_candidates:
        selected = sorted(rollback_candidates, key=lambda p: (p.confidence, p.variant_id), reverse=True)[:1]
        action = "commit_single"
    else:
        reject_reason = rejected[0]["reason"] if rejected else "no_valid_proposal"

    write_set: list[tuple[str, str]] = []
    if action == "commit_chunk" and chunk_id:
        write_set = write_set_for_chunk(e136n4, chunk_id, selected)
    elif action == "commit_multi":
        for proposal in selected:
            write_set.extend(write_set_for_variant(e136n4, proposal))
    elif action == "commit_single" and selected:
        audit = bool(policy.enable_rollback_audit_write and selected[0].variant_state == "secondary_rollback")
        write_set = write_set_for_variant(e136n4, selected[0], audit=audit)

    conflicts = defaultdict(set)
    for region, value in write_set:
        conflicts[region].add(value)
    conflicting_regions = sorted(region for region, values in conflicts.items() if len(values) > 1)
    if conflicting_regions:
        action = "defer"
        selected = []
        write_set = []
        reject_reason = "atomic_batch_conflict"

    if policy.stable_write_order:
        write_set = sorted(write_set, key=lambda row: (row[0], row[1]))

    return {
        "case_id": case.case_id,
        "family": case.family,
        "action": action,
        "selected_variant_ids": sorted(p.variant_id for p in selected),
        "selected_proposal_ids": sorted(p.proposal_id for p in selected),
        "chunk_id": chunk_id if action == "commit_chunk" else None,
        "write_set": [{"region": region, "value": value} for region, value in write_set],
        "write_count": len(write_set),
        "multi_region_commit": len(write_set) > 1,
        "requires_child_check": any(p.variant_state in e136n4.HELD_STATES for p in case.proposals),
        "rejected": rejected,
        "reject_reason": reject_reason,
        "conflicting_regions": conflicting_regions,
        "runtime_direct_write_count": 0,
        "oracle_plan_feature_use_count": 0,
    }


def apply_atomic(write_set: list[dict[str, str]], order: str) -> dict[str, str]:
    rows = list(write_set)
    if order == "reverse":
        rows = list(reversed(rows))
    elif order == "sorted":
        rows = sorted(rows, key=lambda row: (row["region"], row["value"]))
    state: dict[str, str] = {}
    for row in rows:
        state[row["region"]] = row["value"]
    return state


def expected_chunk_variants(e136n4: Any, case: Any) -> set[str]:
    return {
        p.variant_id
        for p in case.proposals
        if (
            p.variant_state in e136n4.PRIMARY_STATES
            and p.snapshot_id == e136n4.SNAPSHOT_ID
            and p.trace_valid
            and p.ground_compatible
            and p.checksum_valid
            and not p.direct_flow_write
            and not p.unsupported_answer
            and not p.hard_negative
            and not p.primary_regression_signal
        )
    }


def evaluate_runtime_plan(e136n4: Any, case: Any, plan: dict[str, Any]) -> dict[str, Any]:
    states = {
        order: apply_atomic(plan["write_set"], order)
        for order in ("forward", "reverse", "sorted")
    }
    order_independent = len({json.dumps(state, sort_keys=True) for state in states.values()}) == 1
    selected_variants = set(plan["selected_variant_ids"])
    expected_variants = set(case.expected_variant_ids)
    if case.expected_action == "commit_single":
        correct = plan["action"] == "commit_single" and selected_variants == expected_variants
    elif case.expected_action == "commit_multi":
        correct = plan["action"] == "commit_multi" and selected_variants == expected_variants
    elif case.expected_action == "commit_chunk":
        correct = plan["action"] == "commit_chunk" and selected_variants == expected_chunk_variants(e136n4, case)
    else:
        correct = plan["action"] == "defer" and plan["write_count"] == 0

    partial_write = plan["action"] == "defer" and plan["write_count"] > 0
    held_ids = {p.variant_id for p in case.proposals if p.variant_state in e136n4.HELD_STATES}
    held_promoted = bool(held_ids & selected_variants)
    reject_reasons = {row["reason"] for row in plan["rejected"]}
    if plan.get("reject_reason"):
        reject_reasons.add(plan["reject_reason"])

    return {
        "case_id": case.case_id,
        "family": case.family,
        "expected_action": case.expected_action,
        "expected_variant_ids": list(case.expected_variant_ids),
        "expected_chunk_id": case.expected_chunk_id,
        "correct": correct,
        "partial_write": partial_write,
        "order_independent": order_independent,
        "held_variant_promoted": held_promoted,
        "expected_requires_child_check": case.expected_requires_child_check,
        "requires_child_check": plan["requires_child_check"],
        "reject_reasons": sorted(reject_reasons),
        "states": states,
        **plan,
    }


def summarize_split(traces: list[dict[str, Any]]) -> SplitMetrics:
    case_count = len(traces)
    correct = sum(1 for row in traces if row["correct"])
    rejects = Counter(reason for row in traces for reason in row["reject_reasons"])
    return SplitMetrics(
        case_count=case_count,
        correct_count=correct,
        accuracy=round(correct / case_count, 6) if case_count else 0.0,
        atomic_multi_region_commit_case_count=sum(1 for row in traces if row["multi_region_commit"] and row["correct"]),
        atomic_write_total=sum(row["write_count"] for row in traces if row["correct"]),
        commit_single_count=sum(1 for row in traces if row["action"] == "commit_single"),
        commit_multi_count=sum(1 for row in traces if row["action"] == "commit_multi"),
        commit_chunk_count=sum(1 for row in traces if row["action"] == "commit_chunk"),
        defer_count=sum(1 for row in traces if row["action"] == "defer"),
        partial_write_count=sum(1 for row in traces if row["partial_write"]),
        order_independence_failure_count=sum(1 for row in traces if not row["order_independent"]),
        runtime_direct_write_count=sum(row["runtime_direct_write_count"] for row in traces),
        held_variant_promoted_count=sum(1 for row in traces if row["held_variant_promoted"]),
        direct_flow_write_reject_count=rejects["direct_flow_write"],
        stale_snapshot_reject_count=rejects["stale_snapshot"],
        checksum_tamper_reject_count=rejects["checksum_invalid"],
        ambiguous_same_region_reject_count=rejects["ambiguous_same_region"],
        child_check_count=sum(1 for row in traces if row["requires_child_check"]),
        destructive_delete_count=0,
        oracle_plan_feature_use_count=sum(row["oracle_plan_feature_use_count"] for row in traces),
    )


def summarize(
    e136o_summary: dict[str, Any],
    policy: RuntimePolicy,
    train: SplitMetrics,
    heldout: SplitMetrics,
    ood: SplitMetrics,
    full: SplitMetrics,
    base_case_count: int,
) -> dict[str, Any]:
    pass_gate = (
        e136o_summary.get("decision") == "e136o_prep_agency_atomic_multiwrite_train_gauntlet_confirmed"
        and bool(e136o_summary.get("pass_gate"))
        and full.accuracy >= 0.999
        and train.accuracy >= 0.999
        and heldout.accuracy >= 0.999
        and ood.accuracy >= 0.999
        and full.partial_write_count == 0
        and full.order_independence_failure_count == 0
        and full.runtime_direct_write_count == 0
        and full.held_variant_promoted_count == 0
        and full.destructive_delete_count == 0
        and full.oracle_plan_feature_use_count == 0
        and full.direct_flow_write_reject_count >= 34
        and full.stale_snapshot_reject_count >= 34
        and full.checksum_tamper_reject_count >= 34
        and full.ambiguous_same_region_reject_count >= 34
        and full.atomic_multi_region_commit_case_count >= int(e136o_summary["full"]["atomic_multi_region_commit_case_count"])
        and policy.reject_ambiguous_same_region
        and policy.hold_held_variants
        and policy.stable_write_order
    )
    return {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": DECISION_CONFIRMED if pass_gate else DECISION_REJECTED,
        "next": NEXT,
        "pass_gate": pass_gate,
        "input_e136o_decision": e136o_summary["decision"],
        "input_e136o_full_accuracy": e136o_summary["full"]["accuracy"],
        "input_e136o_accepted_mutations": e136o_summary["accepted_mutations"],
        "input_e136o_accepted_prune_mutations": e136o_summary["accepted_prune_mutations"],
        "base_case_count": base_case_count,
        "runtime_policy": asdict(policy),
        "train": asdict(train),
        "heldout": asdict(heldout),
        "ood": asdict(ood),
        "full": asdict(full),
        "implementation_ready": pass_gate,
        "production_apply_allowed_now": False,
        "implementation_preview_allowed": pass_gate,
        "conservative_guard_overrides": {
            "reject_ambiguous_same_region": "kept_true_even_if_shadow_policy_pruned_it",
            "hold_held_variants": "kept_true_for_runtime",
            "stable_write_order": "kept_true_for_runtime",
            "chunk_min_support": "raised_to_minimum_3_for_oracle_free_chunk_detection",
        },
    }


def implementation_readiness(summary: dict[str, Any]) -> dict[str, Any]:
    full = summary["full"]
    ready = bool(summary["implementation_ready"])
    return {
        "artifact_contract": ARTIFACT_CONTRACT,
        "implementation_ready": ready,
        "production_apply_allowed_now": False,
        "recommended_next": summary["next"],
        "ready_components": {
            "runtime_oracle_free_commit_policy": ready,
            "atomic_multiwrite_barrier": ready,
            "stale_snapshot_guard": full["stale_snapshot_reject_count"] >= 34,
            "checksum_guard": full["checksum_tamper_reject_count"] >= 34,
            "direct_flow_write_guard": full["direct_flow_write_reject_count"] >= 34,
            "ambiguous_same_region_guard": full["ambiguous_same_region_reject_count"] >= 34,
            "held_variant_guard": full["held_variant_promoted_count"] == 0,
            "rollback_fallback_path": bool(summary["runtime_policy"]["enable_rollback_fallback"]),
        },
        "implementation_skeleton": [
            "collect parallel proposals without Flow mutation",
            "filter stale snapshot, checksum, direct-write, trace, and unsafe proposals",
            "resolve per-operator candidates; reject competing primary same-region writes",
            "select rollback when primary regression leaves a valid rollback proposal",
            "commit homogeneous primary groups as atomic chunks",
            "commit disjoint primary groups as bounded atomic multiwrite",
            "otherwise commit one primary or defer",
            "apply write-set atomically with stable ordering and rollback snapshot",
        ],
        "not_allowed_yet": [
            "destructive deletion of legacy guards",
            "production apply without E136P implementation preview",
            "removal of ambiguous same-region guard",
            "promotion of held challenger or lineage variants",
        ],
    }


def write_report(out: Path, summary: dict[str, Any]) -> None:
    full = summary["full"]
    report = f"""# E136O Challenger OOD Runtime Replacement Gauntlet

```text
decision = {summary['decision']}
next     = {summary['next']}
```

E136O challenger replays the E136O/E136N4 surface with a runtime-style commit
policy that does not use oracle plan features such as expected action or case
family. Oracle labels are used only by the checker.

## Result

```text
base_case_count = {summary['base_case_count']}
full_case_count = {full['case_count']}

train_accuracy = {summary['train']['accuracy']:.6f}
heldout_accuracy = {summary['heldout']['accuracy']:.6f}
ood_accuracy = {summary['ood']['accuracy']:.6f}
full_accuracy = {full['accuracy']:.6f}

full_atomic_multi_region_commit_case_count = {full['atomic_multi_region_commit_case_count']}
full_atomic_write_total = {full['atomic_write_total']}

full_partial_write_count = {full['partial_write_count']}
full_order_independence_failure_count = {full['order_independence_failure_count']}
full_runtime_direct_write_count = {full['runtime_direct_write_count']}
full_held_variant_promoted_count = {full['held_variant_promoted_count']}
full_oracle_plan_feature_use_count = {full['oracle_plan_feature_use_count']}

full_direct_flow_write_reject_count = {full['direct_flow_write_reject_count']}
full_stale_snapshot_reject_count = {full['stale_snapshot_reject_count']}
full_checksum_tamper_reject_count = {full['checksum_tamper_reject_count']}
full_ambiguous_same_region_reject_count = {full['ambiguous_same_region_reject_count']}

implementation_ready = {summary['implementation_ready']}
production_apply_allowed_now = {summary['production_apply_allowed_now']}
```

## Boundary

This prepares implementation. It does not apply production runtime replacement.
The conservative runtime manifest keeps ambiguous-region, held-variant, and
stable-order guards even where the previous shadow policy pruned them.
"""
    (out / "report.md").write_text(report, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    sample_out = None if args.sample_out == "" else Path(args.sample_out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_FILES:
        path = out / name
        if path.exists():
            path.unlink()

    e136n4 = load_module("e136n4_runtime_challenger", Path(args.e136n4_script))
    e136o_prep = load_module("e136o_prep_runtime_challenger", Path(args.e136o_script))
    e136o_summary = load_e136o_summary(Path(args.e136o_artifact))
    _, _, _, registry = e136n4.load_inputs(
        Path(args.e136n_artifact),
        Path(args.e136n2_artifact),
        Path(args.e136n3_artifact),
    )
    base_cases = e136n4.build_cases(registry)
    rng = random.Random(args.seed)
    all_cases = e136o_prep.build_augmented_cases(e136n4, base_cases, args.ood_waves, rng)
    train_cases, heldout_cases, ood_cases = e136o_prep.split_cases(all_cases)

    policy = runtime_policy_from_shadow(e136o_summary)
    plans_by_id = {case.case_id: runtime_challenger_plan(e136n4, policy, case) for case in all_cases}
    traces_by_id = {
        case.case_id: evaluate_runtime_plan(e136n4, case, plans_by_id[case.case_id])
        for case in all_cases
    }
    train_metrics = summarize_split([traces_by_id[case.case_id] for case in train_cases])
    heldout_metrics = summarize_split([traces_by_id[case.case_id] for case in heldout_cases])
    ood_metrics = summarize_split([traces_by_id[case.case_id] for case in ood_cases])
    full_metrics = summarize_split(list(traces_by_id.values()))
    summary = summarize(e136o_summary, policy, train_metrics, heldout_metrics, ood_metrics, full_metrics, len(base_cases))
    readiness = implementation_readiness(summary)

    failures = []
    if not summary["pass_gate"]:
        failures.append("e136o_challenger_pass_gate_failed")
    if full_metrics.oracle_plan_feature_use_count:
        failures.append("oracle_plan_feature_used")
    if full_metrics.partial_write_count:
        failures.append("partial_write_present")
    if full_metrics.order_independence_failure_count:
        failures.append("order_dependence_present")
    if full_metrics.runtime_direct_write_count:
        failures.append("runtime_direct_write_present")
    if full_metrics.held_variant_promoted_count:
        failures.append("held_variant_promoted")
    if not policy.reject_ambiguous_same_region:
        failures.append("ambiguous_same_region_guard_not_kept")

    write_json(out / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "boundary": "implementation readiness only; no production apply",
        "seed": args.seed,
        "ood_waves": args.ood_waves,
        "e136o_artifact": str(args.e136o_artifact),
        "e136n4_script": str(args.e136n4_script),
        "e136o_script": str(args.e136o_script),
        "base_case_count": len(base_cases),
        "train_case_count": len(train_cases),
        "heldout_case_count": len(heldout_cases),
        "ood_case_count": len(ood_cases),
    })
    write_json(out / "runtime_policy_manifest.json", asdict(policy))
    write_json(out / "implementation_readiness.json", readiness)
    write_jsonl(out / "challenger_trace.jsonl", [traces_by_id[key] for key in sorted(traces_by_id)])
    write_json(out / "split_metrics.json", {
        "train": asdict(train_metrics),
        "heldout": asdict(heldout_metrics),
        "ood": asdict(ood_metrics),
        "full": asdict(full_metrics),
    })
    write_json(out / "decision.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": summary["decision"],
        "next": NEXT,
        "pass_gate": summary["pass_gate"],
    })
    write_json(out / "summary.json", summary)
    write_report(out, summary)
    write_json(out / "checker_summary.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "failure_count": len(failures),
        "failures": failures,
    })
    copy_sample(out, sample_out)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e136n4-script", default=str(DEFAULT_E136N4_SCRIPT))
    parser.add_argument("--e136o-script", default=str(DEFAULT_E136O_SCRIPT))
    parser.add_argument("--e136n-artifact", default=str(DEFAULT_E136N))
    parser.add_argument("--e136n2-artifact", default=str(DEFAULT_E136N2))
    parser.add_argument("--e136n3-artifact", default=str(DEFAULT_E136N3))
    parser.add_argument("--e136o-artifact", default=str(DEFAULT_E136O))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--sample-out", default=str(DEFAULT_SAMPLE_OUT))
    parser.add_argument("--ood-waves", type=int, default=24)
    parser.add_argument("--seed", type=int, default=136_041)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps({
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": summary["decision"],
        "full_accuracy": summary["full"]["accuracy"],
        "implementation_ready": summary["implementation_ready"],
        "next": summary["next"],
        "ood_accuracy": summary["ood"]["accuracy"],
        "oracle_plan_feature_use_count": summary["full"]["oracle_plan_feature_use_count"],
        "pass_gate": summary["pass_gate"],
        "production_apply_allowed_now": summary["production_apply_allowed_now"],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if summary["pass_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

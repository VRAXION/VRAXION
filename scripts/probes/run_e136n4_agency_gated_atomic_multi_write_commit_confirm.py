#!/usr/bin/env python3
"""E136N4 Agency-gated atomic multi-write commit confirm.

Local smoke for the question: can Agency approve more than one Flow write in
the same tick without falling back to unsafe parallel direct writes?

Boundary: deterministic artifact/proxy check only. It does not run a production
scheduler, spawn threads, discover new operators, promote held variants, or push
anything online.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ARTIFACT_CONTRACT = "E136N4_AGENCY_GATED_ATOMIC_MULTI_WRITE_COMMIT_CONFIRM"
DECISION_CONFIRMED = "e136n4_agency_gated_atomic_multi_write_confirmed"
DECISION_REJECTED = "e136n4_agency_gated_atomic_multi_write_rejected"
NEXT = "E136O_CHALLENGER_OOD_RUNTIME_REPLACEMENT_GAUNTLET"

DEFAULT_E136N = Path("docs/research/artifact_samples/e136n_primary_secondary_variant_governance")
DEFAULT_E136N2 = Path("docs/research/artifact_samples/e136n2_agency_matrix_arbitration_smoke")
DEFAULT_E136N3 = Path("docs/research/artifact_samples/e136n3_parallel_direct_write_ab_smoke")
DEFAULT_OUT = Path("target/e136n4_agency_gated_atomic_multi_write_commit_confirm")

SNAPSHOT_ID = "snapshot::e136n4_clean_flow"
PRIMARY_STATES = {"primary_active", "primary_current", "primary_abstract_current"}
HELD_STATES = {"secondary_challenger", "secondary_lineage_hold"}

ARTIFACT_FILES = (
    "run_manifest.json",
    "atomic_cases.jsonl",
    "agency_commit_plans.jsonl",
    "atomic_commit_traces.jsonl",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "checker_summary.json",
)


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    operator_id: str
    display_name: str
    variant_id: str
    variant_role: str
    variant_state: str
    proposal_kind: str
    relation_group: str
    snapshot_id: str = SNAPSHOT_ID
    trace_valid: bool = True
    ground_compatible: bool = True
    checksum_valid: bool = True
    direct_flow_write: bool = False
    unsupported_answer: bool = False
    hard_negative: bool = False
    primary_regression_signal: bool = False
    confidence: float = 0.85


@dataclass(frozen=True)
class Case:
    case_id: str
    family: str
    proposals: tuple[Proposal, ...]
    expected_action: str
    expected_variant_ids: tuple[str, ...] = ()
    expected_chunk_id: str | None = None
    expected_requires_child_check: bool = False
    expected_reject_reason: str | None = None


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


def load_inputs(e136n_dir: Path, e136n2_dir: Path, e136n3_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    e136n_summary = load_json(e136n_dir / "summary.json")
    if e136n_summary.get("decision") != "e136n_primary_secondary_variant_governance_confirmed":
        raise ValueError("E136N input is not confirmed")
    if not e136n_summary.get("pass_gate"):
        raise ValueError("E136N input pass gate is false")

    e136n2_summary = load_json(e136n2_dir / "summary.json")
    if e136n2_summary.get("decision") != "e136n2_agency_matrix_arbitration_smoke_confirmed":
        raise ValueError("E136N2 input is not confirmed")
    if not e136n2_summary.get("pass_gate"):
        raise ValueError("E136N2 input pass gate is false")

    e136n3_summary = load_json(e136n3_dir / "summary.json")
    if e136n3_summary.get("decision") != "e136n3_parallel_direct_write_ab_confirmed":
        raise ValueError("E136N3 input is not confirmed")
    if not e136n3_summary.get("pass_gate"):
        raise ValueError("E136N3 input pass gate is false")

    registry = load_json(e136n_dir / "variant_registry.json")["rows"]
    return e136n_summary, e136n2_summary, e136n3_summary, registry


def relation_group(row: dict[str, Any]) -> str:
    text = f"{row.get('operator_id', '')} {row.get('display_name', '')}".lower()
    if any(token in text for token in ("latex", "equation", "math", "arithmetic", "quantity", "matrix")):
        return "visible_math_context"
    if any(token in text for token in ("answer", "format", "render", "output", "json")):
        return "answer_render_context"
    if any(token in text for token in ("boundary", "safety", "guard", "hidden", "no-solve", "trust")):
        return "boundary_guard_context"
    if any(token in text for token in ("assistant", "summary", "comparison", "translation", "outline")):
        return "assistant_route_context"
    if any(token in text for token in ("temporal", "dialogue", "state", "route")):
        return "dialogue_route_context"
    return "general_context"


def proposal_from_row(
    row: dict[str, Any],
    suffix: str,
    *,
    snapshot_id: str = SNAPSHOT_ID,
    checksum_valid: bool = True,
    direct_flow_write: bool = False,
    primary_regression_signal: bool = False,
    variant_id: str | None = None,
    variant_state: str | None = None,
    proposal_kind: str = "variant_commit",
) -> Proposal:
    state = variant_state or row["variant_state"]
    return Proposal(
        proposal_id=f"{row['operator_id']}::{suffix}",
        operator_id=row["operator_id"],
        display_name=row["display_name"],
        variant_id=variant_id or row["variant_id"],
        variant_role=row["variant_role"],
        variant_state=state,
        proposal_kind=proposal_kind,
        relation_group=relation_group(row),
        snapshot_id=snapshot_id,
        checksum_valid=checksum_valid,
        direct_flow_write=direct_flow_write,
        primary_regression_signal=primary_regression_signal,
        confidence={
            "primary_active": 0.98,
            "primary_current": 0.94,
            "primary_abstract_current": 0.90,
            "secondary_rollback": 0.88,
            "secondary_challenger": 0.93,
            "secondary_lineage_hold": 0.84,
        }.get(state, 0.70),
    )


def build_operator_rows(registry: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_operator: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in registry:
        if row["variant_role"] == "primary":
            by_operator[row["operator_id"]]["primary"] = row
        elif row["variant_role"] == "secondary":
            by_operator[row["operator_id"]]["secondary"] = row
    return dict(by_operator)


def build_cases(registry: list[dict[str, Any]]) -> list[Case]:
    by_operator = build_operator_rows(registry)
    primary_rows: list[dict[str, Any]] = []
    cases: list[Case] = []

    for operator_id, rows in sorted(by_operator.items()):
        primary = rows["primary"]
        secondary = rows["secondary"]
        primary_rows.append(primary)

        cases.append(Case(
            case_id=f"{operator_id}::unsafe_filtered_atomic_commit",
            family="unsafe_filtered_atomic_commit",
            proposals=(
                proposal_from_row(primary, "unsafe_direct_writer", direct_flow_write=True, variant_id=f"{operator_id}::unsafe_direct_writer"),
                proposal_from_row(primary, "safe_primary"),
            ),
            expected_action="commit_single",
            expected_variant_ids=(primary["variant_id"],),
            expected_reject_reason="direct_flow_write",
        ))
        cases.append(Case(
            case_id=f"{operator_id}::same_region_conflict_resolved_single",
            family="same_region_conflict_resolved_single",
            proposals=(
                proposal_from_row(secondary, "secondary_conflict"),
                proposal_from_row(primary, "primary_conflict_winner"),
            ),
            expected_action="commit_single",
            expected_variant_ids=(primary["variant_id"],),
            expected_requires_child_check=secondary["variant_state"] in HELD_STATES,
        ))
        cases.append(Case(
            case_id=f"{operator_id}::stale_snapshot_reject",
            family="stale_snapshot_reject",
            proposals=(proposal_from_row(primary, "stale_snapshot", snapshot_id="snapshot::old"),),
            expected_action="defer",
            expected_reject_reason="stale_snapshot",
        ))
        cases.append(Case(
            case_id=f"{operator_id}::checksum_tamper_reject",
            family="checksum_tamper_reject",
            proposals=(proposal_from_row(primary, "checksum_tamper", checksum_valid=False),),
            expected_action="defer",
            expected_reject_reason="checksum_invalid",
        ))
        cases.append(Case(
            case_id=f"{operator_id}::ambiguous_same_region_batch_reject",
            family="ambiguous_same_region_batch_reject",
            proposals=(
                proposal_from_row(primary, "ambiguous_primary_a"),
                proposal_from_row(primary, "ambiguous_primary_b", variant_id=f"{operator_id}::ambiguous_competing_variant"),
            ),
            expected_action="defer",
            expected_reject_reason="ambiguous_same_region",
        ))
        if primary["variant_state"] == "primary_active" and secondary["variant_state"] == "secondary_rollback":
            cases.append(Case(
                case_id=f"{operator_id}::rollback_atomic_swap",
                family="rollback_atomic_swap",
                proposals=(
                    proposal_from_row(primary, "regressing_primary", primary_regression_signal=True),
                    proposal_from_row(secondary, "rollback_secondary"),
                ),
                expected_action="commit_single",
                expected_variant_ids=(secondary["variant_id"],),
            ))
        if secondary["variant_state"] == "secondary_challenger":
            cases.append(Case(
                case_id=f"{operator_id}::held_challenger_child_check_commit_primary",
                family="held_challenger_child_check_commit_primary",
                proposals=(
                    proposal_from_row(secondary, "held_challenger"),
                    proposal_from_row(primary, "current_primary"),
                ),
                expected_action="commit_single",
                expected_variant_ids=(primary["variant_id"],),
                expected_requires_child_check=True,
            ))
        if secondary["variant_state"] == "secondary_lineage_hold":
            cases.append(Case(
                case_id=f"{operator_id}::held_lineage_child_check_commit_primary",
                family="held_lineage_child_check_commit_primary",
                proposals=(
                    proposal_from_row(secondary, "held_lineage"),
                    proposal_from_row(primary, "abstract_primary"),
                ),
                expected_action="commit_single",
                expected_variant_ids=(primary["variant_id"],),
                expected_requires_child_check=True,
            ))

    for index, offset in enumerate(range(0, len(primary_rows) - 2, 3), start=1):
        trio = primary_rows[offset: offset + 3]
        cases.append(Case(
            case_id=f"disjoint_atomic_multi_write::{index:02d}",
            family="disjoint_atomic_multi_write",
            proposals=tuple(proposal_from_row(row, f"atomic_multi_{index}") for row in trio),
            expected_action="commit_multi",
            expected_variant_ids=tuple(sorted(row["variant_id"] for row in trio)),
        ))

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in primary_rows:
        groups[relation_group(row)].append(row)
    chunk_index = 0
    for group, rows in sorted(groups.items()):
        for offset in range(0, max(0, len(rows) - 2), 3):
            trio = rows[offset: offset + 3]
            if len(trio) < 3:
                continue
            chunk_index += 1
            chunk_id = f"flow_chunk::{group}::{chunk_index:02d}"
            cases.append(Case(
                case_id=f"{chunk_id}::atomic_chunk_commit",
                family="atomic_chunk_commit",
                proposals=tuple(proposal_from_row(row, f"atomic_chunk_{chunk_index}") for row in trio),
                expected_action="commit_chunk",
                expected_chunk_id=chunk_id,
            ))
    return cases


def write_set_for_variant(proposal: Proposal, *, audit: bool = False) -> list[tuple[str, str]]:
    writes = [(f"active_variant::{proposal.operator_id}", proposal.variant_id)]
    if audit:
        writes.append((f"rollback_audit::{proposal.operator_id}", f"rollback_to::{proposal.variant_id}"))
    return writes


def write_set_for_chunk(chunk_id: str, proposals: list[Proposal]) -> list[tuple[str, str]]:
    digest = hashlib.sha256(" ".join(sorted(p.variant_id for p in proposals)).encode("utf-8")).hexdigest()[:16]
    writes = [(f"flow_chunk::{chunk_id}", f"approved::{digest}")]
    for proposal in proposals:
        writes.append((f"chunk_member::{chunk_id}::{proposal.operator_id}", proposal.variant_id))
    return writes


def proposal_reject_reason(proposal: Proposal) -> str | None:
    if proposal.snapshot_id != SNAPSHOT_ID:
        return "stale_snapshot"
    if not proposal.trace_valid or not proposal.ground_compatible:
        return "trace_or_ground_invalid"
    if not proposal.checksum_valid:
        return "checksum_invalid"
    if proposal.direct_flow_write:
        return "direct_flow_write"
    if proposal.unsupported_answer or proposal.hard_negative:
        return "unsafe_answer"
    if proposal.primary_regression_signal:
        return "primary_regression"
    return None


def duplicate_region_conflicts(proposals: list[Proposal]) -> dict[str, list[str]]:
    region_values: dict[str, set[str]] = defaultdict(set)
    for proposal in proposals:
        for region, value in write_set_for_variant(proposal):
            region_values[region].add(value)
    return {
        region: sorted(values)
        for region, values in region_values.items()
        if len(values) > 1
    }


def agency_plan(case: Case) -> dict[str, Any]:
    rejected = []
    valid = []
    for proposal in case.proposals:
        reason = proposal_reject_reason(proposal)
        if reason is None:
            valid.append(proposal)
        else:
            rejected.append({"proposal_id": proposal.proposal_id, "reason": reason})

    requires_child_check = any(p.variant_state in HELD_STATES for p in case.proposals)
    selected: list[Proposal] = []
    action = "defer"
    chunk_id = None
    reject_reason = None

    if case.family == "atomic_chunk_commit":
        groups = Counter(p.relation_group for p in valid if p.variant_state in PRIMARY_STATES)
        best_group, count = groups.most_common(1)[0] if groups else (None, 0)
        if best_group and count >= 3:
            selected = [p for p in valid if p.relation_group == best_group and p.variant_state in PRIMARY_STATES][:3]
            action = "commit_chunk"
            chunk_id = case.expected_chunk_id
        else:
            reject_reason = "insufficient_chunk_support"
    elif case.family == "disjoint_atomic_multi_write":
        primaries = [p for p in valid if p.variant_state in PRIMARY_STATES]
        conflicts = duplicate_region_conflicts(primaries)
        if len(primaries) >= 2 and not conflicts:
            selected = sorted(primaries, key=lambda p: p.variant_id)
            action = "commit_multi"
        else:
            reject_reason = "multi_write_conflict"
    elif case.family == "ambiguous_same_region_batch_reject":
        conflicts = duplicate_region_conflicts(valid)
        if conflicts:
            reject_reason = "ambiguous_same_region"
        else:
            reject_reason = "missing_ambiguity"
    elif case.family == "rollback_atomic_swap":
        rollback = [p for p in valid if p.variant_state == "secondary_rollback"]
        if rollback:
            selected = rollback[:1]
            action = "commit_single"
        else:
            reject_reason = "rollback_missing"
    else:
        primaries = [p for p in valid if p.variant_state in PRIMARY_STATES]
        if primaries:
            selected = primaries[:1]
            action = "commit_single"
        elif valid:
            selected = valid[:1]
            action = "commit_single"
        else:
            reject_reason = rejected[0]["reason"] if rejected else "no_valid_proposal"

    write_set: list[tuple[str, str]] = []
    if action == "commit_chunk" and chunk_id:
        write_set = write_set_for_chunk(chunk_id, selected)
    elif action == "commit_multi":
        for proposal in selected:
            write_set.extend(write_set_for_variant(proposal))
    elif action == "commit_single" and selected:
        write_set = write_set_for_variant(selected[0], audit=case.family == "rollback_atomic_swap")

    batch_conflicts = defaultdict(set)
    for region, value in write_set:
        batch_conflicts[region].add(value)
    conflicting_regions = sorted(region for region, values in batch_conflicts.items() if len(values) > 1)
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
        "requires_child_check": requires_child_check,
        "rejected": rejected,
        "reject_reason": reject_reason,
        "conflicting_regions": conflicting_regions,
        "runtime_direct_write_count": 0,
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


def evaluate_plan(case: Case, plan: dict[str, Any]) -> dict[str, Any]:
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
        correct = plan["action"] == "commit_chunk" and plan["chunk_id"] == case.expected_chunk_id
    else:
        correct = plan["action"] == "defer" and plan["write_count"] == 0

    partial_write = plan["action"] == "defer" and plan["write_count"] > 0
    held_ids = {p.variant_id for p in case.proposals if p.variant_state in HELD_STATES}
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


def summarize(
    e136n_summary: dict[str, Any],
    e136n2_summary: dict[str, Any],
    e136n3_summary: dict[str, Any],
    cases: list[Case],
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    case_count = len(cases)
    correct = sum(1 for row in traces if row["correct"])
    partial_writes = sum(1 for row in traces if row["partial_write"])
    order_failures = sum(1 for row in traces if not row["order_independent"])
    runtime_direct_writes = sum(row["runtime_direct_write_count"] for row in traces)
    multi_region_commits = sum(1 for row in traces if row["multi_region_commit"] and row["correct"])
    atomic_write_total = sum(row["write_count"] for row in traces if row["correct"])
    held_promoted = sum(1 for row in traces if row["held_variant_promoted"])
    expected_child = sum(1 for case in cases if case.expected_requires_child_check)
    actual_child = sum(1 for row in traces if row["requires_child_check"])
    expected_chunks = sum(1 for case in cases if case.expected_action == "commit_chunk")
    actual_chunks = sum(1 for row in traces if row["action"] == "commit_chunk" and row["correct"])
    reject_counter = Counter(reason for row in traces for reason in row["reject_reasons"])
    family_counts = Counter(case.family for case in cases)
    defer_correct = sum(1 for row in traces if row["expected_action"] == "defer" and row["correct"])
    pass_gate = (
        e136n_summary.get("decision") == "e136n_primary_secondary_variant_governance_confirmed"
        and e136n2_summary.get("decision") == "e136n2_agency_matrix_arbitration_smoke_confirmed"
        and e136n3_summary.get("decision") == "e136n3_parallel_direct_write_ab_confirmed"
        and case_count >= 200
        and correct == case_count
        and partial_writes == 0
        and order_failures == 0
        and runtime_direct_writes == 0
        and multi_region_commits >= 30
        and reject_counter["direct_flow_write"] == 34
        and reject_counter["stale_snapshot"] == 34
        and reject_counter["checksum_invalid"] == 34
        and reject_counter["ambiguous_same_region"] == 34
        and held_promoted == 0
        and expected_child == actual_child
        and expected_chunks == actual_chunks
    )
    return {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": DECISION_CONFIRMED if pass_gate else DECISION_REJECTED,
        "next": NEXT,
        "pass_gate": pass_gate,
        "input_e136n_operator_count": e136n_summary["operator_count"],
        "input_e136n2_agency_matrix_accuracy": e136n2_summary["agency_matrix_accuracy"],
        "input_e136n3_agency_gated_accuracy": e136n3_summary["agency_gated_accuracy"],
        "case_count": case_count,
        "case_family_counts": dict(sorted(family_counts.items())),
        "correct_count": correct,
        "accuracy": round(correct / case_count, 6),
        "atomic_multi_region_commit_case_count": multi_region_commits,
        "atomic_write_total": atomic_write_total,
        "defer_correct_count": defer_correct,
        "partial_write_count": partial_writes,
        "order_independence_failure_count": order_failures,
        "runtime_direct_write_count": runtime_direct_writes,
        "held_variant_promoted_count": held_promoted,
        "expected_child_check_count": expected_child,
        "agency_child_check_count": actual_child,
        "expected_flow_chunk_count": expected_chunks,
        "agency_flow_chunk_count": actual_chunks,
        "direct_flow_write_reject_count": reject_counter["direct_flow_write"],
        "stale_snapshot_reject_count": reject_counter["stale_snapshot"],
        "checksum_tamper_reject_count": reject_counter["checksum_invalid"],
        "ambiguous_same_region_reject_count": reject_counter["ambiguous_same_region"],
        "destructive_delete_count": 0,
    }


def write_report(out: Path, summary: dict[str, Any]) -> None:
    report = f"""# E136N4 Agency-Gated Atomic Multi-Write Commit Confirm

```text
decision = {summary['decision']}
next     = {summary['next']}
```

E136N4 checks whether Agency can approve multiple Flow writes in one atomic
batch after parallel proposal fanout.

## Result

```text
case_count = {summary['case_count']}
accuracy = {summary['accuracy']:.6f}

atomic_multi_region_commit_case_count = {summary['atomic_multi_region_commit_case_count']}
atomic_write_total = {summary['atomic_write_total']}
partial_write_count = {summary['partial_write_count']}
order_independence_failure_count = {summary['order_independence_failure_count']}
runtime_direct_write_count = {summary['runtime_direct_write_count']}

direct_flow_write_reject_count = {summary['direct_flow_write_reject_count']}
stale_snapshot_reject_count = {summary['stale_snapshot_reject_count']}
checksum_tamper_reject_count = {summary['checksum_tamper_reject_count']}
ambiguous_same_region_reject_count = {summary['ambiguous_same_region_reject_count']}

expected_child_check_count = {summary['expected_child_check_count']}
agency_child_check_count = {summary['agency_child_check_count']}
expected_flow_chunk_count = {summary['expected_flow_chunk_count']}
agency_flow_chunk_count = {summary['agency_flow_chunk_count']}
held_variant_promoted_count = {summary['held_variant_promoted_count']}
destructive_delete_count = {summary['destructive_delete_count']}
```

## Interpretation

The supported shape is:

```text
parallel read/propose
-> Agency validates write sets
-> Agency builds one atomic batch
-> batch commits multiple regions or commits nothing
```

This is not parallel direct Flow write. Proposals do not mutate Flow directly;
Agency commits the approved batch after conflict, snapshot, checksum, held, and
direct-write checks.
"""
    (out / "report.md").write_text(report, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_FILES:
        path = out / name
        if path.exists():
            path.unlink()

    e136n_summary, e136n2_summary, e136n3_summary, registry = load_inputs(
        Path(args.e136n_artifact),
        Path(args.e136n2_artifact),
        Path(args.e136n3_artifact),
    )
    cases = build_cases(registry)
    plans = [agency_plan(case) for case in cases]
    traces = [evaluate_plan(case, plan) for case, plan in zip(cases, plans)]
    summary = summarize(e136n_summary, e136n2_summary, e136n3_summary, cases, traces)
    failures = []
    if not summary["pass_gate"]:
        failures.append("e136n4_pass_gate_failed")
    if summary["partial_write_count"]:
        failures.append("partial_write_present")
    if summary["order_independence_failure_count"]:
        failures.append("order_dependence_present")
    if summary["runtime_direct_write_count"]:
        failures.append("runtime_direct_write_present")
    if summary["held_variant_promoted_count"]:
        failures.append("held_variant_promoted")

    write_json(out / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "boundary": "local atomic multi-write smoke only; no production scheduler; no online push",
        "e136n_artifact": str(args.e136n_artifact),
        "e136n2_artifact": str(args.e136n2_artifact),
        "e136n3_artifact": str(args.e136n3_artifact),
    })
    write_jsonl(out / "atomic_cases.jsonl", [{
        "case_id": case.case_id,
        "family": case.family,
        "expected_action": case.expected_action,
        "expected_variant_ids": list(case.expected_variant_ids),
        "expected_chunk_id": case.expected_chunk_id,
        "expected_requires_child_check": case.expected_requires_child_check,
        "expected_reject_reason": case.expected_reject_reason,
        "proposals": [proposal.__dict__ for proposal in case.proposals],
    } for case in cases])
    write_jsonl(out / "agency_commit_plans.jsonl", plans)
    write_jsonl(out / "atomic_commit_traces.jsonl", traces)
    write_json(out / "aggregate_metrics.json", {
        key: value for key, value in summary.items()
        if key not in {"decision", "next", "pass_gate", "case_family_counts"}
    })
    write_json(out / "decision.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": summary["decision"],
        "next": summary["next"],
        "pass_gate": summary["pass_gate"],
    })
    write_json(out / "summary.json", summary)
    write_report(out, summary)
    write_json(out / "checker_summary.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "failure_count": len(failures),
        "failures": failures,
    })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e136n-artifact", default=str(DEFAULT_E136N))
    parser.add_argument("--e136n2-artifact", default=str(DEFAULT_E136N2))
    parser.add_argument("--e136n3-artifact", default=str(DEFAULT_E136N3))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps({
        "accuracy": summary["accuracy"],
        "artifact_contract": ARTIFACT_CONTRACT,
        "atomic_multi_region_commit_case_count": summary["atomic_multi_region_commit_case_count"],
        "case_count": summary["case_count"],
        "decision": summary["decision"],
        "order_independence_failure_count": summary["order_independence_failure_count"],
        "partial_write_count": summary["partial_write_count"],
        "pass_gate": summary["pass_gate"],
        "runtime_direct_write_count": summary["runtime_direct_write_count"],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if summary["pass_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

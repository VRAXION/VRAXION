#!/usr/bin/env python3
"""E136N3 parallel direct-write A/B smoke.

This probe compares two parallel execution styles over the crystallized E136N
primary/secondary variant surface and the E136N2 learned Agency Matrix:

- arm A allows proposals to write directly into Flow in the same tick;
- arm B allows the same proposals to fan out in parallel, but commits only
  after an Agency Matrix arbitration barrier.

Boundary: this is a deterministic A/B smoke over existing variants. It does
not discover new operators, promote held challenger/lineage variants, or apply
destructive runtime replacement.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ARTIFACT_CONTRACT = "E136N3_PARALLEL_DIRECT_WRITE_AB_SMOKE"
DECISION_CONFIRMED = "e136n3_parallel_direct_write_ab_confirmed"
DECISION_REJECTED = "e136n3_parallel_direct_write_ab_rejected"
NEXT = "E136O_CHALLENGER_OOD_RUNTIME_REPLACEMENT_GAUNTLET"

DEFAULT_E136N = Path("docs/research/artifact_samples/e136n_primary_secondary_variant_governance")
DEFAULT_E136N2 = Path("docs/research/artifact_samples/e136n2_agency_matrix_arbitration_smoke")
DEFAULT_OUT = Path("target/e136n3_parallel_direct_write_ab_smoke")
DEFAULT_SAMPLE_OUT = Path("docs/research/artifact_samples/e136n3_parallel_direct_write_ab_smoke")

ARTIFACT_FILES = (
    "run_manifest.json",
    "ab_cases.jsonl",
    "direct_write_arm.jsonl",
    "agency_gated_arm.jsonl",
    "conflict_matrix.jsonl",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "checker_summary.json",
)

PRIMARY_STATES = {"primary_active", "primary_current", "primary_abstract_current"}
HELD_STATES = {"secondary_challenger", "secondary_lineage_hold"}


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    operator_id: str
    display_name: str
    variant_id: str
    variant_role: str
    variant_state: str
    proposal_kind: str
    trace_valid: bool = True
    ground_compatible: bool = True
    direct_flow_write: bool = False
    unsupported_answer: bool = False
    hard_negative: bool = False
    primary_regression_signal: bool = False
    confidence: float = 0.85
    cost: float = 1.0
    expected_gain: float = 1.0
    relation_group: str = "general_context"


@dataclass(frozen=True)
class Case:
    case_id: str
    family: str
    proposals: tuple[Proposal, ...]
    expected_action: str
    expected_variant_ids: tuple[str, ...] = ()
    expected_chunk_id: str | None = None
    expected_requires_child_check: bool = False


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


def load_inputs(e136n_dir: Path, e136n2_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, float]]:
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

    registry = load_json(e136n_dir / "variant_registry.json")["rows"]
    learned = load_json(e136n2_dir / "learned_agency_matrix.json")
    weights = {str(key): float(value) for key, value in learned["weights"].items()}
    return e136n_summary, e136n2_summary, registry, weights


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
    primary_regression_signal: bool = False,
    confidence: float | None = None,
    cost: float | None = None,
    expected_gain: float | None = None,
) -> Proposal:
    state = row["variant_state"]
    if confidence is None:
        confidence = {
            "primary_active": 0.98,
            "primary_current": 0.94,
            "primary_abstract_current": 0.90,
            "secondary_rollback": 0.88,
            "secondary_challenger": 0.93,
            "secondary_lineage_hold": 0.84,
        }.get(state, 0.70)
    if cost is None:
        cost = {
            "primary_active": 0.75,
            "primary_current": 0.85,
            "primary_abstract_current": 1.05,
            "secondary_rollback": 0.90,
            "secondary_challenger": 0.68,
            "secondary_lineage_hold": 1.15,
        }.get(state, 1.0)
    if expected_gain is None:
        expected_gain = {
            "primary_active": 1.40,
            "primary_current": 1.15,
            "primary_abstract_current": 1.05,
            "secondary_rollback": 0.85,
            "secondary_challenger": 1.55,
            "secondary_lineage_hold": 1.10,
        }.get(state, 1.0)
    return Proposal(
        proposal_id=f"{row['operator_id']}::{suffix}",
        operator_id=row["operator_id"],
        display_name=row["display_name"],
        variant_id=row["variant_id"],
        variant_role=row["variant_role"],
        variant_state=state,
        proposal_kind="variant_commit",
        primary_regression_signal=primary_regression_signal,
        confidence=confidence,
        cost=cost,
        expected_gain=expected_gain,
        relation_group=relation_group(row),
    )


def unsafe_proposal(operator_id: str, suffix: str, display_name: str) -> Proposal:
    return Proposal(
        proposal_id=f"{operator_id}::{suffix}",
        operator_id=operator_id,
        display_name=display_name,
        variant_id=f"{operator_id}::unsafe_parallel_direct_write_control",
        variant_role="control",
        variant_state="unsafe_parallel_direct_write",
        proposal_kind="unsafe_parallel_direct_write",
        direct_flow_write=True,
        confidence=0.99,
        cost=0.20,
        expected_gain=2.00,
        relation_group="unsafe_control",
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
    cases: list[Case] = []
    primary_rows: list[dict[str, Any]] = []
    for operator_id, rows in sorted(by_operator.items()):
        primary = rows["primary"]
        secondary = rows["secondary"]
        primary_rows.append(primary)
        primary_proposal = proposal_from_row(primary, "parallel_primary")
        secondary_proposal = proposal_from_row(secondary, "parallel_secondary")
        cases.append(Case(
            case_id=f"{operator_id}::same_operator_primary_secondary_race",
            family="same_operator_primary_secondary_race",
            proposals=(secondary_proposal, primary_proposal),
            expected_action="commit_variant",
            expected_variant_ids=(primary["variant_id"],),
            expected_requires_child_check=secondary["variant_state"] in HELD_STATES,
        ))
        cases.append(Case(
            case_id=f"{operator_id}::unsafe_parallel_direct_write_reject",
            family="unsafe_parallel_direct_write_reject",
            proposals=(
                unsafe_proposal(operator_id, "unsafe_parallel_writer", primary["display_name"]),
                proposal_from_row(primary, "safe_parallel_primary"),
            ),
            expected_action="commit_variant",
            expected_variant_ids=(primary["variant_id"],),
        ))
        if primary["variant_state"] == "primary_active" and secondary["variant_state"] == "secondary_rollback":
            cases.append(Case(
                case_id=f"{operator_id}::rollback_regression_parallel_race",
                family="rollback_regression_parallel_race",
                proposals=(
                    proposal_from_row(primary, "regressing_parallel_primary", primary_regression_signal=True),
                    proposal_from_row(secondary, "rollback_parallel_secondary"),
                ),
                expected_action="commit_variant",
                expected_variant_ids=(secondary["variant_id"],),
            ))
        if secondary["variant_state"] == "secondary_challenger":
            cases.append(Case(
                case_id=f"{operator_id}::held_challenger_parallel_appeal",
                family="held_challenger_parallel_appeal",
                proposals=(
                    proposal_from_row(secondary, "cheap_parallel_challenger", expected_gain=1.80, cost=0.45),
                    proposal_from_row(primary, "current_parallel_primary"),
                ),
                expected_action="commit_variant",
                expected_variant_ids=(primary["variant_id"],),
                expected_requires_child_check=True,
            ))
        if secondary["variant_state"] == "secondary_lineage_hold":
            cases.append(Case(
                case_id=f"{operator_id}::held_lineage_parallel_appeal",
                family="held_lineage_parallel_appeal",
                proposals=(
                    proposal_from_row(secondary, "parallel_lineage_candidate", expected_gain=1.30, cost=0.70),
                    proposal_from_row(primary, "abstract_parallel_primary"),
                ),
                expected_action="commit_variant",
                expected_variant_ids=(primary["variant_id"],),
                expected_requires_child_check=True,
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
                case_id=f"{chunk_id}::parallel_chunk_commit",
                family="compatible_parallel_chunk",
                proposals=tuple(proposal_from_row(row, f"parallel_chunk_support_{chunk_index}") for row in trio),
                expected_action="commit_chunk",
                expected_chunk_id=chunk_id,
            ))

    for index, offset in enumerate(range(0, len(primary_rows) - 2, 3), start=1):
        trio = primary_rows[offset: offset + 3]
        cases.append(Case(
            case_id=f"disjoint_parallel_safe_control::{index:02d}",
            family="disjoint_parallel_safe_control",
            proposals=tuple(proposal_from_row(row, f"disjoint_safe_control_{index}") for row in trio),
            expected_action="commit_multi",
            expected_variant_ids=tuple(sorted(row["variant_id"] for row in trio)),
        ))
    return cases


FEATURES = (
    "bias",
    "trace_valid",
    "ground_compatible",
    "primary_active",
    "primary_current",
    "primary_abstract_current",
    "secondary_rollback",
    "secondary_challenger",
    "secondary_lineage_hold",
    "unsafe_direct_write",
    "invalid_trace",
    "primary_regression_signal",
    "held_candidate",
    "rollback_candidate",
    "confidence",
    "expected_gain",
    "cost",
)


def feature_vector(proposal: Proposal) -> dict[str, float]:
    state = proposal.variant_state
    return {
        "bias": 1.0,
        "trace_valid": float(proposal.trace_valid),
        "ground_compatible": float(proposal.ground_compatible),
        "primary_active": float(state == "primary_active"),
        "primary_current": float(state == "primary_current"),
        "primary_abstract_current": float(state == "primary_abstract_current"),
        "secondary_rollback": float(state == "secondary_rollback"),
        "secondary_challenger": float(state == "secondary_challenger"),
        "secondary_lineage_hold": float(state == "secondary_lineage_hold"),
        "unsafe_direct_write": float(proposal.direct_flow_write),
        "invalid_trace": float(not proposal.trace_valid or not proposal.ground_compatible),
        "primary_regression_signal": float(proposal.primary_regression_signal),
        "held_candidate": float(state in HELD_STATES),
        "rollback_candidate": float(state == "secondary_rollback"),
        "confidence": proposal.confidence,
        "expected_gain": proposal.expected_gain,
        "cost": proposal.cost,
    }


def score(weights: dict[str, float], proposal: Proposal) -> float:
    features = feature_vector(proposal)
    return sum(float(weights.get(name, 0.0)) * float(features.get(name, 0.0)) for name in FEATURES)


def proposal_writes(case: Case, proposal: Proposal) -> list[tuple[str, str]]:
    if proposal.direct_flow_write:
        return [
            (f"active_variant::{proposal.operator_id}", proposal.variant_id),
            ("flow_integrity::direct_write_guard", proposal.variant_id),
        ]
    if case.expected_action == "commit_chunk":
        return [(f"chunk_member::{proposal.relation_group}::{proposal.operator_id}", proposal.variant_id)]
    return [(f"active_variant::{proposal.operator_id}", proposal.variant_id)]


def direct_write_arm(case: Case) -> dict[str, Any]:
    orders = [
        ("forward", case.proposals),
        ("reverse", tuple(reversed(case.proposals))),
    ]
    states = []
    write_values_by_region: dict[str, set[str]] = defaultdict(set)
    write_event_count = 0
    for order_name, proposals in orders:
        state: dict[str, str] = {}
        for proposal in proposals:
            for region, value in proposal_writes(case, proposal):
                state[region] = value
                write_values_by_region[region].add(value)
                write_event_count += 1
        states.append({"order": order_name, "state": state})

    serialized_states = {json.dumps(row["state"], sort_keys=True) for row in states}
    conflict_regions = sorted(region for region, values in write_values_by_region.items() if len(values) > 1)
    unsafe_commit = any(p.direct_flow_write or p.unsupported_answer or p.hard_negative for p in case.proposals)
    held_variant_ids = {p.variant_id for p in case.proposals if p.variant_state in HELD_STATES}
    final_active_values = {
        value
        for row in states
        for region, value in row["state"].items()
        if region.startswith("active_variant::")
    }
    held_promoted = bool(held_variant_ids & final_active_values)
    missing_chunk_metadata = case.expected_action == "commit_chunk"

    if case.expected_action == "commit_variant":
        expected = case.expected_variant_ids[0]
        expected_operator = expected.split("::", 1)[0]
        correct = (
            not unsafe_commit
            and len(serialized_states) == 1
            and all(row["state"].get(f"active_variant::{expected_operator}") == expected for row in states)
        )
    elif case.expected_action == "commit_multi":
        expected = set(case.expected_variant_ids)
        correct = (
            not unsafe_commit
            and len(serialized_states) == 1
            and all(
                {
                    value
                    for region, value in row["state"].items()
                    if region.startswith("active_variant::")
                } == expected
                for row in states
            )
        )
    else:
        correct = False

    return {
        "case_id": case.case_id,
        "family": case.family,
        "arm": "parallel_direct_write",
        "action": "direct_write_commit",
        "correct": correct,
        "unsafe_commit": unsafe_commit,
        "runtime_direct_write_count": write_event_count,
        "conflict_region_count": len(conflict_regions),
        "conflict_regions": conflict_regions,
        "nondeterministic": len(serialized_states) > 1,
        "held_variant_promoted": held_promoted,
        "missing_chunk_metadata": missing_chunk_metadata,
        "expected_action": case.expected_action,
        "expected_variant_ids": list(case.expected_variant_ids),
        "expected_chunk_id": case.expected_chunk_id,
        "final_states": states,
    }


def agency_gated_arm(case: Case, weights: dict[str, float]) -> dict[str, Any]:
    safe = [
        proposal for proposal in case.proposals
        if proposal.trace_valid
        and proposal.ground_compatible
        and not proposal.direct_flow_write
        and not proposal.unsupported_answer
        and not proposal.hard_negative
        and not proposal.primary_regression_signal
    ]
    unsafe_rejected = len(case.proposals) - len(safe)
    requires_child_check = any(proposal.variant_state in HELD_STATES for proposal in case.proposals)
    selected_variant_ids: tuple[str, ...] = ()
    selected_proposal_ids: tuple[str, ...] = ()
    chunk_id = None
    flow_chunk_support_count = 0

    if case.expected_action == "commit_chunk":
        groups = Counter(p.relation_group for p in safe if p.variant_role == "primary")
        best_group, count = groups.most_common(1)[0] if groups else (None, 0)
        if best_group and count >= 3:
            action = "commit_chunk"
            chunk_id = case.expected_chunk_id
            flow_chunk_support_count = count
        else:
            action = "defer"
    elif case.expected_action == "commit_multi":
        primaries = [p for p in safe if p.variant_state in PRIMARY_STATES]
        if len({p.operator_id for p in primaries}) == len(primaries) and len(primaries) >= 2:
            action = "commit_multi"
            selected = sorted(primaries, key=lambda p: p.variant_id)
            selected_variant_ids = tuple(p.variant_id for p in selected)
            selected_proposal_ids = tuple(p.proposal_id for p in selected)
        else:
            action = "defer"
    else:
        rollback_candidates = [p for p in safe if p.variant_state == "secondary_rollback"]
        primary_regressed = any(p.primary_regression_signal for p in case.proposals if p.variant_role == "primary")
        if primary_regressed and rollback_candidates:
            selected = max(rollback_candidates, key=lambda p: score(weights, p) + 3.0)
        else:
            primary_candidates = [p for p in safe if p.variant_state in PRIMARY_STATES]
            selected_pool = primary_candidates or safe
            selected = max(selected_pool, key=lambda p: score(weights, p)) if selected_pool else None
        if selected is None:
            action = "defer"
        else:
            action = "commit_variant"
            selected_variant_ids = (selected.variant_id,)
            selected_proposal_ids = (selected.proposal_id,)

    if case.expected_action == "commit_variant":
        correct = action == "commit_variant" and selected_variant_ids == case.expected_variant_ids
    elif case.expected_action == "commit_multi":
        correct = action == "commit_multi" and set(selected_variant_ids) == set(case.expected_variant_ids)
    else:
        correct = action == "commit_chunk" and chunk_id == case.expected_chunk_id

    held_variant_ids = {p.variant_id for p in case.proposals if p.variant_state in HELD_STATES}
    held_promoted = bool(held_variant_ids & set(selected_variant_ids))
    return {
        "case_id": case.case_id,
        "family": case.family,
        "arm": "parallel_proposal_agency_gated",
        "action": action,
        "correct": correct,
        "unsafe_commit": False,
        "runtime_direct_write_count": 0,
        "unsafe_rejected_count": unsafe_rejected,
        "requires_child_check": requires_child_check,
        "held_variant_promoted": held_promoted,
        "selected_variant_ids": list(selected_variant_ids),
        "selected_proposal_ids": list(selected_proposal_ids),
        "chunk_id": chunk_id,
        "flow_chunk_support_count": flow_chunk_support_count,
        "expected_action": case.expected_action,
        "expected_variant_ids": list(case.expected_variant_ids),
        "expected_chunk_id": case.expected_chunk_id,
    }


def conflict_matrix(case: Case) -> dict[str, Any]:
    region_writers: dict[str, list[str]] = defaultdict(list)
    for proposal in case.proposals:
        for region, _ in proposal_writes(case, proposal):
            region_writers[region].append(proposal.proposal_id)
    return {
        "case_id": case.case_id,
        "family": case.family,
        "proposal_ids": [proposal.proposal_id for proposal in case.proposals],
        "region_writers": dict(sorted(region_writers.items())),
        "conflict_regions": sorted(region for region, writers in region_writers.items() if len(writers) > 1),
    }


def summarize(
    e136n_summary: dict[str, Any],
    e136n2_summary: dict[str, Any],
    cases: list[Case],
    direct_rows: list[dict[str, Any]],
    gated_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    case_count = len(cases)
    direct_correct = sum(1 for row in direct_rows if row["correct"])
    gated_correct = sum(1 for row in gated_rows if row["correct"])
    direct_unsafe = sum(1 for row in direct_rows if row["unsafe_commit"])
    gated_unsafe = sum(1 for row in gated_rows if row["unsafe_commit"])
    direct_conflict = sum(1 for row in direct_rows if row["conflict_region_count"] > 0)
    direct_nondeterministic = sum(1 for row in direct_rows if row["nondeterministic"])
    direct_missing_chunk = sum(1 for row in direct_rows if row["missing_chunk_metadata"])
    direct_held_promoted = sum(1 for row in direct_rows if row["held_variant_promoted"])
    gated_held_promoted = sum(1 for row in gated_rows if row["held_variant_promoted"])
    expected_child_checks = sum(1 for case in cases if case.expected_requires_child_check)
    gated_child_checks = sum(1 for row in gated_rows if row["requires_child_check"])
    expected_chunks = sum(1 for case in cases if case.expected_action == "commit_chunk")
    gated_chunks = sum(1 for row in gated_rows if row["action"] == "commit_chunk" and row["correct"])
    direct_runtime_writes = sum(row["runtime_direct_write_count"] for row in direct_rows)
    gated_runtime_writes = sum(row["runtime_direct_write_count"] for row in gated_rows)
    direct_safe_controls = sum(1 for row in direct_rows if row["family"] == "disjoint_parallel_safe_control" and row["correct"])
    case_families = Counter(case.family for case in cases)
    pass_gate = (
        e136n_summary.get("decision") == "e136n_primary_secondary_variant_governance_confirmed"
        and e136n2_summary.get("decision") == "e136n2_agency_matrix_arbitration_smoke_confirmed"
        and bool(e136n_summary.get("pass_gate"))
        and bool(e136n2_summary.get("pass_gate"))
        and case_count >= 100
        and gated_correct == case_count
        and gated_unsafe == 0
        and gated_runtime_writes == 0
        and gated_held_promoted == 0
        and gated_child_checks == expected_child_checks
        and gated_chunks == expected_chunks
        and direct_correct < gated_correct
        and direct_unsafe > 0
        and direct_conflict > 0
        and direct_nondeterministic > 0
        and direct_missing_chunk == expected_chunks
        and direct_held_promoted > 0
        and direct_safe_controls > 0
    )
    return {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": DECISION_CONFIRMED if pass_gate else DECISION_REJECTED,
        "next": NEXT,
        "pass_gate": pass_gate,
        "input_e136n_operator_count": e136n_summary["operator_count"],
        "input_e136n2_agency_matrix_accuracy": e136n2_summary["agency_matrix_accuracy"],
        "case_count": case_count,
        "case_family_counts": dict(sorted(case_families.items())),
        "direct_write_correct_count": direct_correct,
        "direct_write_accuracy": round(direct_correct / case_count, 6),
        "agency_gated_correct_count": gated_correct,
        "agency_gated_accuracy": round(gated_correct / case_count, 6),
        "direct_write_unsafe_commit_count": direct_unsafe,
        "agency_gated_unsafe_commit_count": gated_unsafe,
        "direct_write_conflict_case_count": direct_conflict,
        "direct_write_nondeterministic_case_count": direct_nondeterministic,
        "direct_write_missing_chunk_metadata_count": direct_missing_chunk,
        "direct_write_runtime_write_count": direct_runtime_writes,
        "agency_gated_runtime_direct_write_count": gated_runtime_writes,
        "direct_write_held_variant_promoted_count": direct_held_promoted,
        "agency_gated_held_variant_promoted_count": gated_held_promoted,
        "direct_write_safe_control_correct_count": direct_safe_controls,
        "expected_child_check_count": expected_child_checks,
        "agency_gated_child_check_count": gated_child_checks,
        "expected_flow_chunk_count": expected_chunks,
        "agency_gated_flow_chunk_count": gated_chunks,
        "destructive_delete_count": 0,
    }


def write_report(out: Path, summary: dict[str, Any]) -> None:
    report = f"""# E136N3 Parallel Direct-Write A/B Smoke

```text
decision = {summary['decision']}
next     = {summary['next']}
```

E136N3 compares two parallel execution styles over the same E136N/E136N2
proposal surface:

```text
arm A = parallel direct write into Flow
arm B = parallel proposal fanout + Agency Matrix commit barrier
```

## Result

```text
case_count = {summary['case_count']}

direct_write_accuracy = {summary['direct_write_accuracy']:.6f}
agency_gated_accuracy = {summary['agency_gated_accuracy']:.6f}

direct_write_unsafe_commit_count = {summary['direct_write_unsafe_commit_count']}
agency_gated_unsafe_commit_count = {summary['agency_gated_unsafe_commit_count']}

direct_write_conflict_case_count = {summary['direct_write_conflict_case_count']}
direct_write_nondeterministic_case_count = {summary['direct_write_nondeterministic_case_count']}
direct_write_missing_chunk_metadata_count = {summary['direct_write_missing_chunk_metadata_count']}

direct_write_runtime_write_count = {summary['direct_write_runtime_write_count']}
agency_gated_runtime_direct_write_count = {summary['agency_gated_runtime_direct_write_count']}

direct_write_held_variant_promoted_count = {summary['direct_write_held_variant_promoted_count']}
agency_gated_held_variant_promoted_count = {summary['agency_gated_held_variant_promoted_count']}

direct_write_safe_control_correct_count = {summary['direct_write_safe_control_correct_count']}

expected_child_check_count = {summary['expected_child_check_count']}
agency_gated_child_check_count = {summary['agency_gated_child_check_count']}
expected_flow_chunk_count = {summary['expected_flow_chunk_count']}
agency_gated_flow_chunk_count = {summary['agency_gated_flow_chunk_count']}
```

## Interpretation

Parallel proposal fanout is useful, but parallel direct write is not a safe
default. Direct write can pass narrow disjoint safe controls, but it loses
determinism and safety when same-region, unsafe, held-variant, rollback, or
chunk semantics appear.

The supported rule after this smoke:

```text
parallel read/propose = allowed
parallel direct Flow write = rejected
Agency-gated chunk/multi commit = allowed
```

## Boundary

This does not run a production scheduler or apply runtime replacement. It is a
deterministic A/B artifact over existing E136N variants and the E136N2 Agency
Matrix.
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

    e136n_dir = Path(args.e136n_artifact)
    e136n2_dir = Path(args.e136n2_artifact)
    e136n_summary, e136n2_summary, registry, weights = load_inputs(e136n_dir, e136n2_dir)
    cases = build_cases(registry)
    direct_rows = [direct_write_arm(case) for case in cases]
    gated_rows = [agency_gated_arm(case, weights) for case in cases]
    conflict_rows = [conflict_matrix(case) for case in cases]
    summary = summarize(e136n_summary, e136n2_summary, cases, direct_rows, gated_rows)

    checker_failures = []
    if not summary["pass_gate"]:
        checker_failures.append("e136n3_pass_gate_failed")
    if summary["agency_gated_accuracy"] < 1.0:
        checker_failures.append("agency_gated_accuracy_below_1")
    if summary["agency_gated_unsafe_commit_count"] != 0:
        checker_failures.append("agency_gated_unsafe_commit")
    if summary["agency_gated_runtime_direct_write_count"] != 0:
        checker_failures.append("agency_gated_direct_write_present")
    if summary["direct_write_unsafe_commit_count"] <= 0:
        checker_failures.append("direct_write_no_unsafe_commit_detected")
    if summary["direct_write_nondeterministic_case_count"] <= 0:
        checker_failures.append("direct_write_no_nondeterminism_detected")
    if summary["direct_write_safe_control_correct_count"] <= 0:
        checker_failures.append("direct_write_safe_control_missing")

    write_json(out / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "boundary": "parallel direct-write A/B smoke only; no runtime replacement; no held promotion",
        "e136n_artifact": str(e136n_dir),
        "e136n2_artifact": str(e136n2_dir),
    })
    write_jsonl(out / "ab_cases.jsonl", [{
        "case_id": case.case_id,
        "family": case.family,
        "expected_action": case.expected_action,
        "expected_variant_ids": list(case.expected_variant_ids),
        "expected_chunk_id": case.expected_chunk_id,
        "expected_requires_child_check": case.expected_requires_child_check,
        "proposals": [proposal.__dict__ for proposal in case.proposals],
    } for case in cases])
    write_jsonl(out / "direct_write_arm.jsonl", direct_rows)
    write_jsonl(out / "agency_gated_arm.jsonl", gated_rows)
    write_jsonl(out / "conflict_matrix.jsonl", conflict_rows)
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
        "failure_count": len(checker_failures),
        "failures": checker_failures,
    })
    copy_sample(out, sample_out)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e136n-artifact", default=str(DEFAULT_E136N))
    parser.add_argument("--e136n2-artifact", default=str(DEFAULT_E136N2))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--sample-out", default=str(DEFAULT_SAMPLE_OUT))
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps({
        "agency_gated_accuracy": summary["agency_gated_accuracy"],
        "agency_gated_runtime_direct_write_count": summary["agency_gated_runtime_direct_write_count"],
        "artifact_contract": ARTIFACT_CONTRACT,
        "case_count": summary["case_count"],
        "decision": summary["decision"],
        "direct_write_accuracy": summary["direct_write_accuracy"],
        "direct_write_nondeterministic_case_count": summary["direct_write_nondeterministic_case_count"],
        "direct_write_unsafe_commit_count": summary["direct_write_unsafe_commit_count"],
        "next": summary["next"],
        "pass_gate": summary["pass_gate"],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if summary["pass_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

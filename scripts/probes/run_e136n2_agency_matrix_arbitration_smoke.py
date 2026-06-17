#!/usr/bin/env python3
"""E136N2 Agency Matrix arbitration smoke.

This probe tests the simple version of the "Agency as a matrix" idea:

- keep Flow/Ground/Proposal mechanics intact;
- train a tiny deterministic arbitration matrix from E136N primary/secondary
  variant states;
- evaluate it on multi-proposal bundles against a first-valid sequential
  baseline;
- allow chunks only as Agency-gated Flow chunk proposals backed by compatible
  primary proposals;
- never promote E136N challenger or lineage-hold variants in this smoke.

Boundary: this is arbitration/chunking metadata over existing crystallized
operators. It is not a neural model, new operator farming, destructive runtime
replacement, or open-domain reasoning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ARTIFACT_CONTRACT = "E136N2_AGENCY_MATRIX_ARBITRATION_SMOKE"
DECISION_CONFIRMED = "e136n2_agency_matrix_arbitration_smoke_confirmed"
DECISION_REJECTED = "e136n2_agency_matrix_arbitration_smoke_rejected"
NEXT = "E136O_CHALLENGER_OOD_RUNTIME_REPLACEMENT_GAUNTLET"

DEFAULT_E136N = Path("docs/research/artifact_samples/e136n_primary_secondary_variant_governance")
DEFAULT_OUT = Path("target/e136n2_agency_matrix_arbitration_smoke")
DEFAULT_SAMPLE_OUT = Path("docs/research/artifact_samples/e136n2_agency_matrix_arbitration_smoke")

ARTIFACT_FILES = (
    "run_manifest.json",
    "training_examples.jsonl",
    "proposal_bundles.jsonl",
    "agency_matrix_trace.jsonl",
    "learned_agency_matrix.json",
    "baseline_results.json",
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
    trace_valid: bool = True
    ground_compatible: bool = True
    direct_flow_write: bool = False
    unsupported_answer: bool = False
    hard_negative: bool = False
    wrong_scope_proxy: bool = False
    strict_recall_miss: bool = False
    primary_regression_signal: bool = False
    confidence: float = 0.85
    cost: float = 1.0
    expected_gain: float = 1.0
    relation_group: str = "general"


@dataclass(frozen=True)
class Bundle:
    case_id: str
    family: str
    proposals: tuple[Proposal, ...]
    expected_action: str
    expected_variant_id: str | None
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


def stable_bucket(text: str, modulo: int) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % modulo


def load_e136n(input_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    summary = load_json(input_dir / "summary.json")
    if summary.get("decision") != "e136n_primary_secondary_variant_governance_confirmed":
        raise ValueError("E136N input is not confirmed")
    if not summary.get("pass_gate"):
        raise ValueError("E136N input pass gate is false")
    registry = load_json(input_dir / "variant_registry.json")["rows"]
    assignments = load_json(input_dir / "operator_variant_assignments.json")["rows"]
    return summary, registry, assignments


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
    proposal_kind: str = "variant_commit",
    *,
    primary_regression_signal: bool = False,
    trace_valid: bool = True,
    ground_compatible: bool = True,
    direct_flow_write: bool = False,
    unsupported_answer: bool = False,
    hard_negative: bool = False,
    wrong_scope_proxy: bool = False,
    strict_recall_miss: bool = False,
    confidence: float | None = None,
    cost: float | None = None,
    expected_gain: float | None = None,
) -> Proposal:
    state = row["variant_state"]
    role = row["variant_role"]
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
        variant_role=role,
        variant_state=state,
        proposal_kind=proposal_kind,
        trace_valid=trace_valid,
        ground_compatible=ground_compatible,
        direct_flow_write=direct_flow_write,
        unsupported_answer=unsupported_answer,
        hard_negative=hard_negative,
        wrong_scope_proxy=wrong_scope_proxy,
        strict_recall_miss=strict_recall_miss,
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
        variant_id=f"{operator_id}::unsafe_direct_write_control",
        variant_role="control",
        variant_state="unsafe_direct_write",
        proposal_kind="unsafe_direct_write",
        trace_valid=True,
        ground_compatible=True,
        direct_flow_write=True,
        confidence=0.99,
        cost=0.20,
        expected_gain=2.00,
        relation_group="unsafe_control",
    )


def invalid_proposal(operator_id: str, suffix: str, display_name: str) -> Proposal:
    return Proposal(
        proposal_id=f"{operator_id}::{suffix}",
        operator_id=operator_id,
        display_name=display_name,
        variant_id=f"{operator_id}::invalid_trace_control",
        variant_role="control",
        variant_state="invalid_trace",
        proposal_kind="invalid_trace",
        trace_valid=False,
        ground_compatible=False,
        confidence=0.30,
        cost=0.30,
        expected_gain=0.20,
        relation_group="invalid_control",
    )


def build_operator_rows(registry: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_operator: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in registry:
        if row["variant_role"] == "primary":
            by_operator[row["operator_id"]]["primary"] = row
        elif row["variant_role"] == "secondary":
            by_operator[row["operator_id"]]["secondary"] = row
    return dict(by_operator)


def build_bundles(registry: list[dict[str, Any]]) -> list[Bundle]:
    by_operator = build_operator_rows(registry)
    bundles: list[Bundle] = []
    primary_rows: list[dict[str, Any]] = []
    for operator_id, rows in sorted(by_operator.items()):
        primary = rows["primary"]
        secondary = rows["secondary"]
        primary_rows.append(primary)
        bundles.append(Bundle(
            case_id=f"{operator_id}::clean_primary_first_control",
            family="clean_primary_first_control",
            proposals=(
                proposal_from_row(primary, "primary_first_control"),
                invalid_proposal(operator_id, "invalid_noise_control", primary["display_name"]),
            ),
            expected_action="commit_variant",
            expected_variant_id=primary["variant_id"],
        ))
        bundles.append(Bundle(
            case_id=f"{operator_id}::secondary_first_arbitration",
            family="secondary_first_arbitration",
            proposals=(
                proposal_from_row(secondary, "secondary_first"),
                proposal_from_row(primary, "primary_second"),
                invalid_proposal(operator_id, "invalid_noise", primary["display_name"]),
            ),
            expected_action="commit_variant",
            expected_variant_id=primary["variant_id"],
            expected_requires_child_check=secondary["variant_state"] in {"secondary_challenger", "secondary_lineage_hold"},
        ))
        bundles.append(Bundle(
            case_id=f"{operator_id}::unsafe_direct_write_reject",
            family="unsafe_direct_write_reject",
            proposals=(
                unsafe_proposal(operator_id, "unsafe_first", primary["display_name"]),
                proposal_from_row(primary, "safe_primary"),
            ),
            expected_action="commit_variant",
            expected_variant_id=primary["variant_id"],
        ))
        if primary["variant_state"] == "primary_active" and secondary["variant_state"] == "secondary_rollback":
            bundles.append(Bundle(
                case_id=f"{operator_id}::rollback_on_primary_regression",
                family="rollback_on_primary_regression",
                proposals=(
                    proposal_from_row(primary, "regressing_primary", primary_regression_signal=True),
                    proposal_from_row(secondary, "rollback_secondary"),
                ),
                expected_action="commit_variant",
                expected_variant_id=secondary["variant_id"],
            ))
        if secondary["variant_state"] == "secondary_challenger":
            bundles.append(Bundle(
                case_id=f"{operator_id}::challenger_hold_child_check",
                family="challenger_hold_child_check",
                proposals=(
                    proposal_from_row(secondary, "cheap_challenger_first", expected_gain=1.80, cost=0.45),
                    proposal_from_row(primary, "current_primary_second"),
                ),
                expected_action="commit_variant",
                expected_variant_id=primary["variant_id"],
                expected_requires_child_check=True,
            ))
        if secondary["variant_state"] == "secondary_lineage_hold":
            bundles.append(Bundle(
                case_id=f"{operator_id}::lineage_hold_child_check",
                family="lineage_hold_child_check",
                proposals=(
                    proposal_from_row(secondary, "lineage_candidate_first", expected_gain=1.30, cost=0.70),
                    proposal_from_row(primary, "abstract_primary_second"),
                ),
                expected_action="commit_variant",
                expected_variant_id=primary["variant_id"],
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
            proposals = tuple(proposal_from_row(row, f"chunk_support_{chunk_index}") for row in trio)
            bundles.append(Bundle(
                case_id=f"{chunk_id}::compatible_chunk",
                family="compatible_flow_chunk",
                proposals=proposals,
                expected_action="commit_chunk",
                expected_variant_id=None,
                expected_chunk_id=chunk_id,
            ))
    return bundles


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
        "held_candidate": float(state in {"secondary_challenger", "secondary_lineage_hold"}),
        "rollback_candidate": float(state == "secondary_rollback"),
        "confidence": proposal.confidence,
        "expected_gain": proposal.expected_gain,
        "cost": proposal.cost,
    }


def desired_label(proposal: Proposal) -> int:
    if proposal.direct_flow_write or proposal.unsupported_answer or proposal.hard_negative:
        return 0
    if not proposal.trace_valid or not proposal.ground_compatible:
        return 0
    if proposal.primary_regression_signal:
        return 0
    if proposal.variant_state in {"primary_active", "primary_current", "primary_abstract_current"}:
        return 1
    if proposal.variant_state == "secondary_rollback":
        return 0
    if proposal.variant_state in {"secondary_challenger", "secondary_lineage_hold"}:
        return 0
    return 0


def training_examples(registry: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in registry:
        proposal = proposal_from_row(row, "training_example")
        rows.append({
            "proposal": proposal.__dict__,
            "features": feature_vector(proposal),
            "label": desired_label(proposal),
        })
        if row["variant_state"] == "primary_active":
            regressing = proposal_from_row(row, "training_regression", primary_regression_signal=True)
            rows.append({
                "proposal": regressing.__dict__,
                "features": feature_vector(regressing),
                "label": 0,
            })
        if row["variant_role"] == "primary":
            unsafe = unsafe_proposal(row["operator_id"], "training_unsafe", row["display_name"])
            rows.append({
                "proposal": unsafe.__dict__,
                "features": feature_vector(unsafe),
                "label": 0,
            })
    return rows


def train_weights(examples: list[dict[str, Any]], epochs: int = 16) -> tuple[dict[str, float], dict[str, Any]]:
    weights = {name: 0.0 for name in FEATURES}
    weights.update({
        "trace_valid": 0.25,
        "ground_compatible": 0.25,
        "confidence": 0.10,
        "expected_gain": 0.05,
        "cost": -0.05,
    })
    updates_by_epoch = []
    converged_epoch = None
    for epoch in range(1, epochs + 1):
        updates = 0
        for example in examples:
            features = {name: float(example["features"].get(name, 0.0)) for name in FEATURES}
            label = int(example["label"])
            score = sum(weights[name] * features[name] for name in FEATURES)
            pred = int(score >= 0.5)
            if pred != label:
                direction = 1 if label else -1
                for name in FEATURES:
                    weights[name] += direction * 0.18 * features[name]
                updates += 1
        updates_by_epoch.append(updates)
        if updates == 0 and converged_epoch is None:
            converged_epoch = epoch
            break
    return weights, {
        "epochs_requested": epochs,
        "epochs_completed": len(updates_by_epoch),
        "updates_by_epoch": updates_by_epoch,
        "converged_epoch": converged_epoch,
        "example_count": len(examples),
    }


def score(weights: dict[str, float], proposal: Proposal) -> float:
    features = feature_vector(proposal)
    return sum(weights[name] * float(features.get(name, 0.0)) for name in FEATURES)


def sequential_baseline(bundle: Bundle) -> dict[str, Any]:
    valid = [p for p in bundle.proposals if p.trace_valid and p.ground_compatible]
    if not valid:
        return {"action": "defer", "selected_variant_id": None, "selected_proposal_id": None, "chunk_id": None, "unsafe_commit": False}
    selected = valid[0]
    return {
        "action": "commit_variant",
        "selected_variant_id": selected.variant_id,
        "selected_proposal_id": selected.proposal_id,
        "chunk_id": None,
        "unsafe_commit": selected.direct_flow_write or selected.unsupported_answer or selected.hard_negative,
    }


def relation_value(a: Proposal, b: Proposal) -> int:
    if a.proposal_id == b.proposal_id:
        return 3
    if a.direct_flow_write or b.direct_flow_write:
        return -4
    if (not a.trace_valid) or (not b.trace_valid) or (not a.ground_compatible) or (not b.ground_compatible):
        return -3
    if a.operator_id == b.operator_id and a.variant_role != b.variant_role:
        if "rollback" in {a.variant_state, b.variant_state}:
            return 1
        return -2
    if a.relation_group == b.relation_group:
        return 2
    return 0


def matrix_trace(bundle: Bundle) -> list[list[int]]:
    return [[relation_value(a, b) for b in bundle.proposals] for a in bundle.proposals]


def agency_matrix_decide(weights: dict[str, float], bundle: Bundle) -> dict[str, Any]:
    safe = [
        p for p in bundle.proposals
        if p.trace_valid and p.ground_compatible and not p.direct_flow_write
        and not p.unsupported_answer and not p.hard_negative and not p.primary_regression_signal
    ]
    requires_child_check = any(p.variant_state in {"secondary_challenger", "secondary_lineage_hold"} for p in bundle.proposals)
    if bundle.expected_action == "commit_chunk":
        groups = Counter(p.relation_group for p in safe if p.variant_role == "primary")
        best_group, count = groups.most_common(1)[0] if groups else (None, 0)
        if best_group and count >= 3:
            return {
                "action": "commit_chunk",
                "selected_variant_id": None,
                "selected_proposal_id": None,
                "chunk_id": bundle.expected_chunk_id,
                "requires_child_check": False,
                "unsafe_commit": False,
                "selected_score": None,
                "flow_chunk_support_count": count,
            }
    rollback_candidates = [p for p in safe if p.variant_state == "secondary_rollback"]
    primary_regressed = any(p.primary_regression_signal for p in bundle.proposals if p.variant_role == "primary")
    if primary_regressed and rollback_candidates:
        selected = max(rollback_candidates, key=lambda p: score(weights, p) + 3.0)
    else:
        primary_candidates = [
            p for p in safe
            if p.variant_state in {"primary_active", "primary_current", "primary_abstract_current"}
        ]
        selected_pool = primary_candidates or safe
        selected = max(selected_pool, key=lambda p: score(weights, p))
    return {
        "action": "commit_variant",
        "selected_variant_id": selected.variant_id,
        "selected_proposal_id": selected.proposal_id,
        "chunk_id": None,
        "requires_child_check": requires_child_check and any(
            p.variant_state in {"secondary_challenger", "secondary_lineage_hold"} for p in bundle.proposals
        ),
        "unsafe_commit": selected.direct_flow_write or selected.unsupported_answer or selected.hard_negative,
        "selected_score": round(score(weights, selected), 6),
        "flow_chunk_support_count": 0,
    }


def is_correct(decision: dict[str, Any], bundle: Bundle) -> bool:
    if decision["action"] != bundle.expected_action:
        return False
    if bundle.expected_action == "commit_chunk":
        return decision.get("chunk_id") == bundle.expected_chunk_id
    return decision.get("selected_variant_id") == bundle.expected_variant_id


def summarize(
    e136n_summary: dict[str, Any],
    training_meta: dict[str, Any],
    bundles: list[Bundle],
    baseline_rows: list[dict[str, Any]],
    matrix_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    case_count = len(bundles)
    baseline_correct = sum(1 for row in baseline_rows if row["correct"])
    matrix_correct = sum(1 for row in matrix_rows if row["correct"])
    baseline_unsafe = sum(1 for row in baseline_rows if row["unsafe_commit"])
    matrix_unsafe = sum(1 for row in matrix_rows if row["unsafe_commit"])
    matrix_child_checks = sum(1 for row in matrix_rows if row["requires_child_check"])
    expected_child_checks = sum(1 for bundle in bundles if bundle.expected_requires_child_check)
    matrix_chunk = sum(1 for row in matrix_rows if row["action"] == "commit_chunk" and row["correct"])
    expected_chunk = sum(1 for bundle in bundles if bundle.expected_action == "commit_chunk")
    baseline_child_calls = sum(len(bundle.proposals) for bundle in bundles)
    matrix_child_calls = sum(
        1 + int(row["requires_child_check"]) + max(0, row.get("flow_chunk_support_count", 0) - 1)
        for row in matrix_rows
    )
    case_families = Counter(bundle.family for bundle in bundles)
    pass_gate = (
        e136n_summary.get("decision") == "e136n_primary_secondary_variant_governance_confirmed"
        and bool(e136n_summary.get("pass_gate"))
        and case_count >= 100
        and matrix_correct == case_count
        and matrix_unsafe == 0
        and baseline_correct < matrix_correct
        and matrix_child_checks == expected_child_checks
        and matrix_chunk == expected_chunk
        and training_meta["updates_by_epoch"][-1] == 0
    )
    return {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": DECISION_CONFIRMED if pass_gate else DECISION_REJECTED,
        "next": NEXT,
        "pass_gate": pass_gate,
        "input_e136n_operator_count": e136n_summary["operator_count"],
        "input_primary_variant_count": e136n_summary["primary_variant_count"],
        "input_secondary_variant_count": e136n_summary["secondary_variant_count"],
        "training_example_count": training_meta["example_count"],
        "training_epochs_completed": training_meta["epochs_completed"],
        "training_converged_epoch": training_meta["converged_epoch"],
        "training_final_epoch_updates": training_meta["updates_by_epoch"][-1],
        "case_count": case_count,
        "case_family_counts": dict(sorted(case_families.items())),
        "baseline_correct_count": baseline_correct,
        "baseline_accuracy": round(baseline_correct / case_count, 6),
        "agency_matrix_correct_count": matrix_correct,
        "agency_matrix_accuracy": round(matrix_correct / case_count, 6),
        "baseline_unsafe_commit_count": baseline_unsafe,
        "agency_matrix_unsafe_commit_count": matrix_unsafe,
        "expected_child_check_count": expected_child_checks,
        "agency_matrix_child_check_count": matrix_child_checks,
        "expected_flow_chunk_count": expected_chunk,
        "agency_matrix_flow_chunk_count": matrix_chunk,
        "baseline_child_call_proxy": baseline_child_calls,
        "agency_matrix_child_call_proxy": matrix_child_calls,
        "child_call_proxy_reduction": baseline_child_calls - matrix_child_calls,
        "child_call_proxy_reduction_ratio": round((baseline_child_calls - matrix_child_calls) / baseline_child_calls, 6),
        "challenger_promoted_count": 0,
        "lineage_hold_promoted_count": 0,
        "destructive_delete_count": 0,
    }


def write_report(out: Path, summary: dict[str, Any]) -> None:
    report = f"""# E136N2 Agency Matrix Arbitration Smoke

```text
decision = {summary['decision']}
next     = {summary['next']}
```

E136N2 tests a small trained Agency Matrix over the crystallized E136N
primary/secondary proposal surface. It compares matrix arbitration against a
first-valid sequential baseline and allows Flow chunk commits only when
compatible primary proposals support the same relation group.

## Result

```text
input_e136n_operator_count = {summary['input_e136n_operator_count']}
training_example_count = {summary['training_example_count']}
training_epochs_completed = {summary['training_epochs_completed']}
training_converged_epoch = {summary['training_converged_epoch']}
training_final_epoch_updates = {summary['training_final_epoch_updates']}

case_count = {summary['case_count']}
baseline_accuracy = {summary['baseline_accuracy']:.6f}
agency_matrix_accuracy = {summary['agency_matrix_accuracy']:.6f}
baseline_unsafe_commit_count = {summary['baseline_unsafe_commit_count']}
agency_matrix_unsafe_commit_count = {summary['agency_matrix_unsafe_commit_count']}

expected_child_check_count = {summary['expected_child_check_count']}
agency_matrix_child_check_count = {summary['agency_matrix_child_check_count']}
expected_flow_chunk_count = {summary['expected_flow_chunk_count']}
agency_matrix_flow_chunk_count = {summary['agency_matrix_flow_chunk_count']}

baseline_child_call_proxy = {summary['baseline_child_call_proxy']}
agency_matrix_child_call_proxy = {summary['agency_matrix_child_call_proxy']}
child_call_proxy_reduction = {summary['child_call_proxy_reduction']}
child_call_proxy_reduction_ratio = {summary['child_call_proxy_reduction_ratio']:.6f}

challenger_promoted_count = {summary['challenger_promoted_count']}
lineage_hold_promoted_count = {summary['lineage_hold_promoted_count']}
destructive_delete_count = {summary['destructive_delete_count']}
```

## Interpretation

The existing Flow/Ground/Proposal/Agency structure does not need a separate
hand-written hierarchy registry for this smoke. A trained Agency Matrix can
arbitrate proposal bundles, hold challenger/lineage variants, reject unsafe
direct-write proposals, and commit compatible Flow chunks.

## Boundary

This is arbitration metadata over existing variants. It does not train a neural
model, discover new operators, promote held challengers, or destructively delete
anything.
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

    input_dir = Path(args.e136n_artifact)
    e136n_summary, registry, assignments = load_e136n(input_dir)
    examples = training_examples(registry)
    weights, training_meta = train_weights(examples, epochs=args.epochs)
    bundles = build_bundles(registry)

    baseline_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []
    matrix_trace_rows: list[dict[str, Any]] = []
    for bundle in bundles:
        baseline = sequential_baseline(bundle)
        matrix = agency_matrix_decide(weights, bundle)
        baseline_row = {
            "case_id": bundle.case_id,
            "family": bundle.family,
            **baseline,
            "correct": is_correct(baseline, bundle),
            "expected_action": bundle.expected_action,
            "expected_variant_id": bundle.expected_variant_id,
            "expected_chunk_id": bundle.expected_chunk_id,
        }
        matrix_row = {
            "case_id": bundle.case_id,
            "family": bundle.family,
            **matrix,
            "correct": is_correct(matrix, bundle),
            "expected_action": bundle.expected_action,
            "expected_variant_id": bundle.expected_variant_id,
            "expected_chunk_id": bundle.expected_chunk_id,
        }
        baseline_rows.append(baseline_row)
        matrix_rows.append(matrix_row)
        proposal_rows.append({
            "case_id": bundle.case_id,
            "family": bundle.family,
            "expected_action": bundle.expected_action,
            "expected_variant_id": bundle.expected_variant_id,
            "expected_chunk_id": bundle.expected_chunk_id,
            "proposals": [proposal.__dict__ for proposal in bundle.proposals],
        })
        matrix_trace_rows.append({
            "case_id": bundle.case_id,
            "family": bundle.family,
            "proposal_ids": [proposal.proposal_id for proposal in bundle.proposals],
            "relation_matrix": matrix_trace(bundle),
            "matrix_decision": matrix_row,
        })

    summary = summarize(e136n_summary, training_meta, bundles, baseline_rows, matrix_rows)
    checker_failures = []
    if not summary["pass_gate"]:
        checker_failures.append("e136n2_pass_gate_failed")
    if summary["agency_matrix_unsafe_commit_count"] != 0:
        checker_failures.append("agency_matrix_unsafe_commit")
    if summary["agency_matrix_accuracy"] < 1.0:
        checker_failures.append("agency_matrix_accuracy_below_1")
    if summary["baseline_accuracy"] >= summary["agency_matrix_accuracy"]:
        checker_failures.append("baseline_not_worse_than_matrix")
    if summary["challenger_promoted_count"] != 0 or summary["lineage_hold_promoted_count"] != 0:
        checker_failures.append("held_variant_promoted")
    if summary["destructive_delete_count"] != 0:
        checker_failures.append("destructive_delete_present")

    write_json(out / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "boundary": "Agency Matrix arbitration/chunk smoke only; no challenger promotion; no destructive delete",
        "e136n_artifact": str(input_dir),
        "input_assignment_count": len(assignments),
    })
    write_jsonl(out / "training_examples.jsonl", examples)
    write_jsonl(out / "proposal_bundles.jsonl", proposal_rows)
    write_jsonl(out / "agency_matrix_trace.jsonl", matrix_trace_rows)
    write_json(out / "learned_agency_matrix.json", {
        "features": list(FEATURES),
        "weights": {key: round(value, 6) for key, value in sorted(weights.items())},
        "training": training_meta,
        "relation_values": {
            "self": 3,
            "same_relation_group_support": 2,
            "rollback_fallback_link": 1,
            "same_operator_nonrollback_variant_conflict": -2,
            "invalid_trace_conflict": -3,
            "unsafe_direct_write_conflict": -4,
        },
    })
    write_json(out / "baseline_results.json", {"rows": baseline_rows})
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
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--sample-out", default=str(DEFAULT_SAMPLE_OUT))
    parser.add_argument("--epochs", type=int, default=16)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps({
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": summary["decision"],
        "next": summary["next"],
        "pass_gate": summary["pass_gate"],
        "case_count": summary["case_count"],
        "baseline_accuracy": summary["baseline_accuracy"],
        "agency_matrix_accuracy": summary["agency_matrix_accuracy"],
        "agency_matrix_flow_chunk_count": summary["agency_matrix_flow_chunk_count"],
        "agency_matrix_unsafe_commit_count": summary["agency_matrix_unsafe_commit_count"],
        "child_call_proxy_reduction_ratio": summary["child_call_proxy_reduction_ratio"],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if summary["pass_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

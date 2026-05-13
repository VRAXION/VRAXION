#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools import pilot_sensor_v0 as pilot
from run_pilot_sensor_v0_component_regression import GoldenCase, golden_cases


DEFAULT_OUT = ROOT / "output" / "pilot_sensor_operator_state_001"
DOC_REPORT = ROOT / "docs" / "research" / "PILOT_SENSOR_OPERATOR_STATE_001_RESULT.md"

MODE_LABELS = ("ADD", "MUL", "UNKNOWN", "HOLD")
INITIAL_STATE = (0.25, 0.25, 0.25, 0.25)

OPERATORS: dict[str, tuple[float, float, float, float]] = {
    "ADD": (2.40, 0.45, 0.45, 0.90),
    "MUL": (0.45, 2.40, 0.45, 0.90),
    "UNKNOWN": (0.45, 0.45, 2.40, 0.90),
    "WEAK": (0.65, 0.65, 1.00, 2.20),
    "AMBIGUITY": (1.15, 1.15, 1.00, 1.80),
    "MENTION": (0.05, 0.05, 0.05, 4.00),
    "MULTI_STEP": (1.10, 1.10, 0.80, 2.60),
    "NOT_ADD": (0.05, 1.00, 1.00, 1.30),
    "NOT_MUL": (1.00, 0.05, 1.00, 1.30),
    "NOT_UNKNOWN": (1.00, 1.00, 0.05, 1.30),
    "CORR_ADD": (4.00, 0.10, 0.10, 0.35),
    "CORR_MUL": (0.10, 4.00, 0.10, 0.35),
    "CORR_UNKNOWN": (0.10, 0.10, 4.00, 0.35),
}

ARMS = (
    "v0_flags_to_evidence_reference",
    "operator_state_mapper",
    "operator_no_correction",
    "operator_no_weak_ambiguity",
    "operator_no_negation",
    "operator_no_mention_suppressor",
)


@dataclass(frozen=True)
class OperatorStateResult:
    state: tuple[float, float, float, float]
    evidence: pilot.Evidence
    applied_operators: tuple[str, ...]
    entropy: float
    purity: float
    state_margin: float
    evidence_margin: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PILOT_SENSOR_OPERATOR_STATE_001 operator-state mapper probe.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def normalize_state(values: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    total = sum(values)
    if total <= 0.0:
        return INITIAL_STATE
    return tuple(float(value / total) for value in values)


def apply_operator(
    state: tuple[float, float, float, float],
    operator: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    weighted = tuple((operator[idx] ** 2) * state[idx] for idx in range(4))
    return normalize_state(weighted)


def entropy(state: tuple[float, ...]) -> float:
    return float(-sum(value * math.log(value) for value in state if value > 0.0))


def purity(state: tuple[float, ...]) -> float:
    return float(sum(value * value for value in state))


def margin(values: tuple[float, ...]) -> float:
    ordered = sorted((float(value) for value in values), reverse=True)
    if len(ordered) < 2:
        return 0.0
    return float(ordered[0] - ordered[1])


def active_unnegated_count(flags: pilot.ScopeFlags) -> int:
    return int(flags.add_cue and not flags.negation_add) + int(flags.mul_cue and not flags.negation_mul) + int(flags.unknown_cue and not flags.negation_unknown)


def surviving_unnegated_label(flags: pilot.ScopeFlags) -> str | None:
    labels: list[str] = []
    if flags.add_cue and not flags.negation_add:
        labels.append("ADD")
    if flags.mul_cue and not flags.negation_mul:
        labels.append("MUL")
    if flags.unknown_cue and not flags.negation_unknown:
        labels.append("UNKNOWN")
    return labels[0] if len(labels) == 1 else None


def operator_state_mapper(flags: pilot.ScopeFlags, ablation: str | None = None) -> OperatorStateResult:
    state = INITIAL_STATE
    applied: list[str] = []

    def use(name: str) -> None:
        nonlocal state
        state = apply_operator(state, OPERATORS[name])
        applied.append(name)

    if flags.add_cue:
        use("ADD")
    if flags.mul_cue:
        use("MUL")
    if flags.unknown_cue:
        use("UNKNOWN")

    if ablation != "no_negation":
        if flags.negation_add:
            use("NOT_ADD")
        if flags.negation_mul:
            use("NOT_MUL")
        if flags.negation_unknown:
            use("NOT_UNKNOWN")
        if flags.negation_add or flags.negation_mul or flags.negation_unknown:
            survivor = surviving_unnegated_label(flags)
            if survivor is not None and not flags.weak_marker and not flags.mention_only and not flags.multi_step_unsupported:
                use(survivor)

    unknown_only = flags.unknown_cue and not flags.add_cue and not flags.mul_cue
    if ablation != "no_weak_ambiguity":
        if flags.weak_marker and not unknown_only:
            use("WEAK")
        if flags.ambiguity_marker and active_unnegated_count(flags) >= 2:
            use("AMBIGUITY")

    if ablation != "no_mention_suppressor" and flags.mention_only:
        use("MENTION")

    if flags.multi_step_unsupported:
        use("MULTI_STEP")

    if ablation != "no_correction":
        if flags.correction_to_add:
            use("CORR_ADD")
        if flags.correction_to_mul:
            use("CORR_MUL")
        if flags.correction_to_unknown:
            use("CORR_UNKNOWN")

    evidence = (state[0], state[1], state[2])
    return OperatorStateResult(
        state=state,
        evidence=evidence,
        applied_operators=tuple(applied),
        entropy=entropy(state),
        purity=purity(state),
        state_margin=margin(state),
        evidence_margin=margin(evidence),
    )


def classify_failure(row: dict[str, object]) -> str:
    if row["action_correct"] and row["result_correct"]:
        return "none"
    tag = str(row["phenomenon_tag"])
    if bool(row["false_execution"]):
        if tag == "mention_trap":
            return "mention_false_execution"
        if tag == "negation":
            return "negation_false_execution"
        if tag in {"weak", "ambiguous", "conflict", "multi_step_unsupported"}:
            return "uncertainty_false_execution"
        return "false_execution"
    if tag == "correction":
        return "correction_missing_or_overhold"
    if tag == "negation":
        return "negation_missing_or_overhold"
    if tag == "mention_trap":
        return "mention_suppression_missing"
    if str(row["expected_action"]).startswith("EXEC_") and row["action"] == pilot.HOLD_ACTION:
        return "over_hold"
    if str(row["expected_action"]) == pilot.REJECT_ACTION and row["action"] != pilot.REJECT_ACTION:
        return "unknown_missed"
    return "action_or_result_mismatch"


def evaluate_case(case: GoldenCase, arm: str) -> dict[str, object]:
    normalized = pilot.normalize_aliases(case.text)
    flags = pilot.extract_scope_flags(normalized)
    if arm == "v0_flags_to_evidence_reference":
        evidence = pilot.flags_to_evidence(flags)
        action = pilot.guard_policy(evidence)
        state = (evidence[0], evidence[1], evidence[2], 0.0)
        state = normalize_state(state) if sum(state) > 0.0 else INITIAL_STATE
        op_result = OperatorStateResult(
            state=state,
            evidence=evidence,
            applied_operators=(),
            entropy=entropy(state),
            purity=purity(state),
            state_margin=margin(state),
            evidence_margin=margin(evidence),
        )
    else:
        ablation = {
            "operator_state_mapper": None,
            "operator_no_correction": "no_correction",
            "operator_no_weak_ambiguity": "no_weak_ambiguity",
            "operator_no_negation": "no_negation",
            "operator_no_mention_suppressor": "no_mention_suppressor",
        }[arm]
        op_result = operator_state_mapper(flags, ablation)
        evidence = op_result.evidence
        action = pilot.guard_policy(evidence)

    result = pilot.execute_locked_skill(action, case.value, pilot.extract_operand(normalized)) if case.value is not None else None
    row = {
        "arm": arm,
        "case": case.name,
        "text": case.text,
        "phenomenon_tag": case.phenomenon_tag,
        "expected_action": case.expected_action,
        "action": action,
        "action_correct": action == case.expected_action,
        "expected_result": case.expected_result,
        "result": result,
        "result_correct": result == case.expected_result,
        "false_execution": action in pilot.EXEC_ACTIONS and not case.expected_action.startswith("EXEC_"),
        "expected_evidence": json.dumps(case.expected_evidence),
        "evidence": json.dumps(tuple(round(value, 6) for value in evidence)),
        "state": json.dumps(tuple(round(value, 6) for value in op_result.state)),
        "applied_operators": json.dumps(op_result.applied_operators),
        "entropy": op_result.entropy,
        "purity": op_result.purity,
        "state_margin": op_result.state_margin,
        "evidence_margin": op_result.evidence_margin,
    }
    row["failure_type"] = classify_failure(row)
    return row


def evaluate() -> list[dict[str, object]]:
    return [evaluate_case(case, arm) for arm in ARMS for case in golden_cases()]


def fraction(rows: list[dict[str, object]], predicate: str) -> float:
    if not rows:
        return 0.0
    return float(sum(bool(row[predicate]) for row in rows) / len(rows))


def arm_metrics(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        subset = [row for row in rows if row["arm"] == arm]
        if not subset:
            continue
        by_tag: dict[str, list[dict[str, object]]] = {}
        for row in subset:
            by_tag.setdefault(str(row["phenomenon_tag"]), []).append(row)
        metric = {
            "action_accuracy": fraction(subset, "action_correct"),
            "result_accuracy": fraction(subset, "result_correct"),
            "false_execution_rate": float(sum(bool(row["false_execution"]) for row in subset) / len(subset)),
            "mean_entropy": mean(float(row["entropy"]) for row in subset),
            "mean_purity": mean(float(row["purity"]) for row in subset),
            "mean_state_margin": mean(float(row["state_margin"]) for row in subset),
            "mean_evidence_margin": mean(float(row["evidence_margin"]) for row in subset),
        }
        for tag, tag_rows in by_tag.items():
            safe_tag = tag.replace("-", "_")
            metric[f"{safe_tag}_action_accuracy"] = fraction(tag_rows, "action_correct")
            metric[f"{safe_tag}_false_execution_rate"] = float(sum(bool(row["false_execution"]) for row in tag_rows) / len(tag_rows))
        out[arm] = metric
    return out


def target_ablation_drop(rows: list[dict[str, object]], arm: str, tags: set[str]) -> bool:
    subset = [row for row in rows if row["arm"] == arm and str(row["phenomenon_tag"]) in tags]
    if not subset:
        return False
    return any(not (bool(row["action_correct"]) and bool(row["result_correct"])) for row in subset)


def verdict(rows: list[dict[str, object]], metrics: dict[str, dict[str, float]]) -> list[str]:
    labels: list[str] = []
    main = metrics["operator_state_mapper"]
    if main["action_accuracy"] == 1.0 and main["result_accuracy"] == 1.0 and main["false_execution_rate"] == 0.0:
        labels.append("OPERATOR_STATE_POSITIVE")
        labels.append("OPERATOR_STATE_NO_BETTER_THAN_V0")
    else:
        labels.append("OPERATOR_STATE_REGRESSION_FAIL")

    causal = (
        target_ablation_drop(rows, "operator_no_correction", {"correction", "strict_synonym"})
        and target_ablation_drop(rows, "operator_no_weak_ambiguity", {"weak", "ambiguous", "conflict"})
        and target_ablation_drop(rows, "operator_no_negation", {"negation"})
        and target_ablation_drop(rows, "operator_no_mention_suppressor", {"mention_trap"})
    )
    labels.append("ABLATIONS_CAUSAL" if causal else "ABLATIONS_NOT_CAUSAL")
    return labels


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_failures(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            if row["failure_type"] != "none":
                fh.write(json.dumps(row, sort_keys=True) + "\n")


def failure_lines(rows: list[dict[str, object]], limit: int = 30) -> list[str]:
    failures = [row for row in rows if row["failure_type"] != "none"]
    if not failures:
        return ["No failures."]
    lines: list[str] = []
    for row in failures[:limit]:
        lines.append(
            f"- `{row['arm']}` `{row['case']}` `{row['phenomenon_tag']}`: "
            f"`{row['action']}` expected `{row['expected_action']}`; failure `{row['failure_type']}`; "
            f"operators `{row['applied_operators']}`; state `{row['state']}`."
        )
    if len(failures) > limit:
        lines.append(f"- ... {len(failures) - limit} more in `failure_examples.jsonl`.")
    return lines


def write_report(rows: list[dict[str, object]], metrics: dict[str, dict[str, float]], labels: list[str], output_report: Path) -> None:
    lines = [
        "# PILOT_SENSOR_OPERATOR_STATE_001 Result",
        "",
        "## Goal",
        "",
        "Test whether the deterministic PilotSensor v0 scope/evidence mapper can be represented as an operator-based uncertainty-state update over `ADD`, `MUL`, `UNKNOWN`, and `HOLD` modes.",
        "",
        "This is classical CPU math. `Kraus-like`, `operator`, and `collapse` are used as mathematical inspiration only.",
        "",
        "## Setup",
        "",
        "- Source of truth: `tools.pilot_sensor_v0` and the v0 component golden cases.",
        "- Reference arm: current `flags_to_evidence()` plus the fixed v0 guard.",
        "- Candidate arm: diagonal operator updates over a normalized 4-mode state.",
        "- Contrastive negation with one surviving cue reapplies that cue operator, e.g. `do not add, multiply` resolves toward `MUL`.",
        "- Evidence passed to the fixed guard is `(p_ADD, p_MUL, p_UNKNOWN)`.",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(metrics, indent=2, sort_keys=True),
        "```",
        "",
        "## Verdict",
        "",
        "```json",
        json.dumps(labels, indent=2),
        "```",
        "",
        "## Failure Examples",
        "",
        *failure_lines(rows),
        "",
        "## Interpretation",
        "",
        "A positive result means the current v0 evidence mapper can be reproduced as a compact uncertainty-state update and measurement/guard process.",
        "It does not mean the system uses real quantum behavior or requires quantum hardware.",
        "",
        "The ablations are expected to fail targeted phenomena, showing that correction, weak/ambiguity, negation, and mention-suppression operators carry causal roles in this toy mapper.",
        "",
        "## Claim Boundary",
        "",
        "Toy command domain only. No real quantum claim, quantum hardware requirement, general NLU, full PilotPulse, production VRAXION/INSTNCT, biology, or consciousness claim.",
    ]
    text = "\n".join(lines) + "\n"
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(text)
    DOC_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC_REPORT.write_text(text)


def main() -> None:
    args = parse_args()
    rows = evaluate()
    metrics = arm_metrics(rows)
    labels = verdict(rows, metrics)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_dir / "metrics.csv")
    write_failures(rows, args.out_dir / "failure_examples.jsonl")
    summary = {
        "metrics": metrics,
        "verdict": labels,
        "failure_count": sum(1 for row in rows if row["failure_type"] != "none"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_report(rows, metrics, labels, args.out_dir / "report.md")
    print(json.dumps({
        "doc_report": str(DOC_REPORT),
        "metrics": str(args.out_dir / "metrics.csv"),
        "report": str(args.out_dir / "report.md"),
        "failures": str(args.out_dir / "failure_examples.jsonl"),
        "verdict": labels,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

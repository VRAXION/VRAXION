#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
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
import run_pilot_sensor_operator_state_probe as opstate


DEFAULT_OUT = ROOT / "output" / "pilot_sensor_operator_trajectory_001"
DOC_REPORT = ROOT / "docs" / "research" / "PILOT_SENSOR_OPERATOR_TRAJECTORY_001_RESULT.md"

HARD_NONE = "NONE"
PROV_NONE = "NONE"
COMMIT_CUES = {"END", "COMMIT_SIGNAL"}

BASE_OPERATORS = dict(opstate.OPERATORS)
ARMS = (
    "operator_trajectory",
    "no_provisional_commit_delay",
    "final_only_operator_mapper",
    "no_correction_operator",
    "no_weak_ambiguity_operator",
    "no_negation_operator",
    "no_mention_suppressor",
)


@dataclass(frozen=True)
class TrajectoryCase:
    name: str
    cues: tuple[str, ...]
    expected_final_action: str
    expected_step_actions: tuple[str, ...]
    phenomenon_tag: str


@dataclass(frozen=True)
class StepRecord:
    case: str
    arm: str
    step_index: int
    cue: str
    applied_operator: str
    state: tuple[float, float, float, float]
    evidence: pilot.Evidence
    entropy: float
    purity: float
    state_margin: float
    evidence_margin: float
    provisional_action: str
    hard_commit_action: str
    emitted_action: str
    expected_step_action: str
    premature_hard_commit: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PILOT_SENSOR_OPERATOR_TRAJECTORY_001 temporal operator-state probe.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--jitter-samples", type=int, default=500)
    parser.add_argument("--parameter-search-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def trajectory_cases() -> list[TrajectoryCase]:
    return [
        TrajectoryCase("clean_add", ("ADD_CUE", "END"), "HARD_EXEC_ADD", ("PROVISIONAL_ADD", "HARD_EXEC_ADD"), "known"),
        TrajectoryCase("clean_mul", ("MUL_CUE", "END"), "HARD_EXEC_MUL", ("PROVISIONAL_MUL", "HARD_EXEC_MUL"), "known"),
        TrajectoryCase("weak_add", ("WEAK", "ADD_CUE", "END"), pilot.HOLD_ACTION, (pilot.HOLD_ACTION, pilot.HOLD_ACTION, pilot.HOLD_ACTION), "weak"),
        TrajectoryCase("ambiguous_add_mul", ("ADD_CUE", "AMBIGUITY", "MUL_CUE", "END"), pilot.HOLD_ACTION, ("PROVISIONAL_ADD", pilot.HOLD_ACTION, pilot.HOLD_ACTION, pilot.HOLD_ACTION), "ambiguous"),
        TrajectoryCase("correction_add_to_mul", ("ADD_CUE", "CORRECTION_MUL", "END"), "HARD_EXEC_MUL", ("PROVISIONAL_ADD", "PROVISIONAL_MUL", "HARD_EXEC_MUL"), "correction"),
        TrajectoryCase("correction_mul_to_add", ("MUL_CUE", "CORRECTION_ADD", "END"), "HARD_EXEC_ADD", ("PROVISIONAL_MUL", "PROVISIONAL_ADD", "HARD_EXEC_ADD"), "correction"),
        TrajectoryCase("negated_add", ("ADD_CUE", "NOT_ADD", "END"), pilot.HOLD_ACTION, ("PROVISIONAL_ADD", pilot.HOLD_ACTION, pilot.HOLD_ACTION), "negation"),
        TrajectoryCase("negated_add_then_mul", ("ADD_CUE", "NOT_ADD", "MUL_CUE", "END"), "HARD_EXEC_MUL", ("PROVISIONAL_ADD", pilot.HOLD_ACTION, "PROVISIONAL_MUL", "HARD_EXEC_MUL"), "negation"),
        TrajectoryCase("mention_trap", ("ADD_CUE", "MENTION_CONTEXT", "END"), pilot.HOLD_ACTION, ("PROVISIONAL_ADD", pilot.HOLD_ACTION, pilot.HOLD_ACTION), "mention_trap"),
        TrajectoryCase("unknown_div", ("UNKNOWN_CUE", "END"), pilot.REJECT_ACTION, (pilot.REJECT_ACTION, pilot.REJECT_ACTION), "unknown"),
        TrajectoryCase("multistep_unsupported", ("ADD_CUE", "THEN", "MUL_CUE", "END"), pilot.HOLD_ACTION, ("PROVISIONAL_ADD", pilot.HOLD_ACTION, pilot.HOLD_ACTION, pilot.HOLD_ACTION), "multi_step_unsupported"),
        TrajectoryCase("no_evidence", ("END",), pilot.HOLD_ACTION, (pilot.HOLD_ACTION,), "no_evidence"),
    ]


def cue_to_operator(cue: str) -> str | None:
    return {
        "ADD_CUE": "ADD",
        "MUL_CUE": "MUL",
        "UNKNOWN_CUE": "UNKNOWN",
        "WEAK": "WEAK",
        "AMBIGUITY": "AMBIGUITY",
        "MENTION_CONTEXT": "MENTION",
        "THEN": "MULTI_STEP",
        "NOT_ADD": "NOT_ADD",
        "NOT_MUL": "NOT_MUL",
        "NOT_UNKNOWN": "NOT_UNKNOWN",
        "CORRECTION_ADD": "CORR_ADD",
        "CORRECTION_MUL": "CORR_MUL",
        "CORRECTION_UNKNOWN": "CORR_UNKNOWN",
    }.get(cue)


def ablation_for_arm(arm: str) -> str | None:
    return {
        "operator_trajectory": None,
        "no_provisional_commit_delay": None,
        "final_only_operator_mapper": None,
        "no_correction_operator": "no_correction",
        "no_weak_ambiguity_operator": "no_weak_ambiguity",
        "no_negation_operator": "no_negation",
        "no_mention_suppressor": "no_mention_suppressor",
    }[arm]


def should_skip_operator(operator_name: str, ablation: str | None) -> bool:
    if ablation == "no_correction" and operator_name.startswith("CORR_"):
        return True
    if ablation == "no_weak_ambiguity" and operator_name in {"WEAK", "AMBIGUITY"}:
        return True
    if ablation == "no_negation" and operator_name.startswith("NOT_"):
        return True
    if ablation == "no_mention_suppressor" and operator_name == "MENTION":
        return True
    return False


def guard_to_provisional(action: str) -> str:
    if action == "EXEC_ADD":
        return "PROVISIONAL_ADD"
    if action == "EXEC_MUL":
        return "PROVISIONAL_MUL"
    return action


def guard_to_hard(action: str) -> str:
    if action == "EXEC_ADD":
        return "HARD_EXEC_ADD"
    if action == "EXEC_MUL":
        return "HARD_EXEC_MUL"
    return action


def trajectory_emit(action: str, cue: str, arm: str) -> tuple[str, str, str]:
    if arm == "no_provisional_commit_delay" and action in pilot.EXEC_ACTIONS:
        hard = guard_to_hard(action)
        return hard, PROV_NONE, hard
    if cue in COMMIT_CUES:
        hard = guard_to_hard(action) if action in pilot.EXEC_ACTIONS else action
        return hard, PROV_NONE, hard
    provisional = guard_to_provisional(action)
    hard = HARD_NONE
    return provisional, provisional, hard


def apply_named_operator(
    state: tuple[float, float, float, float],
    operator_name: str,
    operators: dict[str, tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    return opstate.apply_operator(state, operators[operator_name])


def run_case(
    case: TrajectoryCase,
    arm: str,
    operators: dict[str, tuple[float, float, float, float]] | None = None,
) -> list[StepRecord]:
    operators = operators or BASE_OPERATORS
    if arm == "final_only_operator_mapper":
        return run_final_only_case(case, operators)

    ablation = ablation_for_arm(arm)
    state = opstate.INITIAL_STATE
    records: list[StepRecord] = []
    suppressed_labels: set[str] = set()
    for index, cue in enumerate(case.cues):
        operator_name = cue_to_operator(cue)
        applied = "NONE"
        if operator_name is not None and not should_skip_operator(operator_name, ablation):
            state = apply_named_operator(state, operator_name, operators)
            applied = operator_name
            if operator_name == "NOT_ADD":
                suppressed_labels.add("ADD")
            elif operator_name == "NOT_MUL":
                suppressed_labels.add("MUL")
            elif operator_name == "NOT_UNKNOWN":
                suppressed_labels.add("UNKNOWN")
            elif operator_name in {"ADD", "MUL", "UNKNOWN"} and suppressed_labels and operator_name not in suppressed_labels:
                # Temporal counterpart of the final-state contrastive negation rule:
                # after suppressing one mode, a later surviving cue is stronger evidence
                # for the intended route, e.g. "do not add ... multiply".
                state = apply_named_operator(state, operator_name, operators)
                applied = f"{operator_name}+CONTRASTIVE_{operator_name}"
        evidence = (state[0], state[1], state[2])
        guard_action = pilot.guard_policy(evidence)
        emitted, provisional, hard = trajectory_emit(guard_action, cue, arm)
        expected = case.expected_step_actions[index]
        premature = hard.startswith("HARD_EXEC_") and cue not in COMMIT_CUES
        records.append(StepRecord(
            case=case.name,
            arm=arm,
            step_index=index,
            cue=cue,
            applied_operator=applied,
            state=state,
            evidence=evidence,
            entropy=opstate.entropy(state),
            purity=opstate.purity(state),
            state_margin=opstate.margin(state),
            evidence_margin=opstate.margin(evidence),
            provisional_action=provisional,
            hard_commit_action=hard,
            emitted_action=emitted,
            expected_step_action=expected,
            premature_hard_commit=premature,
        ))
    return records


def run_final_only_case(
    case: TrajectoryCase,
    operators: dict[str, tuple[float, float, float, float]],
) -> list[StepRecord]:
    state = opstate.INITIAL_STATE
    for cue in case.cues:
        operator_name = cue_to_operator(cue)
        if operator_name is not None:
            state = apply_named_operator(state, operator_name, operators)
    evidence = (state[0], state[1], state[2])
    guard_action = pilot.guard_policy(evidence)
    final_action = guard_to_hard(guard_action) if guard_action in pilot.EXEC_ACTIONS else guard_action
    records: list[StepRecord] = []
    for index, cue in enumerate(case.cues):
        is_final = index == len(case.cues) - 1
        emitted = final_action if is_final else "FINAL_ONLY_NO_STEP_ACTION"
        hard = final_action if is_final else HARD_NONE
        expected = case.expected_step_actions[index]
        records.append(StepRecord(
            case=case.name,
            arm="final_only_operator_mapper",
            step_index=index,
            cue=cue,
            applied_operator="FINAL_ONLY",
            state=state,
            evidence=evidence,
            entropy=opstate.entropy(state),
            purity=opstate.purity(state),
            state_margin=opstate.margin(state),
            evidence_margin=opstate.margin(evidence),
            provisional_action=PROV_NONE,
            hard_commit_action=hard,
            emitted_action=emitted,
            expected_step_action=expected,
            premature_hard_commit=False,
        ))
    return records


def final_record(records: list[StepRecord]) -> StepRecord:
    return records[-1]


def record_to_row(record: StepRecord, case: TrajectoryCase) -> dict[str, object]:
    return {
        "arm": record.arm,
        "case": record.case,
        "phenomenon_tag": case.phenomenon_tag,
        "step_index": record.step_index,
        "cue": record.cue,
        "applied_operator": record.applied_operator,
        "state": json.dumps(tuple(round(value, 6) for value in record.state)),
        "evidence": json.dumps(tuple(round(value, 6) for value in record.evidence)),
        "entropy": record.entropy,
        "purity": record.purity,
        "state_margin": record.state_margin,
        "evidence_margin": record.evidence_margin,
        "provisional_action": record.provisional_action,
        "hard_commit_action": record.hard_commit_action,
        "emitted_action": record.emitted_action,
        "expected_step_action": record.expected_step_action,
        "step_action_correct": record.emitted_action == record.expected_step_action,
        "expected_final_action": case.expected_final_action,
        "premature_hard_commit": record.premature_hard_commit,
    }


def evaluate_rows(operators: dict[str, tuple[float, float, float, float]] | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cases = trajectory_cases()
    for arm in ARMS:
        for case in cases:
            records = run_case(case, arm, operators)
            final = final_record(records)
            final_correct = final.emitted_action == case.expected_final_action
            false_execution = final.emitted_action.startswith("HARD_EXEC_") and not case.expected_final_action.startswith("HARD_EXEC_")
            for record in records:
                row = record_to_row(record, case)
                row["final_action"] = final.emitted_action
                row["final_action_correct"] = final_correct
                row["false_execution"] = false_execution
                row["failure_type"] = classify_failure(row)
                rows.append(row)
    return rows


def classify_failure(row: dict[str, object]) -> str:
    if bool(row["step_action_correct"]) and bool(row["final_action_correct"]) and not bool(row["false_execution"]) and not bool(row["premature_hard_commit"]):
        return "none"
    tag = str(row["phenomenon_tag"])
    if bool(row["premature_hard_commit"]):
        return "premature_hard_commit"
    if bool(row["false_execution"]):
        return "false_execution"
    if not bool(row["final_action_correct"]):
        if tag == "correction":
            return "correction_recollapse_error"
        if tag == "negation":
            return "negation_suppression_error"
        if tag == "mention_trap":
            return "mention_suppression_error"
        if tag in {"weak", "ambiguous", "multi_step_unsupported"}:
            return "uncertainty_hold_error"
        if tag == "unknown":
            return "unknown_reject_error"
        return "final_action_error"
    if not bool(row["step_action_correct"]):
        return "per_step_timing_error"
    return "none"


def frac(rows: list[dict[str, object]], field: str) -> float:
    if not rows:
        return 0.0
    return float(sum(bool(row[field]) for row in rows) / len(rows))


def arm_rows(rows: list[dict[str, object]], arm: str) -> list[dict[str, object]]:
    return [row for row in rows if row["arm"] == arm]


def final_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if row["cue"] in COMMIT_CUES]


def case_final_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[tuple[str, str]] = set()
    finals: list[dict[str, object]] = []
    for row in rows:
        key = (str(row["arm"]), str(row["case"]))
        if key in seen:
            continue
        case_rows = [item for item in rows if item["arm"] == row["arm"] and item["case"] == row["case"]]
        finals.append(case_rows[-1])
        seen.add(key)
    return finals


def tag_accuracy(finals: list[dict[str, object]], tag: str) -> float:
    subset = [row for row in finals if row["phenomenon_tag"] == tag]
    return frac(subset, "final_action_correct")


def metrics_for_arm(rows: list[dict[str, object]]) -> dict[str, float]:
    finals = case_final_rows(rows)
    commit_rows = [row for row in finals if str(row["final_action"]).startswith("HARD_EXEC_")]
    return {
        "final_action_accuracy": frac(finals, "final_action_correct"),
        "per_step_expected_action_accuracy": frac(rows, "step_action_correct"),
        "premature_hard_commit_rate": float(sum(bool(row["premature_hard_commit"]) for row in rows) / len(rows)) if rows else 0.0,
        "false_execution_rate": float(sum(bool(row["false_execution"]) for row in finals) / len(finals)) if finals else 0.0,
        "correction_recollapse_accuracy": tag_accuracy(finals, "correction"),
        "negation_suppression_accuracy": tag_accuracy(finals, "negation"),
        "mention_suppression_accuracy": tag_accuracy(finals, "mention_trap"),
        "weak_hold_accuracy": tag_accuracy(finals, "weak"),
        "ambiguous_hold_accuracy": tag_accuracy(finals, "ambiguous"),
        "unknown_reject_accuracy": tag_accuracy(finals, "unknown"),
        "mean_commit_step": mean(float(row["step_index"]) for row in commit_rows) if commit_rows else 0.0,
        "state_margin_at_commit": mean(float(row["state_margin"]) for row in commit_rows) if commit_rows else 0.0,
        "purity_at_commit": mean(float(row["purity"]) for row in commit_rows) if commit_rows else 0.0,
        "entropy_at_commit": mean(float(row["entropy"]) for row in commit_rows) if commit_rows else 0.0,
    }


def aggregate(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    return {arm: metrics_for_arm(arm_rows(rows, arm)) for arm in ARMS}


def pass_main(metrics: dict[str, float]) -> bool:
    return (
        metrics["final_action_accuracy"] == 1.0
        and metrics["false_execution_rate"] == 0.0
        and metrics["premature_hard_commit_rate"] == 0.0
        and metrics["correction_recollapse_accuracy"] == 1.0
        and metrics["weak_hold_accuracy"] == 1.0
        and metrics["ambiguous_hold_accuracy"] == 1.0
    )


def jittered_operators(rng: random.Random, jitter: float) -> dict[str, tuple[float, float, float, float]]:
    out: dict[str, tuple[float, float, float, float]] = {}
    for name, values in BASE_OPERATORS.items():
        out[name] = tuple(value * rng.uniform(1.0 - jitter, 1.0 + jitter) for value in values)
    return out


def robustness(jitter_samples: int, seed: int) -> dict[str, object]:
    rng = random.Random(seed)
    levels = [0.10, 0.20, 0.30]
    output: dict[str, object] = {}
    for level in levels:
        passes = 0
        false_execution = 0
        premature = 0
        weakest_case: dict[str, int] = {}
        min_margin: dict[str, float] = {}
        min_purity: dict[str, float] = {}
        for _ in range(jitter_samples):
            rows = evaluate_rows(jittered_operators(rng, level))
            main_rows = arm_rows(rows, "operator_trajectory")
            metrics = metrics_for_arm(main_rows)
            if pass_main(metrics):
                passes += 1
            if metrics["false_execution_rate"] > 0.0:
                false_execution += 1
            if metrics["premature_hard_commit_rate"] > 0.0:
                premature += 1
            for row in case_final_rows(main_rows):
                case = str(row["case"])
                if not bool(row["final_action_correct"]):
                    weakest_case[case] = weakest_case.get(case, 0) + 1
                min_margin[case] = min(min_margin.get(case, 999.0), float(row["state_margin"]))
                min_purity[case] = min(min_purity.get(case, 999.0), float(row["purity"]))
        output[f"{int(level * 100)}pct"] = {
            "samples": jitter_samples,
            "pass_rate_under_jitter": passes / jitter_samples if jitter_samples else 0.0,
            "false_execution_under_jitter": false_execution / jitter_samples if jitter_samples else 0.0,
            "premature_commit_under_jitter": premature / jitter_samples if jitter_samples else 0.0,
            "weakest_case": sorted(weakest_case.items(), key=lambda item: (-item[1], item[0]))[:5],
            "min_state_margin_by_case": dict(sorted(min_margin.items())),
            "min_purity_by_case": dict(sorted(min_purity.items())),
        }
    return output


def robustness_labels(robust: dict[str, object], main_pass: bool) -> list[str]:
    if not main_pass:
        return []
    ten = robust["10pct"]  # type: ignore[index]
    twenty = robust["20pct"]  # type: ignore[index]
    thirty = robust["30pct"]  # type: ignore[index]
    ten_ok = ten["pass_rate_under_jitter"] >= 0.95 and ten["false_execution_under_jitter"] <= 0.01 and ten["premature_commit_under_jitter"] <= 0.01  # type: ignore[index]
    twenty_ok = twenty["pass_rate_under_jitter"] >= 0.95 and twenty["false_execution_under_jitter"] <= 0.01 and twenty["premature_commit_under_jitter"] <= 0.01  # type: ignore[index]
    thirty_ok = thirty["pass_rate_under_jitter"] >= 0.95 and thirty["false_execution_under_jitter"] <= 0.01 and thirty["premature_commit_under_jitter"] <= 0.01  # type: ignore[index]
    if ten_ok and twenty_ok:
        return ["OPERATOR_STATE_ROBUST"] if thirty_ok else ["OPERATOR_STATE_ROBUST", "OPERATOR_FORM_VALID_BUT_CALIBRATION_SENSITIVE"]
    if ten_ok:
        return ["OPERATOR_FORM_VALID_BUT_CALIBRATION_SENSITIVE"]
    return ["OPERATOR_STATE_KNIFE_EDGE"]


def parameter_search(samples: int, seed: int) -> dict[str, object]:
    if samples <= 0:
        return {}
    rng = random.Random(seed + 17)
    best: list[tuple[tuple[float, float, float, float, float], dict[str, tuple[float, float, float, float]], dict[str, float]]] = []
    for _ in range(samples):
        operators = jittered_operators(rng, 0.20)
        rows = arm_rows(evaluate_rows(operators), "operator_trajectory")
        metrics = metrics_for_arm(rows)
        finals = case_final_rows(rows)
        min_margin = min(float(row["state_margin"]) for row in finals) if finals else 0.0
        score = (
            metrics["false_execution_rate"],
            metrics["premature_hard_commit_rate"],
            -metrics["final_action_accuracy"],
            -min_margin,
            -metrics["per_step_expected_action_accuracy"],
        )
        best.append((score, operators, metrics))
    best.sort(key=lambda item: item[0])
    return {
        "samples": samples,
        "objective_order": [
            "minimize false_execution",
            "minimize premature_hard_commit",
            "maximize final_action_accuracy",
            "maximize margin safety",
            "prefer HOLD over wrong EXEC",
        ],
        "best": [
            {
                "score": item[0],
                "metrics": item[2],
                "operators": {name: tuple(round(value, 4) for value in values) for name, values in item[1].items()},
            }
            for item in best[:5]
        ],
    }


def verdict(metrics: dict[str, dict[str, float]], robust: dict[str, object]) -> list[str]:
    labels: list[str] = []
    main_pass = pass_main(metrics["operator_trajectory"])
    labels.append("OPERATOR_TRAJECTORY_POSITIVE" if main_pass else "OPERATOR_STATE_ONLY_FINAL_MAPPER")
    delay_metrics = metrics["no_provisional_commit_delay"]
    if delay_metrics["correction_recollapse_accuracy"] < metrics["operator_trajectory"]["correction_recollapse_accuracy"] or delay_metrics["premature_hard_commit_rate"] > 0.0:
        labels.append("COLLAPSE_DELAY_CAUSAL")
    else:
        labels.append("COLLAPSE_DELAY_NOT_CAUSAL")
    if metrics["final_only_operator_mapper"]["per_step_expected_action_accuracy"] < metrics["operator_trajectory"]["per_step_expected_action_accuracy"]:
        labels.append("FINAL_ONLY_LACKS_TRAJECTORY_TIMING")
    labels += robustness_labels(robust, main_pass)
    return labels


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def failure_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if row["failure_type"] != "none"]


def report_failure_lines(rows: list[dict[str, object]], limit: int = 35) -> list[str]:
    failures = failure_rows(rows)
    if not failures:
        return ["No failures."]
    lines: list[str] = []
    for row in failures[:limit]:
        lines.append(
            f"- `{row['arm']}` `{row['case']}` step `{row['step_index']}` `{row['cue']}`: "
            f"`{row['emitted_action']}` expected `{row['expected_step_action']}`; "
            f"final `{row['final_action']}` expected `{row['expected_final_action']}`; failure `{row['failure_type']}`."
        )
    if len(failures) > limit:
        lines.append(f"- ... {len(failures) - limit} more in `failure_examples.jsonl`.")
    return lines


def write_report(
    rows: list[dict[str, object]],
    metrics: dict[str, dict[str, float]],
    robust: dict[str, object],
    search: dict[str, object],
    labels: list[str],
    output_report: Path,
) -> None:
    lines = [
        "# PILOT_SENSOR_OPERATOR_TRAJECTORY_001 Result",
        "",
        "## Goal",
        "",
        "Test whether the operator-state/collapse formalism adds temporal Pilot value beyond the final v0 mapper.",
        "",
        "This probe separates provisional hypotheses from hard commits. Hard execution is allowed only at `END` or explicit commit signals in the main arm.",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(metrics, indent=2, sort_keys=True),
        "```",
        "",
        "## Robustness",
        "",
        "```json",
        json.dumps(robust, indent=2, sort_keys=True),
        "```",
    ]
    if search:
        lines += [
            "",
            "## Parameter Search Diagnostic",
            "",
            "```json",
            json.dumps(search, indent=2, sort_keys=True),
            "```",
        ]
    lines += [
        "",
        "## Verdict",
        "",
        "```json",
        json.dumps(labels, indent=2),
        "```",
        "",
        "## Failure Examples",
        "",
        *report_failure_lines(rows),
        "",
        "## Interpretation",
        "",
        "A positive trajectory result means the operator-state formalism supports time-resolved evidence accumulation, provisional hypotheses, delayed hard commit, and re-collapse after correction in this toy command domain.",
        "",
        "The `no_provisional_commit_delay` ablation tests whether delaying hard collapse is causal. The `final_only_operator_mapper` control tests whether final-state accuracy alone is insufficient evidence for trajectory behavior.",
        "",
        "## Claim Boundary",
        "",
        "Toy command domain only. No real quantum behavior, quantum hardware requirement, general NLU, full PilotPulse, production VRAXION/INSTNCT, biology, or consciousness claim.",
    ]
    text = "\n".join(lines) + "\n"
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(text)
    DOC_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC_REPORT.write_text(text)


def main() -> None:
    args = parse_args()
    rows = evaluate_rows()
    metrics = aggregate(rows)
    robust = robustness(args.jitter_samples, args.seed)
    labels = verdict(metrics, robust)
    need_search = (
        "OPERATOR_STATE_KNIFE_EDGE" in labels
        or any(str(label).startswith("OPERATOR_FORM_VALID_BUT_CALIBRATION_SENSITIVE") for label in labels)
        or any(level["false_execution_under_jitter"] > 0.0 for level in robust.values())  # type: ignore[union-attr]
    )
    search = parameter_search(args.parameter_search_samples, args.seed) if need_search and args.parameter_search_samples > 0 else {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_dir / "metrics.csv")
    write_jsonl(rows, args.out_dir / "trajectory_traces.jsonl")
    write_jsonl(failure_rows(rows), args.out_dir / "failure_examples.jsonl")
    (args.out_dir / "robustness.json").write_text(json.dumps(robust, indent=2, sort_keys=True) + "\n")
    summary = {
        "metrics": metrics,
        "robustness": robust,
        "parameter_search": search,
        "verdict": labels,
        "failure_count": len(failure_rows(rows)),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_report(rows, metrics, robust, search, labels, args.out_dir / "report.md")
    print(json.dumps({
        "doc_report": str(DOC_REPORT),
        "metrics": str(args.out_dir / "metrics.csv"),
        "traces": str(args.out_dir / "trajectory_traces.jsonl"),
        "failures": str(args.out_dir / "failure_examples.jsonl"),
        "robustness": str(args.out_dir / "robustness.json"),
        "verdict": labels,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

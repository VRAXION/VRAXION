#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_pilot_sensor_augmented_robustness_probe as robustness
import run_pilot_sensor_factor_heldout_probe as factor_probe
import run_pilot_sensor_lexicon_extension_probe as lexicon_probe
import run_pilot_sensor_locked_skill_integration_probe as locked_skill
import run_pilot_sensor_probe as sensor_probe
import run_pilot_sensor_scope_stack_nightly as nightly


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "output" / "pilot_sensor_v0_regression_001"
DOC_REPORT = ROOT / "docs" / "research" / "PILOT_SENSOR_V0_REGRESSION_001_RESULT.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PILOT_SENSOR_V0_REGRESSION_001 parser-assisted baseline regression.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def v0_evidence(text: str):
    return sensor_probe.structured_rule_sensor(lexicon_probe.normalize_aliases(text))


def execution_cases() -> list[locked_skill.ExecutionCase]:
    rows: list[locked_skill.ExecutionCase] = []
    for idx, row in enumerate(robustness.stress_cases()):
        rows.append(locked_skill.from_sensor_case(row, 10 + (idx % 5)))
    for idx, row in enumerate(factor_probe.factor_eval_cases()):
        if row.phenomenon_tag == nightly.DIAGNOSTIC_TAG:
            continue
        rows.append(locked_skill.from_sensor_case(row, 20 + (idx % 7)))
    for idx, row in enumerate(lexicon_probe.cases()):
        rows.append(locked_skill.ExecutionCase(row.split, row.name, row.text, 30 + (idx % 3), row.expected, row.phenomenon_tag))
    return rows


def evaluate() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    primitive_ok = all(
        locked_skill.add_module(v, n) == v + n and locked_skill.mul_module(v, n) == v * n
        for v in range(-5, 6)
        for n in range(1, 6)
    )
    for row in execution_cases():
        evidence = v0_evidence(row.text)
        action = sensor_probe.policy_action(evidence, "evidence_strength_margin_guard")
        result = locked_skill.execute_action(action, row.value, row.text)
        exp_result = locked_skill.expected_result(row)
        rows.append({
            "model_name": "pilot_sensor_v0",
            "split": row.split,
            "case": row.name,
            "text": row.text,
            "value": row.value,
            "operand": locked_skill.extract_operand(row.text),
            "phenomenon_tag": row.phenomenon_tag,
            "expected_action": "|".join(row.expected),
            "student_action": action,
            "student_evidence": json.dumps([round(float(x), 4) for x in evidence]),
            "expected_result": exp_result,
            "student_result": result,
            "action_correct": action in row.expected,
            "result_correct": (exp_result is None and result is None) or (exp_result is not None and result == exp_result),
            "false_execution": sensor_probe.is_false_commit(action, row.expected),
            "missed_execution": sensor_probe.is_missed_execute(action, row.expected),
            "primitive_accuracy_before": primitive_ok,
            "primitive_accuracy_after": primitive_ok,
            "primitive_drift": 0.0 if primitive_ok else 1.0,
        })
    return rows


def fraction(rows: list[dict[str, object]], predicate, field: str = "action_correct") -> float:
    subset = [row for row in rows if predicate(row)]
    if not subset:
        return 0.0
    return float(sum(bool(row[field]) for row in subset) / len(subset))


def metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    expected_exec = [row for row in rows if str(row["expected_action"]).startswith("EXEC_")]
    return {
        "action_accuracy": fraction(rows, lambda _: True),
        "result_accuracy": fraction(rows, lambda _: True, "result_correct"),
        "exec_result_accuracy": fraction(expected_exec, lambda _: True, "result_correct"),
        "false_execution_rate": fraction(rows, lambda _: True, "false_execution"),
        "missed_execution_rate": fraction(rows, lambda _: True, "missed_execution"),
        "known_accuracy": fraction(rows, lambda row: row["phenomenon_tag"] == "known"),
        "weak_accuracy": fraction(rows, lambda row: row["phenomenon_tag"] == "weak"),
        "ambiguous_accuracy": fraction(rows, lambda row: row["phenomenon_tag"] == "ambiguous"),
        "negation_accuracy": fraction(rows, lambda row: row["phenomenon_tag"] == "negation"),
        "correction_accuracy": fraction(rows, lambda row: row["phenomenon_tag"] == "correction"),
        "mention_accuracy": fraction(rows, lambda row: row["phenomenon_tag"] == "mention_trap"),
        "strict_synonym_accuracy": fraction(rows, lambda row: row["phenomenon_tag"] == "strict_synonym"),
        "primitive_drift": mean(float(row["primitive_drift"]) for row in rows),
    }


def verdict(m: dict[str, float]) -> list[str]:
    if (
        m["action_accuracy"] == 1.0
        and m["result_accuracy"] == 1.0
        and m["false_execution_rate"] == 0.0
        and m["missed_execution_rate"] == 0.0
        and m["primitive_drift"] == 0.0
    ):
        return ["PILOT_SENSOR_V0_REGRESSION_PASS"]
    return ["PILOT_SENSOR_V0_REGRESSION_FAIL"]


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows: list[dict[str, object]], m: dict[str, float], labels: list[str], output_report: Path) -> None:
    failures = [row for row in rows if not bool(row["action_correct"]) or not bool(row["result_correct"])]
    lines = [
        "# PILOT_SENSOR_V0_REGRESSION_001 Result",
        "",
        "## Goal",
        "",
        "Lock a parser-assisted PilotSensor v0 regression baseline after the nightly sensor probes.",
        "",
        "## Pipeline",
        "",
        "```text",
        "raw command text",
        "-> alias normalizer",
        "-> structured scope-aware sensor",
        "-> fixed evidence strength+margin guard",
        "-> locked ADD/MUL skill execution or HOLD/REJECT",
        "```",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(m, indent=2, sort_keys=True),
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
    ]
    if failures:
        for row in failures:
            lines.append(
                f"- `{row['split']}/{row['phenomenon_tag']}`: {row['text']} -> expected "
                f"`{row['expected_action']}`/`{row['expected_result']}`, got "
                f"`{row['student_action']}`/`{row['student_result']}`."
            )
    else:
        lines.append("No failures.")
    lines += [
        "",
        "## Interpretation",
        "",
        "This is the recommended v0 command sensor baseline. It is parser-assisted, not learned NLU.",
        "Learned raw-text sensors should be measured against this baseline, especially on factor-heldout scope combinations.",
        "",
        "## Claim Boundary",
        "",
        "No general NLU, full PilotPulse, production VRAXION/INSTNCT, or consciousness claim.",
    ]
    text = "\n".join(lines) + "\n"
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(text)
    DOC_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC_REPORT.write_text(text)


def main() -> None:
    args = parse_args()
    rows = evaluate()
    m = metrics(rows)
    labels = verdict(m)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_dir / "metrics.csv")
    (args.out_dir / "summary.json").write_text(json.dumps({"metrics": m, "verdict": labels}, indent=2, sort_keys=True) + "\n")
    write_report(rows, m, labels, args.out_dir / "report.md")
    print(json.dumps({
        "doc_report": str(DOC_REPORT),
        "metrics": str(args.out_dir / "metrics.csv"),
        "report": str(args.out_dir / "report.md"),
        "verdict": labels,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

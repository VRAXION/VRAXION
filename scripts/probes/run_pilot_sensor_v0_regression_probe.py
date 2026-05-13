#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_pilot_sensor_augmented_robustness_probe as robustness
import run_pilot_sensor_factor_heldout_probe as factor_probe

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import pilot_sensor_v0 as pilot


DEFAULT_OUT = ROOT / "output" / "pilot_sensor_v0_regression_001"
DOC_REPORT = ROOT / "docs" / "research" / "PILOT_SENSOR_V0_REGRESSION_001_RESULT.md"


@dataclass(frozen=True)
class ExecutionCase:
    split: str
    name: str
    text: str
    value: int
    expected: tuple[str, ...]
    phenomenon_tag: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PILOT_SENSOR_V0_REGRESSION_001 parser-assisted baseline regression.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def alias_cases() -> list[ExecutionCase]:
    return [
        ExecutionCase("strict_alias", "increment", "increment by 9", 30, ("EXEC_ADD",), "strict_synonym"),
        ExecutionCase("strict_alias", "raise_value", "raise the value by 9", 31, ("EXEC_ADD",), "strict_synonym"),
        ExecutionCase("strict_alias", "product", "product with 9", 32, ("EXEC_MUL",), "strict_synonym"),
        ExecutionCase("strict_alias", "halve", "halve it", 30, ("REJECT_UNKNOWN",), "strict_synonym"),
        ExecutionCase("strict_alias", "exponentiate", "exponentiate by 3", 31, ("REJECT_UNKNOWN",), "strict_synonym"),
        ExecutionCase("scope_alias", "mention_increment", "the word increment appears in the note", 32, ("HOLD_ASK_RESEARCH",), "mention_trap"),
        ExecutionCase("scope_alias", "neg_increment", "do not increment by 3", 30, ("HOLD_ASK_RESEARCH",), "negation"),
        ExecutionCase("scope_alias", "neg_product_then_add", "do not product with 3, add 3 instead", 31, ("EXEC_ADD",), "negation"),
        ExecutionCase("scope_alias", "weak_increment", "maybe increment by 3", 32, ("HOLD_ASK_RESEARCH",), "weak"),
        ExecutionCase("scope_alias", "amb_increment_product", "increment or product with 3", 30, ("HOLD_ASK_RESEARCH",), "ambiguous"),
        ExecutionCase("scope_alias", "correction_product_increment", "product with 3. correction: increment by 3", 31, ("EXEC_ADD",), "correction"),
    ]


def from_sensor_case(row, value: int) -> ExecutionCase:
    return ExecutionCase(row.split, row.name, row.text, value, row.expected, row.phenomenon_tag)


def execution_cases() -> list[ExecutionCase]:
    rows: list[ExecutionCase] = []
    for idx, row in enumerate(robustness.stress_cases()):
        rows.append(from_sensor_case(row, 10 + (idx % 5)))
    for idx, row in enumerate(factor_probe.factor_eval_cases()):
        if row.phenomenon_tag == "strict_unseen_synonym":
            continue
        rows.append(from_sensor_case(row, 20 + (idx % 7)))
    rows += alias_cases()
    return rows


def expected_result(case: ExecutionCase) -> int | None:
    action = case.expected[0]
    operand = pilot.extract_operand(pilot.normalize_aliases(case.text))
    if operand is None:
        return None
    if action == "EXEC_ADD":
        return case.value + operand
    if action == "EXEC_MUL":
        return case.value * operand
    return None


def is_false_execution(action: str, expected: tuple[str, ...]) -> bool:
    return action in pilot.EXEC_ACTIONS and all(not item.startswith("EXEC_") for item in expected)


def is_missed_execution(action: str, expected: tuple[str, ...]) -> bool:
    return any(item.startswith("EXEC_") for item in expected) and action not in pilot.EXEC_ACTIONS


def evaluate() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    primitive_ok = all(
        pilot.execute_locked_skill("EXEC_ADD", v, n) == v + n and pilot.execute_locked_skill("EXEC_MUL", v, n) == v * n
        for v in range(-5, 6)
        for n in range(1, 6)
    )
    for row in execution_cases():
        pilot_result = pilot.run_pilot_sensor_v0(row.text, row.value)
        exp_result = expected_result(row)
        rows.append({
            "model_name": "pilot_sensor_v0",
            "split": row.split,
            "case": row.name,
            "text": row.text,
            "value": row.value,
            "normalized_text": pilot_result.normalized_text,
            "operand": pilot_result.operand,
            "phenomenon_tag": row.phenomenon_tag,
            "expected_action": "|".join(row.expected),
            "student_action": pilot_result.action,
            "student_evidence": json.dumps([round(float(x), 4) for x in pilot_result.evidence]),
            "expected_result": exp_result,
            "student_result": pilot_result.result,
            "action_correct": pilot_result.action in row.expected,
            "result_correct": (exp_result is None and pilot_result.result is None) or (exp_result is not None and pilot_result.result == exp_result),
            "false_execution": is_false_execution(pilot_result.action, row.expected),
            "missed_execution": is_missed_execution(pilot_result.action, row.expected),
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

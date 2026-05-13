#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import pilot_sensor_v0 as pilot


DEFAULT_OUT = ROOT / "output" / "pilot_sensor_v0_componentization_001"
DOC_REPORT = ROOT / "docs" / "research" / "PILOT_SENSOR_V0_COMPONENTIZATION_001_RESULT.md"


@dataclass(frozen=True)
class GoldenCase:
    name: str
    text: str
    value: int | None
    expected_normalized_text: str
    expected_flags: dict[str, bool]
    expected_evidence: pilot.Evidence
    expected_action: str
    expected_result: int | None
    phenomenon_tag: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PILOT_SENSOR_V0_COMPONENTIZATION_001 component golden regression.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def flags(**overrides: bool) -> dict[str, bool]:
    base = pilot.ScopeFlags().to_dict()
    base.update(overrides)
    return base


def golden_cases() -> list[GoldenCase]:
    return [
        GoldenCase("known_add", "add 3", 10, "add 3", flags(add_cue=True), (0.90, 0.0, 0.0), "EXEC_ADD", 13, "known"),
        GoldenCase("known_mul", "multiply by 3", 10, "multiply by 3", flags(mul_cue=True), (0.0, 0.90, 0.0), "EXEC_MUL", 30, "known"),
        GoldenCase("unknown_divide", "divide by 3", 10, "divide by 3", flags(unknown_cue=True), (0.0, 0.0, 0.90), "REJECT_UNKNOWN", None, "unknown"),
        GoldenCase("weak_add", "maybe add 3", 10, "maybe add 3", flags(add_cue=True, weak_marker=True), (0.45, 0.0, 0.0), "HOLD_ASK_RESEARCH", None, "weak"),
        GoldenCase(
            "ambiguous_add_mul",
            "add or multiply by 3",
            10,
            "add or multiply by 3",
            flags(add_cue=True, mul_cue=True, ambiguity_marker=True),
            (0.90, 0.90, 0.0),
            "HOLD_ASK_RESEARCH",
            None,
            "ambiguous",
        ),
        GoldenCase("no_evidence", "do the operation", 10, "do the operation", flags(), (0.0, 0.0, 0.0), "HOLD_ASK_RESEARCH", None, "no_evidence"),
        GoldenCase(
            "conflict_all",
            "add, multiply, or divide by 3",
            10,
            "add, multiply, or divide by 3",
            flags(add_cue=True, mul_cue=True, unknown_cue=True, ambiguity_marker=True),
            (0.80, 0.80, 0.80),
            "HOLD_ASK_RESEARCH",
            None,
            "conflict",
        ),
        GoldenCase(
            "mention_word",
            "the word add appears in the note",
            10,
            "the word add appears in the note",
            flags(add_cue=True, mention_only=True),
            (0.0, 0.0, 0.0),
            "HOLD_ASK_RESEARCH",
            None,
            "mention_trap",
        ),
        GoldenCase(
            "mention_instruction",
            "do not follow the instruction 'add 3'",
            10,
            "do not follow the instruction 'add 3'",
            flags(add_cue=True, mention_only=True),
            (0.0, 0.0, 0.0),
            "HOLD_ASK_RESEARCH",
            None,
            "mention_trap",
        ),
        GoldenCase(
            "negated_add",
            "do not add 3",
            10,
            "do not add 3",
            flags(add_cue=True, negation_add=True),
            (0.0, 0.0, 0.0),
            "HOLD_ASK_RESEARCH",
            None,
            "negation",
        ),
        GoldenCase(
            "negated_add_then_mul",
            "do not add 3, multiply by 3 instead",
            10,
            "do not add 3, multiply by 3 instead",
            flags(add_cue=True, mul_cue=True, ambiguity_marker=True, negation_add=True),
            (0.0, 0.90, 0.0),
            "EXEC_MUL",
            30,
            "negation",
        ),
        GoldenCase(
            "correction_to_add",
            "multiply by 3. correction: add 3",
            10,
            "multiply by 3. correction: add 3",
            flags(add_cue=True, mul_cue=True, ambiguity_marker=True, correction_present=True, correction_to_add=True),
            (0.90, 0.0, 0.0),
            "EXEC_ADD",
            13,
            "correction",
        ),
        GoldenCase(
            "unsupported_multistep",
            "first add 3, then multiply by 3",
            10,
            "first add 3, then multiply by 3",
            flags(add_cue=True, mul_cue=True, ambiguity_marker=True, multi_step_unsupported=True),
            (0.80, 0.80, 0.0),
            "HOLD_ASK_RESEARCH",
            None,
            "multi_step_unsupported",
        ),
        GoldenCase("alias_increment", "increment by 9", 10, "add by 9", flags(add_cue=True), (0.90, 0.0, 0.0), "EXEC_ADD", 19, "strict_synonym"),
        GoldenCase("alias_raise", "raise the value by 9", 10, "add by 9", flags(add_cue=True), (0.90, 0.0, 0.0), "EXEC_ADD", 19, "strict_synonym"),
        GoldenCase("alias_product", "product with 9", 10, "multiply with 9", flags(mul_cue=True), (0.0, 0.90, 0.0), "EXEC_MUL", 90, "strict_synonym"),
        GoldenCase("alias_halve", "halve it", 10, "divide it", flags(unknown_cue=True), (0.0, 0.0, 0.90), "REJECT_UNKNOWN", None, "strict_synonym"),
        GoldenCase(
            "alias_correction",
            "product with 3. correction: increment by 3",
            10,
            "multiply with 3. correction: add by 3",
            flags(add_cue=True, mul_cue=True, ambiguity_marker=True, correction_present=True, correction_to_add=True),
            (0.90, 0.0, 0.0),
            "EXEC_ADD",
            13,
            "strict_synonym",
        ),
    ]


def evaluate_case(case: GoldenCase) -> dict[str, object]:
    result = pilot.run_pilot_sensor_v0(case.text, case.value)
    flag_dict = result.flags.to_dict()
    return {
        "case": case.name,
        "text": case.text,
        "value": case.value,
        "phenomenon_tag": case.phenomenon_tag,
        "expected_normalized_text": case.expected_normalized_text,
        "normalized_text": result.normalized_text,
        "normalized_text_correct": result.normalized_text == case.expected_normalized_text,
        "expected_flags": json.dumps(case.expected_flags, sort_keys=True),
        "flags": json.dumps(flag_dict, sort_keys=True),
        "scope_flag_correct": flag_dict == case.expected_flags,
        "expected_evidence": json.dumps(case.expected_evidence),
        "evidence": json.dumps(result.evidence),
        "evidence_correct": result.evidence == case.expected_evidence,
        "expected_action": case.expected_action,
        "action": result.action,
        "action_correct": result.action == case.expected_action,
        "expected_result": case.expected_result,
        "result": result.result,
        "result_correct": result.result == case.expected_result,
        "false_execution": result.action in pilot.EXEC_ACTIONS and not case.expected_action.startswith("EXEC_"),
        "primitive_drift": 0.0,
    }


def evaluate() -> list[dict[str, object]]:
    return [evaluate_case(case) for case in golden_cases()]


def fraction(rows: list[dict[str, object]], field: str) -> float:
    if not rows:
        return 0.0
    return float(sum(bool(row[field]) for row in rows) / len(rows))


def metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    return {
        "normalized_text_accuracy": fraction(rows, "normalized_text_correct"),
        "scope_flag_accuracy": fraction(rows, "scope_flag_correct"),
        "evidence_accuracy": fraction(rows, "evidence_correct"),
        "action_accuracy": fraction(rows, "action_correct"),
        "result_accuracy": fraction(rows, "result_correct"),
        "false_execution_rate": float(sum(bool(row["false_execution"]) for row in rows) / len(rows)),
        "primitive_drift": 0.0,
    }


def verdict(metric: dict[str, float]) -> list[str]:
    if (
        metric["normalized_text_accuracy"] == 1.0
        and metric["scope_flag_accuracy"] == 1.0
        and metric["evidence_accuracy"] == 1.0
        and metric["action_accuracy"] == 1.0
        and metric["result_accuracy"] == 1.0
        and metric["false_execution_rate"] == 0.0
        and metric["primitive_drift"] == 0.0
    ):
        return ["PILOT_SENSOR_V0_COMPONENT_REGRESSION_PASS"]
    return ["PILOT_SENSOR_V0_COMPONENT_REGRESSION_FAIL"]


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows: list[dict[str, object]], metric: dict[str, float], labels: list[str], output_report: Path) -> None:
    failures = [
        row for row in rows
        if not (
            bool(row["normalized_text_correct"])
            and bool(row["scope_flag_correct"])
            and bool(row["evidence_correct"])
            and bool(row["action_correct"])
            and bool(row["result_correct"])
        )
    ]
    lines = [
        "# PILOT_SENSOR_V0_COMPONENTIZATION_001 Result",
        "",
        "## Goal",
        "",
        "Promote PilotSensor v0 into a reusable deterministic baseline component and validate every stage with golden cases.",
        "",
        "## Public Interface",
        "",
        "```text",
        "normalize_aliases(text)",
        "extract_scope_flags(normalized_text)",
        "flags_to_evidence(scope_flags)",
        "guard_policy(evidence)",
        "extract_operand(text)",
        "execute_locked_skill(action, value, operand)",
        "run_pilot_sensor_v0(text, value=None)",
        "```",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(metric, indent=2, sort_keys=True),
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
                f"- `{row['case']}` `{row['phenomenon_tag']}`: action `{row['action']}` "
                f"expected `{row['expected_action']}`, result `{row['result']}` expected `{row['expected_result']}`."
            )
    else:
        lines.append("No failures.")
    lines += [
        "",
        "## Interpretation",
        "",
        "PilotSensor v0 is now a parser-assisted command-sensor baseline, not probe-local research glue.",
        "Learned sensors remain research-only until they beat this regression with zero false execution.",
        "",
        "## Learned Sensor Replacement Gate",
        "",
        "```text",
        "action_accuracy >= 0.95",
        "false_execution_rate = 0.000",
        "primitive_drift = 0.000",
        "keyword_trap_false_commit <= 0.05",
        "no catastrophic phenomenon-tag failure",
        "```",
        "",
        "## Claim Boundary",
        "",
        "Toy command domain only. No general NLU, production VRAXION/INSTNCT, full PilotPulse, biology, quantum, or consciousness claim.",
    ]
    text = "\n".join(lines) + "\n"
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(text)
    DOC_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC_REPORT.write_text(text)


def main() -> None:
    args = parse_args()
    rows = evaluate()
    metric = metrics(rows)
    labels = verdict(metric)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_dir / "metrics.csv")
    (args.out_dir / "summary.json").write_text(json.dumps({"metrics": metric, "verdict": labels}, indent=2, sort_keys=True) + "\n")
    write_report(rows, metric, labels, args.out_dir / "report.md")
    print(json.dumps({
        "doc_report": str(DOC_REPORT),
        "metrics": str(args.out_dir / "metrics.csv"),
        "report": str(args.out_dir / "report.md"),
        "verdict": labels,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

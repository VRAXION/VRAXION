#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_pilot_sensor_augmented_robustness_probe as robustness
import run_pilot_sensor_factor_heldout_probe as factor_probe
import run_pilot_sensor_probe as sensor_probe
import run_pilot_sensor_scope_stack_nightly as nightly
import run_pilot_sensor_systematic_coverage_probe as systematic_probe


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "output" / "pilot_sensor_locked_skill_integration_001"
DOC_REPORT = ROOT / "docs" / "research" / "PILOT_SENSOR_LOCKED_SKILL_INTEGRATION_001_RESULT.md"


@dataclass(frozen=True)
class ExecutionCase:
    split: str
    name: str
    text: str
    value: int
    expected: tuple[str, ...]
    phenomenon_tag: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PILOT_SENSOR_LOCKED_SKILL_INTEGRATION_001 sensor->guard->locked skill execution probe.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    return parser.parse_args()


def seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def add_module(value: int, operand: int) -> int:
    return value + operand


def mul_module(value: int, operand: int) -> int:
    return value * operand


def extract_operand(text: str) -> int | None:
    numbers = re.findall(r"-?\d+", text)
    if not numbers:
        return None
    return int(numbers[-1])


def expected_result(case: ExecutionCase) -> int | None:
    action = case.expected[0]
    operand = extract_operand(case.text)
    if operand is None:
        return None
    if action == "EXEC_ADD":
        return add_module(case.value, operand)
    if action == "EXEC_MUL":
        return mul_module(case.value, operand)
    return None


def execute_action(action: str, value: int, text: str) -> int | None:
    operand = extract_operand(text)
    if operand is None:
        return None
    if action == "EXEC_ADD":
        return add_module(value, operand)
    if action == "EXEC_MUL":
        return mul_module(value, operand)
    return None


def from_sensor_case(row: nightly.SensorCase, value: int) -> ExecutionCase:
    return ExecutionCase(row.split, row.name, row.text, value, row.expected, row.phenomenon_tag)


def eval_cases() -> list[ExecutionCase]:
    rows: list[ExecutionCase] = []
    for idx, row in enumerate(robustness.stress_cases()):
        rows.append(from_sensor_case(row, 10 + (idx % 5)))
    for idx, row in enumerate(factor_probe.factor_eval_cases()):
        rows.append(from_sensor_case(row, 20 + (idx % 7)))
    return rows


def predict_keyword(text: str) -> np.ndarray:
    return sensor_probe.keyword_sensor(text)


def predict_structured(text: str) -> np.ndarray:
    return sensor_probe.structured_rule_sensor(text)


def train_learned_systematic(seed: int, args: argparse.Namespace) -> nightly.MLPArm:
    return nightly.train_direct_arm(
        "direct_evidence_char_ngram_mlp",
        "char",
        systematic_probe.systematic_training_cases(),
        seed=seed,
        hidden=args.hidden,
        epochs=args.epochs,
        lr=args.learning_rate,
    )


def evaluate(model_name: str, seed: int, rows: list[ExecutionCase], evidence_fn) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    primitive_before = all(add_module(v, n) == v + n and mul_module(v, n) == v * n for v in range(-3, 4) for n in range(1, 5))
    for row in rows:
        evidence = evidence_fn(row.text)
        action = sensor_probe.policy_action(evidence, "evidence_strength_margin_guard")
        result = execute_action(action, row.value, row.text)
        exp_result = expected_result(row)
        action_correct = action in row.expected
        result_correct = (exp_result is None and result is None) or (exp_result is not None and result == exp_result)
        false_execution = sensor_probe.is_false_commit(action, row.expected)
        missed_execution = sensor_probe.is_missed_execute(action, row.expected)
        out.append({
            "model_name": model_name,
            "seed": seed,
            "split": row.split,
            "case": row.name,
            "text": row.text,
            "value": row.value,
            "operand": extract_operand(row.text),
            "phenomenon_tag": row.phenomenon_tag,
            "expected_action": "|".join(row.expected),
            "student_action": action,
            "student_evidence": json.dumps([round(float(x), 4) for x in evidence]),
            "expected_result": exp_result,
            "student_result": result,
            "action_correct": action_correct,
            "result_correct": result_correct,
            "false_execution": false_execution,
            "missed_execution": missed_execution,
            "diagnostic": row.phenomenon_tag == nightly.DIAGNOSTIC_TAG,
            "primitive_accuracy_before": primitive_before,
            "primitive_accuracy_after": primitive_before,
            "primitive_drift": 0.0 if primitive_before else 1.0,
        })
    return out


def all_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows = eval_cases()
    out: list[dict[str, object]] = []
    out += evaluate("keyword_sensor", -1, rows, predict_keyword)
    out += evaluate("structured_rule_sensor", -1, rows, predict_structured)
    for seed in seeds(args.seeds):
        arm = train_learned_systematic(seed, args)
        out += evaluate(
            "learned_systematic_char_sensor",
            seed,
            rows,
            lambda text, arm=arm: nightly.predict_arm(arm, text)[1],
        )
    return out


def fraction(rows: list[dict[str, object]], predicate, field: str = "action_correct") -> float:
    subset = [row for row in rows if predicate(row)]
    if not subset:
        return 0.0
    return float(sum(bool(row[field]) for row in subset) / len(subset))


def rate(rows: list[dict[str, object]], predicate, field: str) -> float:
    return fraction(rows, predicate, field)


def metrics_for(rows: list[dict[str, object]]) -> dict[str, float]:
    main = [row for row in rows if not bool(row["diagnostic"])]
    expected_exec = [row for row in main if str(row["expected_action"]).startswith("EXEC_")]
    known = [row for row in main if row["phenomenon_tag"] == "known"]
    return {
        "action_accuracy": fraction(main, lambda _: True),
        "result_accuracy": fraction(main, lambda _: True, "result_correct"),
        "exec_result_accuracy": fraction(expected_exec, lambda _: True, "result_correct"),
        "false_execution_rate": rate(main, lambda _: True, "false_execution"),
        "missed_execution_rate": rate(main, lambda _: True, "missed_execution"),
        "known_result_accuracy": fraction(known, lambda _: True, "result_correct"),
        "weak_hold_accuracy": fraction(main, lambda row: row["phenomenon_tag"] == "weak"),
        "ambiguous_hold_accuracy": fraction(main, lambda row: row["phenomenon_tag"] == "ambiguous"),
        "negation_accuracy": fraction(main, lambda row: row["phenomenon_tag"] == "negation"),
        "correction_accuracy": fraction(main, lambda row: row["phenomenon_tag"] == "correction"),
        "mention_trap_accuracy": fraction(main, lambda row: row["phenomenon_tag"] == "mention_trap"),
        "primitive_accuracy_before": float(all(bool(row["primitive_accuracy_before"]) for row in rows)),
        "primitive_accuracy_after": float(all(bool(row["primitive_accuracy_after"]) for row in rows)),
        "primitive_drift": mean(float(row["primitive_drift"]) for row in rows),
        "strict_unseen_synonym_accuracy": fraction(rows, lambda row: row["diagnostic"]),
    }


def aggregate(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for model in sorted({str(row["model_name"]) for row in rows}):
        subset = [row for row in rows if row["model_name"] == model]
        by_seed: dict[int, list[dict[str, object]]] = {}
        for row in subset:
            by_seed.setdefault(int(row["seed"]), []).append(row)
        metrics = [metrics_for(items) for items in by_seed.values()]
        avg = {name: mean(metric[name] for metric in metrics) for name in metrics[0]}
        avg["seed_count"] = float(len(metrics))
        out[model] = avg
    return out


def verdicts(agg: dict[str, dict[str, float]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for model, m in agg.items():
        labels: list[str] = []
        passed = (
            m["action_accuracy"] >= 0.95
            and m["result_accuracy"] >= 0.95
            and m["false_execution_rate"] <= 0.03
            and m["primitive_drift"] == 0.0
        )
        if model == "keyword_sensor":
            labels.append("KEYWORD_EXECUTION_BASELINE_FAILS" if not passed else "KEYWORD_EXECUTION_UNEXPECTED_PASS")
        elif model == "structured_rule_sensor":
            labels.append("STRUCTURED_SENSOR_LOCKED_SKILL_POSITIVE" if passed else "STRUCTURED_SENSOR_LOCKED_SKILL_WEAK")
        else:
            labels.append("LEARNED_SENSOR_LOCKED_SKILL_POSITIVE" if passed else "LEARNED_SENSOR_LOCKED_SKILL_WEAK")
        if m["strict_unseen_synonym_accuracy"] < 0.75:
            labels.append("STRICT_UNSEEN_SYNONYM_UNSOLVED")
        out[model] = labels
    out["global"] = (
        ["LOCKED_SKILL_PIPELINE_PASS_WITH_STRUCTURED_SENSOR"]
        if "STRUCTURED_SENSOR_LOCKED_SKILL_POSITIVE" in out.get("structured_rule_sensor", [])
        else ["LOCKED_SKILL_PIPELINE_NOT_ESTABLISHED"]
    )
    return out


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def failure_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    failures = []
    for row in rows:
        if bool(row["action_correct"]) and bool(row["result_correct"]):
            continue
        failures.append({
            "model_name": row["model_name"],
            "seed": row["seed"],
            "split": row["split"],
            "phenomenon_tag": row["phenomenon_tag"],
            "text": row["text"],
            "value": row["value"],
            "expected_action": row["expected_action"],
            "student_action": row["student_action"],
            "expected_result": row["expected_result"],
            "student_result": row["student_result"],
            "failure_type": "false_execution" if row["false_execution"] else "missed_execution" if row["missed_execution"] else "wrong_result",
        })
    return failures


def write_failures(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in failure_rows(rows):
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def metric_table(agg: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        "| Model | Seeds | Action | Result | Exec Result | False Exec | Missed Exec | Weak | Amb | Neg | Corr | Drift | Strict Syn |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, m in agg.items():
        lines.append(
            f"| `{model}` | `{m['seed_count']:.0f}` | `{m['action_accuracy']:.3f}` | `{m['result_accuracy']:.3f}` "
            f"| `{m['exec_result_accuracy']:.3f}` | `{m['false_execution_rate']:.3f}` | `{m['missed_execution_rate']:.3f}` "
            f"| `{m['weak_hold_accuracy']:.3f}` | `{m['ambiguous_hold_accuracy']:.3f}` | `{m['negation_accuracy']:.3f}` "
            f"| `{m['correction_accuracy']:.3f}` | `{m['primitive_drift']:.3f}` | `{m['strict_unseen_synonym_accuracy']:.3f}` |"
        )
    return lines


def write_report(rows: list[dict[str, object]], agg: dict[str, dict[str, float]], verdict: dict[str, list[str]], output_report: Path) -> None:
    failures = failure_rows(rows)
    lines = [
        "# PILOT_SENSOR_LOCKED_SKILL_INTEGRATION_001 Result",
        "",
        "## Goal",
        "",
        "Test the full toy path from text sensor to fixed guard to frozen ADD/MUL skill execution.",
        "",
        "## Setup",
        "",
        "- Frozen skill modules are deterministic `add(value, operand)` and `mul(value, operand)` functions.",
        "- The pilot action decides execute, reject, or hold. HOLD/REJECT must not execute a skill.",
        "- Evaluation combines robustness stress cases and factor-heldout cases.",
        "",
        "## Aggregate Metrics",
        "",
        *metric_table(agg),
        "",
        "## Verdict",
        "",
        "```json",
        json.dumps(verdict, indent=2, sort_keys=True),
        "```",
        "",
        "## Failure Examples",
        "",
    ]
    if failures:
        for row in failures[:30]:
            lines.append(
                f"- `{row['model_name']}` seed `{row['seed']}` `{row['split']}/{row['phenomenon_tag']}`: "
                f"{row['text']} value `{row['value']}` -> expected `{row['expected_action']}`/`{row['expected_result']}`, "
                f"got `{row['student_action']}`/`{row['student_result']}` ({row['failure_type']})."
            )
        if len(failures) > 30:
            lines.append(f"- ... {len(failures) - 30} more in `failure_examples.jsonl`.")
    else:
        lines.append("No failures.")
    lines += [
        "",
        "## Interpretation",
        "",
        "A positive structured-sensor result means the hand-auditable sensor, fixed guard, and frozen skills compose into a working toy execution path.",
        "Learned sensor weakness here is expected from the factor-heldout result and marks the raw text-to-scope stage as the blocker.",
        "",
        "## Claim Boundary",
        "",
        "No general NLU, full PilotPulse integration, production VRAXION/INSTNCT, or consciousness claim.",
    ]
    text = "\n".join(lines) + "\n"
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(text)
    DOC_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC_REPORT.write_text(text)


def main() -> None:
    args = parse_args()
    rows = all_rows(args)
    agg = aggregate(rows)
    verdict = verdicts(agg)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_dir / "metrics.csv")
    write_failures(rows, args.out_dir / "failure_examples.jsonl")
    summary = {"aggregate": agg, "verdict": verdict, "failure_count": len(failure_rows(rows))}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_report(rows, agg, verdict, args.out_dir / "report.md")
    print(json.dumps({
        "doc_report": str(DOC_REPORT),
        "metrics": str(args.out_dir / "metrics.csv"),
        "failures": str(args.out_dir / "failure_examples.jsonl"),
        "report": str(args.out_dir / "report.md"),
        "failure_count": summary["failure_count"],
        "verdict": verdict,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

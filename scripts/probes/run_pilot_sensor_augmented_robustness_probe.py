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

import run_pilot_sensor_scope_stack_nightly as nightly
from run_pilot_sensor_weak_ambiguity_calibration_probe import augmented_training_cases


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "output" / "pilot_sensor_augmented_robustness_001"
DOC_REPORT = ROOT / "docs" / "research" / "PILOT_SENSOR_AUGMENTED_ROBUSTNESS_001_RESULT.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PILOT_SENSOR_AUGMENTED_ROBUSTNESS_001 robustness stress for augmented learned sensor.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    return parser.parse_args()


def seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def expected_from_teacher(text: str) -> tuple[str, ...]:
    row = nightly.SensorCase("stress", "tmp", text, ("HOLD_ASK_RESEARCH",), "tmp")
    return (nightly.action_for(nightly.teacher_evidence(row)),)


def c(name: str, text: str, tag: str, split: str = "stress_eval") -> nightly.SensorCase:
    return nightly.SensorCase(split, name, text, expected_from_teacher(text), tag)


def stress_cases() -> list[nightly.SensorCase]:
    rows: list[nightly.SensorCase] = []
    nums = ["7", "8", "9", "11"]
    for n in nums:
        rows += [
            c(f"known_add_polite_{n}", f"please, add {n} now", "known"),
            c(f"known_mul_polite_{n}", f"for this task, multiply by {n}", "known"),
            c(f"unknown_div_punct_{n}", f"please: divide by {n}.", "unknown"),
            c(f"unknown_sqrt_punct_{n}", f"now take sqrt {n}", "unknown"),
            c(f"weak_prob_plus_{n}", f"probably plus {n} now", "weak"),
            c(f"weak_might_times_{n}", f"might times {n} please", "weak"),
            c(f"weak_could_add_{n}", f"it could be add {n}", "weak"),
            c(f"weak_unsure_mul_{n}", f"unsure multiply by {n}", "weak"),
            c(f"amb_either_{n}", f"either add or multiply by {n}", "ambiguous"),
            c(f"amb_div_add_mul_{n}", f"divide, add, or multiply by {n}", "ambiguous"),
            c(f"amb_maybe_plus_times_{n}", f"maybe plus, maybe times {n}", "ambiguous"),
            c(f"neg_not_plus_{n}", f"not plus {n}", "negation"),
            c(f"neg_mul_then_add_{n}", f"do not multiply by {n}, add {n} instead", "negation"),
            c(f"neg_add_then_mul_{n}", f"do not add {n}, multiply by {n} instead", "negation"),
            c(f"corr_semicolon_{n}", f"please add {n}; actually multiply by {n}", "correction"),
            c(f"corr_times_plus_{n}", f"times {n}. correction: plus {n}", "correction"),
            c(f"multi_first_then_{n}", f"first add {n}, then multiply by {n}", "multi_step_unsupported"),
        ]
    rows += [
        c("mention_note_plus", "in the note, the word plus appears", "mention_trap"),
        c("mention_said_mul", "a teammate said multiply, but no operation is requested", "mention_trap"),
        c("mention_instruction_quote", "do not follow the instruction 'multiply by 8'", "mention_trap"),
        c("substring_additive", "the additive label is visible on the page", "substring_trap"),
        c("substring_plus_page", "the plus sign appears on the page", "substring_trap"),
        c("substring_multiply_styled", "multiply-styled text, but no operation", "substring_trap"),
        nightly.SensorCase("strict_unseen_synonym_diagnostic", "strict_increment", "increment by 9", ("EXEC_ADD",), "strict_unseen_synonym"),
        nightly.SensorCase("strict_unseen_synonym_diagnostic", "strict_product", "product with 9", ("EXEC_MUL",), "strict_unseen_synonym"),
        nightly.SensorCase("strict_unseen_synonym_diagnostic", "strict_halve", "halve it", ("REJECT_UNKNOWN",), "strict_unseen_synonym"),
    ]
    return rows


def train_eval(arm: str, seed: int, args: argparse.Namespace) -> list[dict[str, object]]:
    train_rows = augmented_training_cases()
    eval_rows = stress_cases()
    rows = nightly.train_and_eval_arm(arm, seed, train_rows, eval_rows, args)
    for row in rows:
        row["stage"] = "stress"
        row["model_name"] = f"{row['model_name']}_augmented"
    return rows


def all_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows = nightly.baseline_rows(stress_cases())
    for row in rows:
        row["stage"] = "reference"
    arms = [
        "direct_evidence_char_ngram_mlp",
        "direct_evidence_word_ngram_mlp",
        "scope_stack_char_ngram_mlp",
    ]
    for seed in seeds(args.seeds):
        for arm in arms:
            rows += train_eval(arm, seed, args)
    return rows


def aggregate(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for model in sorted({str(row["model_name"]) for row in rows}):
        subset = [row for row in rows if row["model_name"] == model]
        by_seed: dict[int, list[dict[str, object]]] = {}
        for row in subset:
            by_seed.setdefault(int(row["seed"]), []).append(row)
        metrics = [stress_metrics_for(items) for items in by_seed.values()]
        avg = {name: mean(metric[name] for metric in metrics) for name in metrics[0]}
        avg["selection_score"] = selection_score(avg)
        avg["seed_count"] = float(len(metrics))
        out[model] = avg
    return out


def fraction(rows: list[dict[str, object]], predicate, field: str = "action_correct") -> float:
    subset = [row for row in rows if predicate(row)]
    if not subset:
        return 0.0
    return float(sum(bool(row[field]) for row in subset) / len(subset))


def rate(rows: list[dict[str, object]], predicate, field: str) -> float:
    return fraction(rows, predicate, field)


def tag_accuracy(rows: list[dict[str, object]], tag: str) -> float:
    return fraction(rows, lambda row: row["phenomenon_tag"] == tag)


def tag_false_commit(rows: list[dict[str, object]], tag: str) -> float:
    return rate(rows, lambda row: row["phenomenon_tag"] == tag, "false_commit")


def stress_metrics_for(rows: list[dict[str, object]]) -> dict[str, float]:
    main = [row for row in rows if not bool(row["diagnostic"])]
    known = [row for row in main if row["phenomenon_tag"] == "known"]
    weak_amb = [row for row in main if row["phenomenon_tag"] in {"weak", "ambiguous"}]
    return {
        "action_accuracy_after_guard": fraction(main, lambda _: True),
        "evidence_band_accuracy": fraction(main, lambda _: True, "evidence_band_correct"),
        "scope_flag_accuracy": fraction(main, lambda row: row["model_type"] in {"scope", "oracle"}, "scope_flag_correct"),
        "false_commit_rate": rate(main, lambda _: True, "false_commit"),
        "keyword_trap_false_commit_rate": rate(main, lambda row: row["phenomenon_tag"] in {"mention_trap", "substring_trap"}, "false_commit"),
        "missed_execute_rate": rate(main, lambda _: True, "missed_execute"),
        "over_hold_rate_on_known": float(sum(row["student_action"] == "HOLD_ASK_RESEARCH" for row in known) / len(known)) if known else 0.0,
        "weak_accuracy": tag_accuracy(main, "weak"),
        "ambiguous_accuracy": tag_accuracy(main, "ambiguous"),
        "weak_ambiguous_accuracy": fraction(weak_amb, lambda _: True),
        "negation_accuracy": tag_accuracy(main, "negation"),
        "correction_accuracy": tag_accuracy(main, "correction"),
        "mention_trap_accuracy": tag_accuracy(main, "mention_trap"),
        "substring_trap_accuracy": tag_accuracy(main, "substring_trap"),
        "multi_step_unsupported_hold_accuracy": tag_accuracy(main, "multi_step_unsupported"),
        "weak_false_commit_rate": tag_false_commit(main, "weak"),
        "ambiguous_false_commit_rate": tag_false_commit(main, "ambiguous"),
        "negation_false_commit_rate": tag_false_commit(main, "negation"),
        "teacher_student_disagreement_rate": rate(main, lambda _: True, "teacher_student_disagreement"),
        "known_execute_accuracy": float(sum(bool(row["action_correct"]) for row in known) / len(known)) if known else 0.0,
        "strict_unseen_synonym_accuracy": fraction(rows, lambda row: row["diagnostic"]),
    }


def selection_score(metrics: dict[str, float]) -> float:
    return (
        metrics["action_accuracy_after_guard"]
        - 2.0 * metrics["false_commit_rate"]
        - 2.0 * metrics["keyword_trap_false_commit_rate"]
        - max(0.0, 0.90 - metrics["weak_ambiguous_accuracy"])
        - max(0.0, 0.90 - metrics["negation_accuracy"])
        - max(0.0, 0.90 - metrics["correction_accuracy"])
        - max(0.0, 0.90 - metrics["mention_trap_accuracy"])
    )


def verdicts(agg: dict[str, dict[str, float]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for model, m in agg.items():
        labels: list[str] = []
        if model == "structured_rule_sensor_teacher":
            labels.append("TEACHER_REFERENCE_PASS" if m["action_accuracy_after_guard"] >= 0.99 else "TEACHER_REFERENCE_WEAK")
        elif model == "oracle_flags_mapper":
            labels.append("ORACLE_FLAGS_MAPPER_PASS" if m["action_accuracy_after_guard"] >= 0.99 else "ORACLE_FLAGS_MAPPER_WEAK")
        elif model == "keyword_sensor":
            labels.append("KEYWORD_BASELINE_FAILS" if m["false_commit_rate"] > 0.10 else "KEYWORD_BASELINE_UNEXPECTED")
        elif (
            m["action_accuracy_after_guard"] >= 0.95
            and m["false_commit_rate"] <= 0.03
            and m["keyword_trap_false_commit_rate"] <= 0.03
            and m["weak_ambiguous_accuracy"] >= 0.90
            and m["mention_trap_accuracy"] >= 0.90
            and m["substring_trap_accuracy"] >= 0.90
            and m["negation_accuracy"] >= 0.90
            and m["correction_accuracy"] >= 0.90
        ):
            labels.append("AUGMENTED_ROBUSTNESS_POSITIVE")
        else:
            labels.append("AUGMENTED_ROBUSTNESS_WEAK")
        if m["strict_unseen_synonym_accuracy"] < 0.75:
            labels.append("STRICT_UNSEEN_SYNONYM_UNSOLVED")
        out[model] = labels
    out["global"] = (
        ["AUGMENTED_SENSOR_ROBUST"]
        if any("AUGMENTED_ROBUSTNESS_POSITIVE" in labels for labels in out.values())
        else ["AUGMENTED_SENSOR_NOT_ROBUST"]
    )
    return out


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_failures(rows: list[dict[str, object]], path: Path) -> None:
    failures = nightly.failure_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in failures:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def metric_table(agg: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        "| Model | Seeds | Score | Action | Weak/Amb | False Commit | Trap False | Mention | Substr | Neg | Corr | Strict Syn |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, m in agg.items():
        lines.append(
            f"| `{model}` | `{m['seed_count']:.0f}` | `{m['selection_score']:.3f}` | `{m['action_accuracy_after_guard']:.3f}` "
            f"| `{m['weak_ambiguous_accuracy']:.3f}` | `{m['false_commit_rate']:.3f}` "
            f"| `{m['keyword_trap_false_commit_rate']:.3f}` | `{m['mention_trap_accuracy']:.3f}` "
            f"| `{m['substring_trap_accuracy']:.3f}` | `{m['negation_accuracy']:.3f}` | `{m['correction_accuracy']:.3f}` "
            f"| `{m['strict_unseen_synonym_accuracy']:.3f}` |"
        )
    return lines


def write_report(rows: list[dict[str, object]], agg: dict[str, dict[str, float]], verdict: dict[str, list[str]], output_report: Path) -> None:
    failures = nightly.failure_rows(rows)
    lines = [
        "# PILOT_SENSOR_AUGMENTED_ROBUSTNESS_001 Result",
        "",
        "## Goal",
        "",
        "Stress the positive weak/ambiguous augmentation result on new teacher-supported surface forms and heldout combinations.",
        "",
        "## Setup",
        "",
        "- Trains on the augmented weak/ambiguous dataset from the calibration probe.",
        "- Evaluates on newly generated stress cases with new numbers, fillers, punctuation, and combinations.",
        "- Action always comes from predicted evidence through the fixed guard.",
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
                f"{row['text']} -> expected `{row['expected_action']}`, got `{row['student_action']}` ({row['failure_type']})."
            )
        if len(failures) > 30:
            lines.append(f"- ... {len(failures) - 30} more in `failure_examples.jsonl`.")
    else:
        lines.append("No failures.")
    lines += [
        "",
        "## Interpretation",
        "",
        "Positive robustness means the augmented learned evidence sensor generalizes beyond the exact calibration eval list within this toy command grammar.",
        "",
        "Strict unseen synonym remains diagnostic only.",
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
    summary = {"aggregate": agg, "verdict": verdict, "failure_count": len(nightly.failure_rows(rows))}
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

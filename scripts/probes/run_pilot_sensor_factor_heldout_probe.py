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
import run_pilot_sensor_scope_stack_nightly as nightly


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "output" / "pilot_sensor_factor_heldout_001"
DOC_REPORT = ROOT / "docs" / "research" / "PILOT_SENSOR_FACTOR_HELDOUT_001_RESULT.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PILOT_SENSOR_FACTOR_HELDOUT_001 compositional template-factor heldout probe.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    return parser.parse_args()


def seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def expected_from_teacher(text: str) -> tuple[str, ...]:
    row = nightly.SensorCase("factor", "tmp", text, ("HOLD_ASK_RESEARCH",), "tmp")
    return (nightly.action_for(nightly.teacher_evidence(row)),)


def c(split: str, name: str, text: str, tag: str) -> nightly.SensorCase:
    return nightly.SensorCase(split, name, text, expected_from_teacher(text), tag)


def factor_train_cases() -> list[nightly.SensorCase]:
    rows = nightly.training_cases()
    nums = ["1", "2", "3", "4", "5", "6"]
    for n in nums:
        # Every primitive appears during training, but selected marker/op/scope combinations are held out.
        rows += [
            c("train_factor", f"known_plus_{n}", f"plus {n}", "known"),
            c("train_factor", f"known_times_{n}", f"times {n}", "known"),
            c("train_factor", f"weak_might_add_{n}", f"might add {n}", "weak"),
            c("train_factor", f"weak_might_plus_{n}", f"might plus {n}", "weak"),
            c("train_factor", f"weak_maybe_times_{n}", f"maybe times {n}", "weak"),
            c("train_factor", f"weak_prob_multiply_{n}", f"probably multiply by {n}", "weak"),
            c("train_factor", f"weak_unsure_add_{n}", f"unsure add {n}", "weak"),
            c("train_factor", f"weak_could_multiply_{n}", f"could multiply {n}", "weak"),
            c("train_factor", f"amb_plus_times_{n}", f"maybe plus, maybe times {n}", "ambiguous"),
            c("train_factor", f"amb_either_add_multiply_{n}", f"either add or multiply by {n}", "ambiguous"),
            c("train_factor", f"neg_not_add_{n}", f"not add {n}", "negation"),
            c("train_factor", f"neg_do_not_multiply_{n}", f"do not multiply by {n}", "negation"),
            c("train_factor", f"neg_add_then_mul_{n}", f"do not add {n}, multiply by {n} instead", "negation"),
            c("train_factor", f"neg_mul_then_add_{n}", f"do not multiply by {n}, add {n} instead", "negation"),
            c("train_factor", f"corr_add_mul_{n}", f"add {n}. correction: multiply by {n}", "correction"),
            c("train_factor", f"corr_mul_add_{n}", f"multiply by {n}. correction: add {n}", "correction"),
            c("train_factor", f"corr_plus_times_{n}", f"plus {n}. correction: times {n}", "correction"),
            c("train_factor", f"mention_quote_add_{n}", f"do not follow the instruction 'add {n}'", "mention_trap"),
            c("train_factor", f"mention_word_multiply_{n}", f"the word multiply appears in the note", "mention_trap"),
            c("train_factor", f"multi_add_mul_{n}", f"first add {n}, then multiply by {n}", "multi_step_unsupported"),
        ]
    rows += [
        c("train_factor", "mention_plus_note", "in the note, the word plus appears", "mention_trap"),
        c("train_factor", "mention_said_add", "someone said add, but no operation is requested", "mention_trap"),
        c("train_factor", "substring_plus_page", "the plus sign appears on the page", "substring_trap"),
        c("train_factor", "substring_multiply_styled", "multiply-styled text, but no operation", "substring_trap"),
    ]
    return rows


def factor_eval_cases() -> list[nightly.SensorCase]:
    rows: list[nightly.SensorCase] = []
    for n in ["7", "8", "9", "11"]:
        rows += [
            c("heldout_factor", f"known_polite_add_{n}", f"please, add {n} now", "known"),
            c("heldout_factor", f"known_polite_mul_{n}", f"for this task, multiply by {n}", "known"),
            c("heldout_factor", f"weak_might_times_{n}", f"might times {n} please", "weak"),
            c("heldout_factor", f"weak_prob_plus_{n}", f"probably plus {n} now", "weak"),
            c("heldout_factor", f"weak_unsure_mul_{n}", f"unsure multiply by {n}", "weak"),
            c("heldout_factor", f"amb_div_add_mul_{n}", f"divide, add, or multiply by {n}", "ambiguous"),
            c("heldout_factor", f"neg_not_plus_{n}", f"not plus {n}", "negation"),
            c("heldout_factor", f"neg_mul_then_add_{n}", f"do not multiply by {n}, add {n} instead", "negation"),
            c("heldout_factor", f"corr_times_plus_{n}", f"times {n}. correction: plus {n}", "correction"),
            c("heldout_factor", f"corr_semicolon_{n}", f"please add {n}; actually multiply by {n}", "correction"),
            c("heldout_factor", f"multi_plus_times_{n}", f"first plus {n}, then times {n}", "multi_step_unsupported"),
        ]
    rows += [
        c("heldout_factor", "mention_quote_multiply", "do not follow the instruction 'multiply by 8'", "mention_trap"),
        c("heldout_factor", "mention_said_multiply", "a teammate said multiply, but no operation is requested", "mention_trap"),
        c("heldout_factor", "substring_additive_page", "the additive label is visible on the page", "substring_trap"),
        nightly.SensorCase("strict_unseen_synonym_diagnostic", "strict_increment", "increment by 9", ("EXEC_ADD",), "strict_unseen_synonym"),
        nightly.SensorCase("strict_unseen_synonym_diagnostic", "strict_product", "product with 9", ("EXEC_MUL",), "strict_unseen_synonym"),
    ]
    return rows


def train_eval(arm: str, seed: int, args: argparse.Namespace) -> list[dict[str, object]]:
    rows = nightly.train_and_eval_arm(arm, seed, factor_train_cases(), factor_eval_cases(), args)
    for row in rows:
        row["stage"] = "factor_heldout"
        row["model_name"] = f"{row['model_name']}_factor"
    return rows


def all_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows = nightly.baseline_rows(factor_eval_cases())
    for row in rows:
        row["stage"] = "reference"
    for seed in seeds(args.seeds):
        for arm in ["direct_evidence_char_ngram_mlp", "direct_evidence_word_ngram_mlp", "scope_stack_char_ngram_mlp"]:
            rows += train_eval(arm, seed, args)
    return rows


def aggregate(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for model in sorted({str(row["model_name"]) for row in rows}):
        subset = [row for row in rows if row["model_name"] == model]
        by_seed: dict[int, list[dict[str, object]]] = {}
        for row in subset:
            by_seed.setdefault(int(row["seed"]), []).append(row)
        metrics = [robustness.stress_metrics_for(items) for items in by_seed.values()]
        avg = {name: mean(metric[name] for metric in metrics) for name in metrics[0]}
        avg["selection_score"] = robustness.selection_score(avg)
        avg["seed_count"] = float(len(metrics))
        out[model] = avg
    return out


def verdicts(agg: dict[str, dict[str, float]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    passes = []
    for model, m in agg.items():
        labels: list[str] = []
        if model == "structured_rule_sensor_teacher":
            labels.append("TEACHER_REFERENCE_PASS" if m["action_accuracy_after_guard"] >= 0.99 else "TEACHER_REFERENCE_WEAK")
        elif model == "oracle_flags_mapper":
            labels.append("ORACLE_FLAGS_MAPPER_PASS" if m["action_accuracy_after_guard"] >= 0.99 else "ORACLE_FLAGS_MAPPER_WEAK")
        elif model == "keyword_sensor":
            labels.append("KEYWORD_BASELINE_FAILS" if m["false_commit_rate"] > 0.10 else "KEYWORD_BASELINE_UNEXPECTED")
        else:
            passed = (
                m["action_accuracy_after_guard"] >= 0.95
                and m["false_commit_rate"] <= 0.03
                and m["keyword_trap_false_commit_rate"] <= 0.03
                and m["weak_ambiguous_accuracy"] >= 0.90
                and m["mention_trap_accuracy"] >= 0.90
                and m["negation_accuracy"] >= 0.90
                and m["correction_accuracy"] >= 0.90
            )
            if passed:
                labels.append("FACTOR_HELDOUT_POSITIVE")
                passes.append(model)
            elif m["action_accuracy_after_guard"] >= 0.85 and m["false_commit_rate"] <= 0.10:
                labels.append("FACTOR_HELDOUT_PARTIAL")
            else:
                labels.append("FACTOR_HELDOUT_WEAK")
        if m["strict_unseen_synonym_accuracy"] < 0.75:
            labels.append("STRICT_UNSEEN_SYNONYM_UNSOLVED")
        out[model] = labels
    out["global"] = ["FACTOR_HELDOUT_SOLVED", *passes] if passes else ["FACTOR_HELDOUT_REMAINS_BOTTLENECK"]
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
        "| Model | Seeds | Score | Action | Weak/Amb | False Commit | Trap False | Mention | Neg | Corr | Strict Syn |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, m in agg.items():
        lines.append(
            f"| `{model}` | `{m['seed_count']:.0f}` | `{m['selection_score']:.3f}` | `{m['action_accuracy_after_guard']:.3f}` "
            f"| `{m['weak_ambiguous_accuracy']:.3f}` | `{m['false_commit_rate']:.3f}` "
            f"| `{m['keyword_trap_false_commit_rate']:.3f}` | `{m['mention_trap_accuracy']:.3f}` "
            f"| `{m['negation_accuracy']:.3f}` | `{m['correction_accuracy']:.3f}` "
            f"| `{m['strict_unseen_synonym_accuracy']:.3f}` |"
        )
    return lines


def write_report(rows: list[dict[str, object]], agg: dict[str, dict[str, float]], verdict: dict[str, list[str]], output_report: Path) -> None:
    failures = nightly.failure_rows(rows)
    lines = [
        "# PILOT_SENSOR_FACTOR_HELDOUT_001 Result",
        "",
        "## Goal",
        "",
        "Test whether learned sensors compose seen weak, scope, correction, and operation factors into heldout combinations.",
        "",
        "## Setup",
        "",
        "- Training includes every relevant atom, but with selected factor combinations held out.",
        "- Evaluation uses heldout combinations such as `might times`, `not plus`, `times. correction: plus`, and quoted multiply instructions.",
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
        "A positive result means systematic template coverage can generalize across some heldout factor combinations within this toy grammar.",
        "A negative result means the previous systematic pass was closer to template coverage than compositional scope handling.",
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

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


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "output" / "pilot_sensor_weak_ambiguity_calibration_001"
DOC_REPORT = ROOT / "docs" / "research" / "PILOT_SENSOR_WEAK_AMBIGUITY_CALIBRATION_001_RESULT.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PILOT_SENSOR_WEAK_AMBIGUITY_CALIBRATION_001 targeted false-commit calibration probe.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    return parser.parse_args()


def seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def c(split: str, name: str, text: str, expected: str | tuple[str, ...], tag: str) -> nightly.SensorCase:
    return nightly.case(split, name, text, expected, tag)


def augmented_training_cases() -> list[nightly.SensorCase]:
    rows = nightly.training_cases()
    nums = ["1", "2", "4", "5"]
    weak_templates = [
        ("probably add {n}", "weak"),
        ("probably multiply by {n}", "weak"),
        ("it could be add", "weak"),
        ("it could be multiply", "weak"),
        ("might add {n}", "weak"),
        ("might multiply by {n}", "weak"),
        ("unsure add {n}", "weak"),
        ("unsure multiply {n}", "weak"),
        ("maybe plus {n}", "weak"),
        ("maybe times {n}", "weak"),
        ("perhaps plus {n}", "weak"),
        ("perhaps times {n}", "weak"),
    ]
    ambiguous_templates = [
        ("add, multiply, or divide by {n}", "ambiguous"),
        ("maybe plus, maybe times {n}", "ambiguous"),
        ("add versus multiply by {n}", "ambiguous"),
        ("either add or multiply by {n}", "ambiguous"),
        ("choose add or multiply by {n}", "ambiguous"),
        ("not sure whether to plus or times {n}", "ambiguous"),
    ]
    for n in nums:
        for idx, (template, tag) in enumerate(weak_templates):
            rows.append(c("train_augmented", f"aug_weak_{n}_{idx}", template.format(n=n), "HOLD_ASK_RESEARCH", tag))
        for idx, (template, tag) in enumerate(ambiguous_templates):
            rows.append(c("train_augmented", f"aug_amb_{n}_{idx}", template.format(n=n), "HOLD_ASK_RESEARCH", tag))
    return rows


def eval_rows() -> list[nightly.SensorCase]:
    return nightly.eval_cases()


def train_eval(arm_name: str, train_rows: list[nightly.SensorCase], seed: int, args: argparse.Namespace) -> list[dict[str, object]]:
    rows = nightly.train_and_eval_arm(arm_name, seed, train_rows, eval_rows(), args)
    for row in rows:
        row["stage"] = "main"
    return rows


def all_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows = nightly.baseline_rows(eval_rows())
    for row in rows:
        row["stage"] = "reference"
    original_train = nightly.training_cases()
    augmented_train = augmented_training_cases()
    arms = [
        ("direct_evidence_char_ngram_mlp", original_train, "original"),
        ("scope_stack_char_ngram_mlp", original_train, "original"),
        ("direct_evidence_char_ngram_mlp", augmented_train, "augmented"),
        ("scope_stack_char_ngram_mlp", augmented_train, "augmented"),
        ("direct_evidence_word_ngram_mlp", augmented_train, "augmented"),
        ("scope_stack_word_ngram_mlp", augmented_train, "augmented"),
    ]
    for seed in seeds(args.seeds):
        for arm, train_rows, train_variant in arms:
            arm_rows = train_eval(arm, train_rows, seed, args)
            for row in arm_rows:
                row["train_variant"] = train_variant
                row["model_name"] = f"{row['model_name']}_{train_variant}"
            rows += arm_rows
    for row in rows:
        row.setdefault("train_variant", "reference")
    return rows


def aggregate(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    keys = sorted({str(row["model_name"]) for row in rows})
    for key in keys:
        subset = [row for row in rows if row["model_name"] == key]
        by_seed: dict[int, list[dict[str, object]]] = {}
        for row in subset:
            by_seed.setdefault(int(row["seed"]), []).append(row)
        metrics = [nightly.metrics_for(items) for items in by_seed.values()]
        avg = {name: mean(metric[name] for metric in metrics) for name in metrics[0]}
        avg["selection_score"] = nightly.selection_score(avg)
        avg["seed_count"] = float(len(metrics))
        out[key] = avg
    return out


def verdicts(agg: dict[str, dict[str, float]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    best_aug = max((m["selection_score"] for k, m in agg.items() if k.endswith("_augmented")), default=-999.0)
    best_orig = max((m["selection_score"] for k, m in agg.items() if k.endswith("_original")), default=-999.0)
    for key, m in agg.items():
        labels: list[str] = []
        if key == "structured_rule_sensor_teacher":
            labels.append("TEACHER_REFERENCE_PASS" if m["action_accuracy_after_guard"] >= 0.99 else "TEACHER_REFERENCE_WEAK")
        elif key == "oracle_flags_mapper":
            labels.append("ORACLE_FLAGS_MAPPER_PASS" if m["action_accuracy_after_guard"] >= 0.99 else "ORACLE_FLAGS_MAPPER_WEAK")
        elif key == "keyword_sensor":
            labels.append("KEYWORD_BASELINE_FAILS" if m["false_commit_rate"] > 0.10 else "KEYWORD_BASELINE_UNEXPECTED")
        else:
            if (
                m["action_accuracy_after_guard"] >= 0.95
                and m["heldout_weak_ambiguous_accuracy"] >= 0.85
                and m["false_commit_rate"] <= 0.03
                and m["heldout_scope_accuracy"] >= 0.95
                and m["heldout_negation_accuracy"] >= 0.95
                and m["heldout_correction_accuracy"] >= 0.90
            ):
                labels.append("WEAK_AMBIGUITY_CALIBRATION_POSITIVE")
            elif m["heldout_weak_ambiguous_accuracy"] >= 0.75 and m["false_commit_rate"] <= 0.05:
                labels.append("WEAK_AMBIGUITY_CALIBRATION_PARTIAL")
            else:
                labels.append("WEAK_AMBIGUITY_STILL_BOTTLENECK")
            if key.endswith("_augmented") and best_aug > best_orig + 0.05:
                labels.append("AUGMENTATION_HELPS")
            if m["strict_unseen_synonym_accuracy"] < 0.75:
                labels.append("STRICT_UNSEEN_SYNONYM_UNSOLVED")
        out[key] = labels
    out["global"] = (
        ["AUGMENTATION_HELPS_WEAK_AMBIGUITY"]
        if best_aug > best_orig + 0.05
        else ["AUGMENTATION_DOES_NOT_SOLVE_WEAK_AMBIGUITY"]
    )
    return out


def failure_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return nightly.failure_rows(rows)


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
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def metric_table(agg: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        "| Model | Seeds | Score | Action | Weak/Amb | False Commit | Scope | Neg | Corr | Evidence Band | Strict Syn |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, m in agg.items():
        lines.append(
            f"| `{key}` | `{m['seed_count']:.0f}` | `{m['selection_score']:.3f}` | `{m['action_accuracy_after_guard']:.3f}` "
            f"| `{m['heldout_weak_ambiguous_accuracy']:.3f}` | `{m['false_commit_rate']:.3f}` "
            f"| `{m['heldout_scope_accuracy']:.3f}` | `{m['heldout_negation_accuracy']:.3f}` "
            f"| `{m['heldout_correction_accuracy']:.3f}` | `{m['evidence_band_accuracy']:.3f}` "
            f"| `{m['strict_unseen_synonym_accuracy']:.3f}` |"
        )
    return lines


def failure_lines(failures: list[dict[str, object]], limit: int = 30) -> list[str]:
    if not failures:
        return ["No failures."]
    lines = []
    for row in failures[:limit]:
        lines.append(
            f"- `{row['model_name']}` seed `{row['seed']}` `{row['split']}/{row['phenomenon_tag']}`: "
            f"{row['text']} -> expected `{row['expected_action']}`, got `{row['student_action']}` ({row['failure_type']})."
        )
    if len(failures) > limit:
        lines.append(f"- ... {len(failures) - limit} more in `failure_examples.jsonl`.")
    return lines


def write_report(rows: list[dict[str, object]], agg: dict[str, dict[str, float]], verdict: dict[str, list[str]], failures: list[dict[str, object]], output_report: Path) -> None:
    lines = [
        "# PILOT_SENSOR_WEAK_AMBIGUITY_CALIBRATION_001 Result",
        "",
        "## Goal",
        "",
        "Test whether targeted weak/ambiguous training coverage reduces learned sensor false commits without changing the fixed guard.",
        "",
        "## Setup",
        "",
        "- Reuses `PILOT_SENSOR_SCOPE_STACK_NIGHTLY_001` direct and scope-stack MLP arms.",
        "- Compares original training cases against augmented weak/ambiguous templates.",
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
        *failure_lines(failures),
        "",
        "## Interpretation",
        "",
        "If augmentation passes, the previous bottleneck was mainly train coverage for weak/ambiguous forms. If it fails, the sensor needs stronger abstention/calibration, not just more templates.",
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
    failures = failure_rows(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_dir / "metrics.csv")
    write_failures(failures, args.out_dir / "failure_examples.jsonl")
    summary = {"aggregate": agg, "verdict": verdict, "failure_count": len(failures)}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_report(rows, agg, verdict, failures, args.out_dir / "report.md")
    print(json.dumps({
        "doc_report": str(DOC_REPORT),
        "metrics": str(args.out_dir / "metrics.csv"),
        "failures": str(args.out_dir / "failure_examples.jsonl"),
        "report": str(args.out_dir / "report.md"),
        "failure_count": len(failures),
        "verdict": verdict,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

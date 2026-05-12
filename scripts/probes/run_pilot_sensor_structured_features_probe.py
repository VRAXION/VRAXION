#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_pilot_sensor_augmented_robustness_probe as robustness
import run_pilot_sensor_factor_heldout_probe as factor_probe
import run_pilot_sensor_scope_stack_nightly as nightly
from run_pilot_sensor_distill_probe import evidence_to_bands


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "output" / "pilot_sensor_structured_features_001"
DOC_REPORT = ROOT / "docs" / "research" / "PILOT_SENSOR_STRUCTURED_FEATURES_001_RESULT.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PILOT_SENSOR_STRUCTURED_FEATURES_001 explicit scope/event feature probe.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    return parser.parse_args()


def seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def feature_vector(text: str) -> np.ndarray:
    flags = nightly.teacher_flags(text)
    return np.array([flags[name] for name in nightly.FLAGS], dtype=np.float32)


def feature_matrix(rows: list[nightly.SensorCase]) -> torch.Tensor:
    return torch.tensor(np.stack([feature_vector(row.text) for row in rows]), dtype=torch.float32)


class LinearEvidence(torch.nn.Module):
    def __init__(self, inputs: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(inputs, 9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).view(-1, 3, 3)


def train_feature_student(seed: int, args: argparse.Namespace) -> LinearEvidence:
    torch.manual_seed(seed)
    train_rows = factor_probe.factor_train_cases()
    x = feature_matrix(train_rows)
    y = torch.tensor([evidence_to_bands(nightly.teacher_evidence(row)) for row in train_rows], dtype=torch.long)
    model = LinearEvidence(x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    for _ in range(args.epochs):
        opt.zero_grad()
        logits = model(x)
        loss = sum(F.cross_entropy(logits[:, channel, :], y[:, channel]) for channel in range(3))
        loss.backward()
        opt.step()
    return model


def predict_feature_student(model: LinearEvidence, text: str) -> tuple[dict[str, int], np.ndarray]:
    flags = nightly.teacher_flags(text)
    with torch.no_grad():
        x = torch.tensor(feature_vector(text)[None, :], dtype=torch.float32)
        bands = model(x).argmax(dim=-1).squeeze(0).cpu().tolist()
    evidence = np.array([nightly.BAND_VALUES[int(band)] for band in bands], dtype=float)
    return flags, evidence


def learned_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    eval_rows = factor_probe.factor_eval_cases()
    for seed in seeds(args.seeds):
        model = train_feature_student(seed, args)
        out = nightly.evaluate_model(
            "structured_feature_linear_student",
            "scope",
            seed,
            eval_rows,
            lambda row, model=model: predict_feature_student(model, row.text),
        )
        for row in out:
            row["stage"] = "factor_heldout"
        rows += out
    return rows


def all_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows = nightly.baseline_rows(factor_probe.factor_eval_cases())
    for row in rows:
        row["stage"] = "reference"
    rows += learned_rows(args)
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
            and m["weak_ambiguous_accuracy"] >= 0.90
            and m["negation_accuracy"] >= 0.90
            and m["correction_accuracy"] >= 0.90
            and m["mention_trap_accuracy"] >= 0.90
        ):
            labels.append("STRUCTURED_FEATURES_POSITIVE")
        else:
            labels.append("STRUCTURED_FEATURES_WEAK")
        if m["strict_unseen_synonym_accuracy"] < 0.75:
            labels.append("STRICT_UNSEEN_SYNONYM_UNSOLVED")
        out[model] = labels
    out["global"] = (
        ["EXPLICIT_SCOPE_FEATURES_SOLVE_FACTOR_HELDOUT"]
        if "STRUCTURED_FEATURES_POSITIVE" in out.get("structured_feature_linear_student", [])
        else ["EXPLICIT_SCOPE_FEATURES_NOT_SUFFICIENT"]
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
        "| Model | Seeds | Score | Action | Weak/Amb | False Commit | Mention | Neg | Corr | Scope Flags | Strict Syn |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, m in agg.items():
        lines.append(
            f"| `{model}` | `{m['seed_count']:.0f}` | `{m['selection_score']:.3f}` | `{m['action_accuracy_after_guard']:.3f}` "
            f"| `{m['weak_ambiguous_accuracy']:.3f}` | `{m['false_commit_rate']:.3f}` "
            f"| `{m['mention_trap_accuracy']:.3f}` | `{m['negation_accuracy']:.3f}` | `{m['correction_accuracy']:.3f}` "
            f"| `{m['scope_flag_accuracy']:.3f}` | `{m['strict_unseen_synonym_accuracy']:.3f}` |"
        )
    return lines


def write_report(rows: list[dict[str, object]], agg: dict[str, dict[str, float]], verdict: dict[str, list[str]], output_report: Path) -> None:
    failures = nightly.failure_rows(rows)
    lines = [
        "# PILOT_SENSOR_STRUCTURED_FEATURES_001 Result",
        "",
        "## Goal",
        "",
        "Test whether explicit normalized scope/event features solve the factor-heldout failures that raw n-gram learned sensors missed.",
        "",
        "## Setup",
        "",
        "- Uses the same factor-heldout train/eval split as `PILOT_SENSOR_FACTOR_HELDOUT_001`.",
        "- Replaces raw text n-grams with explicit scope/event flags: cue flags, weak/ambiguity markers, mention-only, negation, correction targets, and multi-step unsupported.",
        "- A linear student maps these features to evidence bands; action still comes from the fixed guard.",
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
        "A positive result means the factor-heldout bottleneck is the raw text-to-scope feature extractor, not the guard or evidence interface.",
        "It does not mean the parser is learned from raw text; this is a parser-assisted feature interface.",
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

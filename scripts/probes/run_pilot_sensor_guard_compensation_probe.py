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

import run_pilot_sensor_factor_heldout_probe as factor_probe
import run_pilot_sensor_probe as sensor_probe
import run_pilot_sensor_scope_stack_nightly as nightly


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "output" / "pilot_sensor_guard_compensation_001"
DOC_REPORT = ROOT / "docs" / "research" / "PILOT_SENSOR_GUARD_COMPENSATION_001_RESULT.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PILOT_SENSOR_GUARD_COMPENSATION_001 threshold sweep over learned sensor evidence.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    return parser.parse_args()


def seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def guarded_action(evidence: list[float], strength_threshold: float, margin_threshold: float) -> str:
    ordered = sorted(enumerate(evidence), key=lambda item: item[1], reverse=True)
    top_idx, top = ordered[0]
    _, second = ordered[1]
    if top < strength_threshold:
        return "HOLD_ASK_RESEARCH"
    if top - second < margin_threshold:
        return "HOLD_ASK_RESEARCH"
    if top_idx == 2:
        return "REJECT_UNKNOWN"
    return "EXEC_ADD" if top_idx == 0 else "EXEC_MUL"


def learned_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    eval_rows = factor_probe.factor_eval_cases()
    train_rows = factor_probe.factor_train_cases()
    for seed in seeds(args.seeds):
        for arm_name in ["direct_evidence_char_ngram_mlp", "scope_stack_char_ngram_mlp"]:
            arm_rows = nightly.train_and_eval_arm(arm_name, seed, train_rows, eval_rows, args)
            for row in arm_rows:
                row["model_name"] = f"{row['model_name']}_factor"
            rows += arm_rows
    return rows


def sweep_rows(base_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for base in base_rows:
        evidence = [float(x) for x in json.loads(str(base["student_evidence"]))]
        for strength in [0.75, 0.85, 0.95]:
            for margin in [0.30, 0.45, 0.60]:
                action = guarded_action(evidence, strength, margin)
                expected = tuple(str(base["expected_action"]).split("|"))
                rows.append({
                    "model_name": base["model_name"],
                    "seed": base["seed"],
                    "split": base["split"],
                    "case": base["case"],
                    "text": base["text"],
                    "phenomenon_tag": base["phenomenon_tag"],
                    "expected_action": base["expected_action"],
                    "student_evidence": base["student_evidence"],
                    "strength_threshold": strength,
                    "margin_threshold": margin,
                    "student_action": action,
                    "action_correct": action in expected,
                    "false_commit": sensor_probe.is_false_commit(action, expected),
                    "missed_execute": sensor_probe.is_missed_execute(action, expected),
                    "diagnostic": base["diagnostic"],
                })
    return rows


def fraction(rows: list[dict[str, object]], predicate, field: str = "action_correct") -> float:
    subset = [row for row in rows if predicate(row)]
    if not subset:
        return 0.0
    return float(sum(bool(row[field]) for row in subset) / len(subset))


def metrics_for(rows: list[dict[str, object]]) -> dict[str, float]:
    main = [row for row in rows if not bool(row["diagnostic"])]
    return {
        "action_accuracy": fraction(main, lambda _: True),
        "false_commit_rate": fraction(main, lambda _: True, "false_commit"),
        "missed_execute_rate": fraction(main, lambda _: True, "missed_execute"),
        "known_accuracy": fraction(main, lambda row: row["phenomenon_tag"] == "known"),
        "weak_ambiguous_accuracy": fraction(main, lambda row: row["phenomenon_tag"] in {"weak", "ambiguous"}),
        "negation_accuracy": fraction(main, lambda row: row["phenomenon_tag"] == "negation"),
        "correction_accuracy": fraction(main, lambda row: row["phenomenon_tag"] == "correction"),
    }


def aggregate(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    keys = sorted({(str(row["model_name"]), float(row["strength_threshold"]), float(row["margin_threshold"])) for row in rows})
    for model, strength, margin in keys:
        subset = [row for row in rows if row["model_name"] == model and float(row["strength_threshold"]) == strength and float(row["margin_threshold"]) == margin]
        by_seed: dict[int, list[dict[str, object]]] = {}
        for row in subset:
            by_seed.setdefault(int(row["seed"]), []).append(row)
        vals = [metrics_for(seed_rows) for seed_rows in by_seed.values()]
        avg = {name: mean(item[name] for item in vals) for name in vals[0]}
        avg["seed_count"] = float(len(vals))
        avg["strength_threshold"] = strength
        avg["margin_threshold"] = margin
        avg["score"] = avg["action_accuracy"] - 2.0 * avg["false_commit_rate"] - avg["missed_execute_rate"]
        out[f"{model}|s={strength:.2f}|m={margin:.2f}"] = avg
    return out


def verdicts(agg: dict[str, dict[str, float]]) -> dict[str, object]:
    viable = {
        key: val for key, val in agg.items()
        if val["action_accuracy"] >= 0.90 and val["false_commit_rate"] <= 0.05 and val["known_accuracy"] >= 0.95
    }
    best = max(agg.items(), key=lambda item: item[1]["score"])
    return {
        "global": ["GUARD_CALIBRATION_CAN_COMPENSATE"] if viable else ["GUARD_CALIBRATION_CANNOT_FIX_SENSOR_SCOPE_ERRORS"],
        "best_by_score": best[0],
        "best_metrics": best[1],
        "viable_count": len(viable),
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_table(agg: dict[str, dict[str, float]]) -> list[str]:
    top = sorted(agg.items(), key=lambda item: item[1]["score"], reverse=True)[:12]
    lines = [
        "| Setting | Score | Action | False Commit | Missed Execute | Known | Weak/Amb | Neg | Corr |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, m in top:
        lines.append(
            f"| `{key}` | `{m['score']:.3f}` | `{m['action_accuracy']:.3f}` | `{m['false_commit_rate']:.3f}` "
            f"| `{m['missed_execute_rate']:.3f}` | `{m['known_accuracy']:.3f}` | `{m['weak_ambiguous_accuracy']:.3f}` "
            f"| `{m['negation_accuracy']:.3f}` | `{m['correction_accuracy']:.3f}` |"
        )
    return lines


def write_report(agg: dict[str, dict[str, float]], verdict: dict[str, object], output_report: Path) -> None:
    lines = [
        "# PILOT_SENSOR_GUARD_COMPENSATION_001 Result",
        "",
        "## Goal",
        "",
        "Test whether stricter strength/margin guard thresholds can compensate for factor-heldout learned sensor errors.",
        "",
        "## Top Threshold Settings",
        "",
        *metric_table(agg),
        "",
        "## Verdict",
        "",
        "```json",
        json.dumps(verdict, indent=2, sort_keys=True),
        "```",
        "",
        "## Interpretation",
        "",
        "If no threshold setting reaches high action accuracy with low false commits, the blocker is sensor evidence quality rather than guard calibration.",
        "False commits from strong but wrong evidence, such as negation/correction scope errors, cannot be reliably fixed downstream by thresholds.",
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
    base = learned_rows(args)
    rows = sweep_rows(base)
    agg = aggregate(rows)
    verdict = verdicts(agg)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_dir / "metrics.csv")
    (args.out_dir / "summary.json").write_text(json.dumps({"aggregate": agg, "verdict": verdict}, indent=2, sort_keys=True) + "\n")
    write_report(agg, verdict, args.out_dir / "report.md")
    print(json.dumps({
        "doc_report": str(DOC_REPORT),
        "metrics": str(args.out_dir / "metrics.csv"),
        "report": str(args.out_dir / "report.md"),
        "verdict": verdict,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

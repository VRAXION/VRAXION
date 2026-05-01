#!/usr/bin/env python3
"""
D15B projection-selectivity gate for D10p/D15 high-H semantic scout outputs.

This is a post-hoc hardening pass: arm-level safe rates can hide whether the
same candidate is genuinely selective or whether controls win on the same
proposal. The gate groups rows by (arm, candidate_idx) and compares each real
candidate against its own adversarial controls.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


CONTROL_TYPES = ["random_label", "random_bigram", "unigram_decoy", "projection_shuffle"]


@dataclass
class GateThresholds:
    weak_margin: float
    strong_margin: float


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return 0.0
    return float(value)


def as_bool(row: dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).lower() == "true"


def load_rows(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "semantic_candidates.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing semantic_candidates.csv in {run_dir}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def classify_candidate(real: dict[str, str], controls: list[dict[str, str]], thresholds: GateThresholds) -> tuple[str, dict[str, float | int | str | bool]]:
    real_mo = as_float(real, "mo_score")
    real_safe = as_bool(real, "safe")
    control_mos = [as_float(row, "mo_score") for row in controls]
    max_control_mo = max(control_mos, default=0.0)
    control_safe_count = sum(1 for row in controls if as_bool(row, "safe"))
    margin = real_mo - max_control_mo
    worst_control = ""
    if controls:
        worst_control = max(controls, key=lambda row: as_float(row, "mo_score")).get("control_type", "")

    if not real_safe:
        cls = "REAL_NOT_SAFE"
    elif control_safe_count > 0:
        cls = "CONTROL_SAFE_REJECT"
    elif margin >= thresholds.strong_margin:
        cls = "SELECTIVE_STRONG"
    elif margin >= thresholds.weak_margin:
        cls = "SELECTIVE_WEAK"
    elif margin > 0:
        cls = "SELECTIVE_UNDER_MARGIN"
    else:
        cls = "CONTROL_MO_REJECT"

    details: dict[str, float | int | str | bool] = {
        "real_mo": real_mo,
        "real_safe": real_safe,
        "max_control_mo": max_control_mo,
        "selectivity_margin": margin,
        "control_safe_count": control_safe_count,
        "worst_control": worst_control,
        "smooth_delta": as_float(real, "smooth_delta"),
        "accuracy_delta": as_float(real, "accuracy_delta"),
        "echo_delta": as_float(real, "echo_delta"),
        "unigram_delta": as_float(real, "unigram_delta"),
    }
    return cls, details


def analyze_run(run_dir: Path, out_dir: Path, thresholds: GateThresholds) -> dict[str, object]:
    rows = load_rows(run_dir)
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["arm"], row["candidate_idx"])][row["control_type"]] = row

    candidate_rows: list[dict[str, object]] = []
    by_arm: dict[str, list[dict[str, object]]] = defaultdict(list)
    missing_controls = 0

    for (arm, candidate_idx), control_map in sorted(grouped.items(), key=lambda item: (item[0][0], int(item[0][1]))):
        real = control_map.get("real")
        if real is None:
            continue
        controls = [control_map[name] for name in CONTROL_TYPES if name in control_map]
        if len(controls) < len(CONTROL_TYPES):
            missing_controls += 1
        cls, details = classify_candidate(real, controls, thresholds)
        output_row: dict[str, object] = {
            "source_run": str(run_dir),
            "arm": arm,
            "candidate_idx": candidate_idx,
            "label": real.get("label", ""),
            "projection_selectivity_class": cls,
            **details,
        }
        candidate_rows.append(output_row)
        by_arm[arm].append(output_row)

    arm_rows: list[dict[str, object]] = []
    for arm, arm_candidates in sorted(by_arm.items()):
        passing_candidates = [
            row
            for row in arm_candidates
            if row["projection_selectivity_class"] in {"SELECTIVE_STRONG", "SELECTIVE_WEAK", "SELECTIVE_UNDER_MARGIN"}
        ]
        best_margin = max(arm_candidates, key=lambda row: float(row["selectivity_margin"]))
        best_pass = max(passing_candidates, key=lambda row: float(row["selectivity_margin"])) if passing_candidates else None
        strong_count = sum(1 for row in arm_candidates if row["projection_selectivity_class"] == "SELECTIVE_STRONG")
        weak_count = sum(1 for row in arm_candidates if row["projection_selectivity_class"] == "SELECTIVE_WEAK")
        under_margin_count = sum(1 for row in arm_candidates if row["projection_selectivity_class"] == "SELECTIVE_UNDER_MARGIN")
        real_safe_count = sum(1 for row in arm_candidates if row["real_safe"])
        control_safe_reject_count = sum(1 for row in arm_candidates if row["projection_selectivity_class"] == "CONTROL_SAFE_REJECT")
        if strong_count:
            verdict = "ARM_SELECTIVE_STRONG"
        elif weak_count:
            verdict = "ARM_SELECTIVE_WEAK"
        elif under_margin_count:
            verdict = "ARM_UNDER_MARGIN_ONLY"
        elif control_safe_reject_count:
            verdict = "ARM_CONTROL_SAFE_BLOCKED"
        else:
            verdict = "ARM_NO_SELECTIVE_SIGNAL"
        arm_rows.append(
            {
                "source_run": str(run_dir),
                "arm": arm,
                "verdict": verdict,
                "candidate_count": len(arm_candidates),
                "real_safe_count": real_safe_count,
                "control_safe_reject_count": control_safe_reject_count,
                "selective_strong_count": strong_count,
                "selective_weak_count": weak_count,
                "selective_under_margin_count": under_margin_count,
                "best_margin_candidate_idx": best_margin["candidate_idx"],
                "best_margin_label": best_margin["label"],
                "best_margin_class": best_margin["projection_selectivity_class"],
                "best_margin_real_mo": best_margin["real_mo"],
                "best_margin_max_control_mo": best_margin["max_control_mo"],
                "best_margin_selectivity_margin": best_margin["selectivity_margin"],
                "best_margin_worst_control": best_margin["worst_control"],
                "best_pass_candidate_idx": best_pass["candidate_idx"] if best_pass else "",
                "best_pass_label": best_pass["label"] if best_pass else "",
                "best_pass_class": best_pass["projection_selectivity_class"] if best_pass else "",
                "best_pass_real_mo": best_pass["real_mo"] if best_pass else "",
                "best_pass_max_control_mo": best_pass["max_control_mo"] if best_pass else "",
                "best_pass_selectivity_margin": best_pass["selectivity_margin"] if best_pass else "",
                "best_pass_worst_control": best_pass["worst_control"] if best_pass else "",
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "projection_selectivity_candidates.csv", candidate_rows)
    write_csv(out_dir / "projection_selectivity_arm_summary.csv", arm_rows)

    total_strong = sum(int(row["selective_strong_count"]) for row in arm_rows)
    total_weak = sum(int(row["selective_weak_count"]) for row in arm_rows)
    if total_strong:
        verdict = "D15B_SELECTIVE_CANDIDATE_FOUND"
    elif total_weak:
        verdict = "D15B_WEAK_SELECTIVITY"
    else:
        verdict = "D15B_PROJECTION_SELECTIVITY_BLOCKED"

    summary: dict[str, object] = {
        "verdict": verdict,
        "source_run": str(run_dir),
        "thresholds": {
            "weak_margin": thresholds.weak_margin,
            "strong_margin": thresholds.strong_margin,
        },
        "candidate_count": len(candidate_rows),
        "arm_count": len(arm_rows),
        "missing_control_groups": missing_controls,
        "selective_strong_count": total_strong,
        "selective_weak_count": total_weak,
        "arms": arm_rows,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(out_dir / "D15B_PROJECTION_SELECTIVITY_GATE_REPORT.md", summary)
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# D15B Projection Selectivity Gate",
        "",
        f"- verdict: `{summary['verdict']}`",
        f"- source_run: `{summary['source_run']}`",
        f"- candidate_count: `{summary['candidate_count']}`",
        f"- selective_strong_count: `{summary['selective_strong_count']}`",
        f"- selective_weak_count: `{summary['selective_weak_count']}`",
        "",
        "## Arm Summary",
        "",
        "| arm | verdict | real_safe | control_safe_reject | strong | weak | best_pass_margin | best_pass_class | worst_control |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for arm in summary["arms"]:  # type: ignore[index]
        best_pass_margin = arm["best_pass_selectivity_margin"]
        margin_text = f"{best_pass_margin:.6f}" if isinstance(best_pass_margin, float) else ""
        lines.append(
            f"| {arm['arm']} | {arm['verdict']} | {arm['real_safe_count']} | "
            f"{arm['control_safe_reject_count']} | {arm['selective_strong_count']} | "
            f"{arm['selective_weak_count']} | {margin_text} | {arm['best_pass_class']} | "
            f"{arm['best_pass_worst_control']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A candidate passes this gate only if the real row is safe and the same candidate is not safe under any adversarial control.",
            "This is stricter than arm-level safe-rate comparison and is intended to catch projection/readout artifacts.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True, help="D10p/D15 run directory containing semantic_candidates.csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--weak-margin", type=float, default=0.00025)
    parser.add_argument("--strong-margin", type=float, default=0.001)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_root = Path(args.out)
    thresholds = GateThresholds(weak_margin=args.weak_margin, strong_margin=args.strong_margin)
    summaries = []
    for run_dir_text in args.run_dir:
        run_dir = Path(run_dir_text)
        safe_name = run_dir.name
        summaries.append(analyze_run(run_dir, out_root / safe_name, thresholds))

    decisive_summaries = [summary for summary in summaries if "confirm" in str(summary["source_run"]).lower()]
    if not decisive_summaries:
        decisive_summaries = summaries
    aggregate = {
        "verdict": "D15B_SELECTIVE_CANDIDATE_FOUND"
        if any(summary["verdict"] == "D15B_SELECTIVE_CANDIDATE_FOUND" for summary in decisive_summaries)
        else "D15B_WEAK_SELECTIVITY"
        if any(summary["verdict"] == "D15B_WEAK_SELECTIVITY" for summary in decisive_summaries)
        else "D15B_PROJECTION_SELECTIVITY_BLOCKED",
        "decisive_source": "confirm" if decisive_summaries != summaries else "all_runs",
        "runs": summaries,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "run_summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

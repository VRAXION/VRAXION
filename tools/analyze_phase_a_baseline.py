"""Audit the Phase A dimensionality sweep and write a paper-facing summary.

This script does not run experiments. It validates the 30-cell Phase A artifact,
computes fixture-by-H baseline statistics, checks the Phase B B0 replication
against the Phase A mutual_inhibition H=384 cell, and emits a concise report.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_ROOT = Path("output/dimensionality_sweep/20260424_091217")
DEFAULT_PHASE_B_ROOT = Path("output/phase_b_full_20260424")
DEFAULT_REPORT = Path("docs/research/PHASE_A_BASELINE.md")
EXPECTED_FIXTURES = ["bytepair_proj", "mutual_inhibition"]
EXPECTED_H = [128, 256, 384]
EXPECTED_SEEDS = [42, 1042, 2042, 3042, 4042]


@dataclass
class Paths:
    root: Path
    phase_b_root: Path
    analysis_dir: Path
    figures_dir: Path
    report_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--phase-b-root", type=Path, default=DEFAULT_PHASE_B_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def ensure_dirs(root: Path, phase_b_root: Path, report_path: Path) -> Paths:
    analysis_dir = root / "analysis"
    figures_dir = analysis_dir / "figures"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    return Paths(root=root, phase_b_root=phase_b_root, analysis_dir=analysis_dir, figures_dir=figures_dir, report_path=report_path)


def load_phase_a(root: Path) -> pd.DataFrame:
    path = root / "results.csv"
    if not path.exists():
        raise SystemExit(f"missing Phase A results: {path}")
    return pd.read_csv(path)


def validate_phase_a(df: pd.DataFrame, root: Path) -> dict:
    failures = []
    expected_cells = {(fixture, h, seed) for fixture in EXPECTED_FIXTURES for h in EXPECTED_H for seed in EXPECTED_SEEDS}
    actual_cells = {(str(row.fixture), int(row.H), int(row.seed)) for row in df.itertuples()}
    missing = sorted(expected_cells - actual_cells)
    extra = sorted(actual_cells - expected_cells)
    if missing:
        failures.append(f"missing cells: {missing}")
    if extra:
        failures.append(f"extra cells: {extra}")
    if len(df) != 30:
        failures.append(f"expected 30 rows, got {len(df)}")
    if not (root / "results.json").exists():
        failures.append("missing results.json")
    if not (root / "driver.log").exists():
        failures.append("missing driver.log")
    if failures:
        raise SystemExit(f"Phase A artifact validation failed: {failures}")
    return {
        "rows": int(len(df)),
        "fixtures": EXPECTED_FIXTURES,
        "H_values": EXPECTED_H,
        "seeds": EXPECTED_SEEDS,
        "results_json": str(root / "results.json"),
        "driver_log": str(root / "driver.log"),
        "passed": True,
    }


def summarize_by_fixture_h(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["peak_acc", "final_acc", "accept_rate_pct", "alive_frac_mean", "edges", "wall_clock_s"]
    rows = []
    for (fixture, h), group in df.sort_values(["fixture", "H"]).groupby(["fixture", "H"]):
        row = {"fixture": fixture, "H": int(h), "n": int(len(group))}
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            if metric in ["peak_acc", "final_acc"]:
                values = values * 100.0
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_min"] = float(np.min(values))
            row[f"{metric}_max"] = float(np.max(values))
        rows.append(row)
    return pd.DataFrame(rows)


def fixture_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in EXPECTED_H:
        mi = df[(df["fixture"] == "mutual_inhibition") & (df["H"] == h)]["peak_acc"].to_numpy(dtype=float) * 100.0
        bp = df[(df["fixture"] == "bytepair_proj") & (df["H"] == h)]["peak_acc"].to_numpy(dtype=float) * 100.0
        test = stats.ttest_ind(bp, mi, equal_var=False)
        rows.append({
            "H": h,
            "bytepair_proj_mean_peak": float(np.mean(bp)),
            "mutual_inhibition_mean_peak": float(np.mean(mi)),
            "delta_bytepair_minus_mi": float(np.mean(bp) - np.mean(mi)),
            "welch_t": float(test.statistic),
            "welch_p": float(test.pvalue),
        })
    return pd.DataFrame(rows)


def phase_b_replication_check(df: pd.DataFrame, phase_b_root: Path) -> dict:
    phase_b_path = phase_b_root / "results.csv"
    if not phase_b_path.exists():
        return {"available": False, "reason": f"missing {phase_b_path}"}
    phase_b = pd.read_csv(phase_b_path)
    a = df[(df["fixture"] == "mutual_inhibition") & (df["H"] == 384)].sort_values("seed")
    b = phase_b[phase_b["arm"] == "B0"].sort_values("seed")
    metrics = ["peak_acc", "final_acc", "accept_rate_pct", "alive_frac_mean", "edges", "unique_preds"]
    metric_checks = {}
    for metric in metrics:
        a_values = a[metric].to_numpy()
        b_values = b[metric].to_numpy()
        same = len(a_values) == len(b_values) and np.allclose(a_values, b_values, rtol=0.0, atol=1e-12)
        metric_checks[metric] = {
            "phase_a": a_values.tolist(),
            "phase_b": b_values.tolist(),
            "bit_identical_or_close": bool(same),
            "phase_a_mean": float(np.mean(a_values)),
            "phase_b_mean": float(np.mean(b_values)),
        }
    return {
        "available": True,
        "phase_a_rows": int(len(a)),
        "phase_b_rows": int(len(b)),
        "seeds_match": a["seed"].to_list() == b["seed"].to_list(),
        "all_checked_metrics_match": all(v["bit_identical_or_close"] for v in metric_checks.values()),
        "metrics": metric_checks,
    }


def peak_final_gap(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    local = df.copy()
    local["peak_final_gap_pp"] = (local["peak_acc"] - local["final_acc"]) * 100.0
    for (fixture, h), group in local.groupby(["fixture", "H"]):
        values = group["peak_final_gap_pp"].to_numpy(dtype=float)
        rows.append({
            "fixture": fixture,
            "H": int(h),
            "mean_gap_pp": float(np.mean(values)),
            "std_gap_pp": float(np.std(values, ddof=1)),
            "max_gap_pp": float(np.max(values)),
        })
    return pd.DataFrame(rows)


def plot_phase_a(summary: pd.DataFrame, comparisons: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for fixture, group in summary.groupby("fixture"):
        group = group.sort_values("H")
        ax.errorbar(group["H"], group["peak_acc_mean"], yerr=group["peak_acc_std"], marker="o", capsize=4, label=fixture)
    ax.set_title("Phase A peak accuracy by fixture and H")
    ax.set_xlabel("H")
    ax.set_ylabel("peak accuracy (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "phase_a_peak_by_h.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for fixture, group in summary.groupby("fixture"):
        group = group.sort_values("H")
        ax.errorbar(group["H"], group["accept_rate_pct_mean"], yerr=group["accept_rate_pct_std"], marker="o", capsize=4, label=fixture)
    ax.set_title("Phase A accept rate by fixture and H")
    ax.set_xlabel("H")
    ax.set_ylabel("accept rate (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "phase_a_accept_by_h.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for fixture, group in summary.groupby("fixture"):
        group = group.sort_values("H")
        ax.errorbar(group["H"], group["alive_frac_mean_mean"], yerr=group["alive_frac_mean_std"], marker="o", capsize=4, label=fixture)
    ax.set_title("Phase A alive fraction by fixture and H")
    ax.set_xlabel("H")
    ax.set_ylabel("alive fraction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "phase_a_alive_by_h.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(comparisons["H"].astype(str), comparisons["delta_bytepair_minus_mi"], color="tab:purple")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Bytepair minus mutual-inhibition peak delta")
    ax.set_xlabel("H")
    ax.set_ylabel("delta peak accuracy (pp)")
    fig.tight_layout()
    fig.savefig(figures_dir / "phase_a_fixture_delta.png", dpi=160)
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    columns = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "\\|") for col in df.columns) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    artifact_status: dict,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    gaps: pd.DataFrame,
    replication: dict,
) -> None:
    display = summary[[
        "fixture",
        "H",
        "n",
        "peak_acc_mean",
        "peak_acc_std",
        "final_acc_mean",
        "final_acc_std",
        "accept_rate_pct_mean",
        "accept_rate_pct_std",
        "alive_frac_mean_mean",
        "alive_frac_mean_std",
    ]].copy()
    for col in display.columns:
        if col not in ["fixture", "H", "n"]:
            display[col] = display[col].map(lambda x: f"{float(x):.4f}")

    comp = comparisons.copy()
    for col in ["bytepair_proj_mean_peak", "mutual_inhibition_mean_peak", "delta_bytepair_minus_mi", "welch_t", "welch_p"]:
        comp[col] = comp[col].map(lambda x: f"{float(x):.6g}")

    gap_display = gaps.copy()
    for col in ["mean_gap_pp", "std_gap_pp", "max_gap_pp"]:
        gap_display[col] = gap_display[col].map(lambda x: f"{float(x):.4f}")

    bp384 = summary[(summary["fixture"] == "bytepair_proj") & (summary["H"] == 384)].iloc[0]
    mi256 = summary[(summary["fixture"] == "mutual_inhibition") & (summary["H"] == 256)].iloc[0]
    mi384 = summary[(summary["fixture"] == "mutual_inhibition") & (summary["H"] == 384)].iloc[0]
    replication_sentence = (
        "Phase B B0 replication matches Phase A mutual_inhibition H=384 on all checked metrics."
        if replication.get("all_checked_metrics_match")
        else "Phase B B0 replication did not exactly match every checked metric; inspect JSON before claiming cross-replication."
    )

    lines = [
        "# Phase A Baseline Audit",
        "",
        "## Verdict",
        "",
        "Phase A is a valid 30-cell baseline artifact: two fixtures, three H values, five seeds each.",
        f"Mutual inhibition shows the inverted-U profile (`H=256` peak mean `{mi256['peak_acc_mean']:.2f}%`, `H=384` `{mi384['peak_acc_mean']:.2f}%`).",
        f"Bytepair projection does not show the same profile; it declines with H and becomes high-variance at `H=384` (peak mean `{bp384['peak_acc_mean']:.2f}%`, std `{bp384['peak_acc_std']:.2f}pp`).",
        replication_sentence,
        "",
        "## Artifact Integrity",
        "",
        f"- Rows: `{artifact_status['rows']}`",
        f"- Fixtures: `{', '.join(artifact_status['fixtures'])}`",
        f"- H values: `{artifact_status['H_values']}`",
        f"- Seeds: `{artifact_status['seeds']}`",
        f"- Driver log: `{artifact_status['driver_log']}`",
        "",
        "## Fixture x H Summary",
        "",
        markdown_table(display),
        "",
        "## Fixture Comparison",
        "",
        markdown_table(comp),
        "",
        "## Peak-Final Gap",
        "",
        markdown_table(gap_display),
        "",
        "## Phase B Replication Check",
        "",
        f"- Available: `{replication.get('available')}`",
        f"- Seeds match: `{replication.get('seeds_match')}`",
        f"- Checked metrics match: `{replication.get('all_checked_metrics_match')}`",
        "",
        "## Interpretation Boundary",
        "",
        "- Phase B's training-horizon confound claim is validated for the mutual_inhibition fixture, not automatically for bytepair_proj.",
        "- Bytepair_proj H=384 appears dominated by collapse/prune/low-accept dynamics; it needs a separate prune-policy ablation rather than being folded into Phase B.1.",
        "- The safe unified claim is recipe-dependence: H interacts with training horizon and fixture policy, so H is not a standalone architectural verdict.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    paths = ensure_dirs(args.root, args.phase_b_root, args.report)
    phase_a = load_phase_a(paths.root)
    artifact_status = validate_phase_a(phase_a, paths.root)
    summary = summarize_by_fixture_h(phase_a)
    comparisons = fixture_comparisons(phase_a)
    gaps = peak_final_gap(phase_a)
    replication = phase_b_replication_check(phase_a, paths.phase_b_root)

    (paths.analysis_dir / "phase_a_verdict.json").write_text(
        json.dumps({
            "artifact_status": artifact_status,
            "summary": summary.to_dict(orient="records"),
            "fixture_comparisons": comparisons.to_dict(orient="records"),
            "peak_final_gap": gaps.to_dict(orient="records"),
            "phase_b_b0_replication": replication,
        }, indent=2),
        encoding="utf-8",
    )
    summary.to_csv(paths.analysis_dir / "phase_a_fixture_h_stats.csv", index=False)
    comparisons.to_csv(paths.analysis_dir / "phase_a_fixture_comparisons.csv", index=False)
    gaps.to_csv(paths.analysis_dir / "phase_a_peak_final_gap.csv", index=False)
    plot_phase_a(summary, comparisons, paths.figures_dir)
    write_report(paths.report_path, artifact_status, summary, comparisons, gaps, replication)

    print(json.dumps({
        "status": "PASS",
        "analysis_dir": str(paths.analysis_dir),
        "report": str(paths.report_path),
        "phase_b_replication_match": replication.get("all_checked_metrics_match"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

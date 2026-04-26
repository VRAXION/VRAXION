"""Analyze Phase D3 coarse K-lock artifacts.

D3 locks the K axis of the Search Aperture Function under strict acceptance:
  SAF(K, tau=0, s=0)

It merges existing strict anchors:
  - D2 H={128,256}, K={1,3,9}
  - D1 H=384, K={1,3,9}

with new D3 coarse samples:
  - H={128,256,384}, K={5,13,18}
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


DEFAULT_D3_ROOT = Path("output/phase_d3_klock_coarse_20260426")
DEFAULT_D2_ROOT = Path("output/phase_d2_cross_h_activation_20260426")
DEFAULT_D1_ROOT = Path("output/phase_d1_activation_20260425")
DEFAULT_REPORT = Path("docs/research/PHASE_D3_K_LOCK_VERDICT.md")
EXPECTED_D3_ARMS = ["D3_K5_STRICT", "D3_K13_STRICT", "D3_K18_STRICT"]
ANCHOR_SEEDS_DEFAULT = "42,1042,2042"


@dataclass
class Paths:
    d3_root: Path
    d2_root: Path
    d1_root: Path
    analysis_dir: Path
    figures_dir: Path
    report_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d3-root", type=Path, default=DEFAULT_D3_ROOT)
    parser.add_argument("--d2-root", type=Path, default=DEFAULT_D2_ROOT)
    parser.add_argument("--d1-root", type=Path, default=DEFAULT_D1_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--expected-H", default="128,256,384")
    parser.add_argument("--expected-d3-seeds", type=int, default=3)
    parser.add_argument("--expected-d3-steps", type=int, default=40_000)
    parser.add_argument("--anchor-seeds", default=ANCHOR_SEEDS_DEFAULT)
    parser.add_argument("--lock-margin-pp", type=float, default=0.5)
    return parser.parse_args()


def ensure_dirs(args: argparse.Namespace) -> Paths:
    analysis_dir = args.d3_root / "analysis"
    figures_dir = analysis_dir / "figures"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    return Paths(args.d3_root, args.d2_root, args.d1_root, analysis_dir, figures_dir, args.report)


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"required artifact missing: {path}")
    return pd.read_csv(path)


def normalize_results(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["H"] = pd.to_numeric(out["H"], errors="coerce").astype("Int64")
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce").astype("Int64")
    out["jackpot"] = pd.to_numeric(out["jackpot"], errors="coerce").astype("Int64")
    for metric in ["peak_acc", "final_acc"]:
        values = pd.to_numeric(out[metric], errors="coerce")
        out[f"{metric}_pct"] = values * 100.0 if values.max(skipna=True) <= 1.0 else values
    out["peak_final_gap_pp"] = out["peak_acc_pct"] - out["final_acc_pct"]
    return out


def normalize_construct(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["H"] = pd.to_numeric(out["H"], errors="coerce").astype("Int64")
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce").astype("Int64")
    out["jackpot"] = pd.to_numeric(out["jackpot"], errors="coerce").astype("Int64")
    return out


def strict_only(df: pd.DataFrame) -> pd.DataFrame:
    policy = df.get("accept_policy", pd.Series([""] * len(df))).astype(str).str.lower()
    arm = df.get("arm", pd.Series([""] * len(df))).astype(str)
    ties = df.get("accept_ties", pd.Series([False] * len(df))).astype(str).str.lower()
    return df[((policy == "strict") | arm.str.endswith("_STRICT")) & ~ties.isin(["true", "1", "yes"])].copy()


def load_anchor_results(root: Path, source: str, h_values: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = normalize_results(read_csv_required(root / "results.csv"))
    construct = normalize_construct(read_csv_required(root / "constructability_summary.csv"))
    results = strict_only(results)
    construct = strict_only(construct)
    results = results[results["H"].isin(h_values) & results["jackpot"].isin([1, 3, 9])].copy()
    construct = construct[construct["H"].isin(h_values) & construct["jackpot"].isin([1, 3, 9])].copy()
    results["source"] = source
    construct["source"] = source
    return results, construct


def load_d3_results(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = normalize_results(read_csv_required(root / "results.csv"))
    construct = normalize_construct(read_csv_required(root / "constructability_summary.csv"))
    results = strict_only(results)
    construct = strict_only(construct)
    results["source"] = "D3"
    construct["source"] = "D3"
    return results, construct


def validate_d3(paths: Paths, d3_results: pd.DataFrame, d3_construct: pd.DataFrame, expected_h: list[int], expected_seeds: int) -> dict:
    expected_runs = len(expected_h) * len(EXPECTED_D3_ARMS) * expected_seeds
    candidates = list(paths.d3_root.glob("H_*/D3_*/seed_*/candidates.csv"))
    checkpoints = list(paths.d3_root.glob("H_*/D3_*/seed_*/final.ckpt"))
    panel_summaries = list(paths.d3_root.glob("H_*/D3_*/seed_*/panel_summary.json"))
    panel_timeseries = list(paths.d3_root.glob("H_*/D3_*/seed_*/panel_timeseries.csv"))
    failures = []
    if len(d3_results) != expected_runs:
        failures.append(f"expected {expected_runs} D3 result rows, got {len(d3_results)}")
    if len(d3_construct) != expected_runs:
        failures.append(f"expected {expected_runs} D3 constructability rows, got {len(d3_construct)}")
    for label, files in [
        ("candidate logs", candidates),
        ("checkpoints", checkpoints),
        ("panel summaries", panel_summaries),
        ("panel timeseries", panel_timeseries),
    ]:
        if len(files) != expected_runs:
            failures.append(f"expected {expected_runs} {label}, got {len(files)}")
    missing_arms = sorted(set(EXPECTED_D3_ARMS) - set(d3_results["arm"]))
    missing_h = sorted(set(expected_h) - {int(x) for x in d3_results["H"].dropna().unique()})
    if missing_arms:
        failures.append(f"missing D3 arms: {missing_arms}")
    if missing_h:
        failures.append(f"missing H values: {missing_h}")
    expected_candidate_rows = int(pd.to_numeric(d3_results["expected_candidate_rows"], errors="coerce").sum())
    actual_candidate_rows = int(pd.to_numeric(d3_construct["candidate_rows"], errors="coerce").sum())
    if expected_candidate_rows != actual_candidate_rows:
        failures.append(f"candidate rows mismatch expected={expected_candidate_rows} actual={actual_candidate_rows}")
    if failures:
        raise SystemExit(f"Phase D3 artifact validation failed: {failures}")
    return {
        "passed": True,
        "runs": int(len(d3_results)),
        "candidate_rows": actual_candidate_rows,
        "checkpoints": len(checkpoints),
        "panel_summaries": len(panel_summaries),
        "panel_timeseries_files": len(panel_timeseries),
    }


def summarize(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: value for col, value in zip(group_cols, key)}
        row["n"] = int(len(group))
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_min"] = float(np.min(values))
            row[f"{metric}_max"] = float(np.max(values))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols) if rows else pd.DataFrame()


def merge_construct_stats(result_stats: pd.DataFrame, construct_stats: pd.DataFrame) -> pd.DataFrame:
    keep = ["H", "jackpot", "C_K_window_ratio_mean", "cost_eval_ms_mean", "accepted_positive_steps_mean", "accepted_nonpositive_steps_mean"]
    available = [col for col in keep if col in construct_stats.columns]
    return result_stats.merge(construct_stats[available], on=["H", "jackpot"], how="left")


def classify_verdict(seed_stats: pd.DataFrame, lock_margin: float) -> tuple[str, list[str], pd.DataFrame]:
    rows = []
    notes = []
    non9_wins = []
    high_k_wins = []
    for h, group in seed_stats.groupby("H"):
        group = group.sort_values("peak_acc_pct_mean", ascending=False)
        best = group.iloc[0]
        k9 = group[group["jackpot"] == 9]
        if k9.empty:
            continue
        k9_mean = float(k9.iloc[0]["peak_acc_pct_mean"])
        best_delta = float(best["peak_acc_pct_mean"]) - k9_mean
        rows.append({
            "H": int(h),
            "best_K": int(best["jackpot"]),
            "best_peak_mean": float(best["peak_acc_pct_mean"]),
            "K9_peak_mean": k9_mean,
            "delta_best_vs_K9_pp": best_delta,
        })
        if int(best["jackpot"]) != 9 and best_delta >= lock_margin:
            non9_wins.append((int(h), int(best["jackpot"]), best_delta))
        higher = group[group["jackpot"].isin([13, 18])]
        if not higher.empty:
            high_best = higher.sort_values("peak_acc_pct_mean", ascending=False).iloc[0]
            high_delta = float(high_best["peak_acc_pct_mean"]) - k9_mean
            if high_delta >= lock_margin:
                high_k_wins.append((int(h), int(high_best["jackpot"]), high_delta))
    winner_df = pd.DataFrame(rows).sort_values("H") if rows else pd.DataFrame()

    if not non9_wins:
        verdict = "K=9 LOCK"
        notes.append(f"No non-9 K beats K=9 by the lock margin ({lock_margin:.2f}pp) on any H.")
    elif high_k_wins and (any(h == 384 for h, _, _ in high_k_wins) or len(high_k_wins) >= 2):
        verdict = "FINE SWEEP REQUIRED"
        notes.append("A higher-K candidate beats K=9 by the lock margin on H=384 or at least two H values.")
    else:
        verdict = "K(H) TABLE"
        notes.append("The best K appears H-dependent under the lock margin; use a provisional K(H) table and fine sweep the affected region.")

    for h, k, delta in non9_wins:
        notes.append(f"H={h}: K={k} beats K=9 by {delta:.2f}pp.")
    return verdict, notes, winner_df


def plot_k_curve(stats: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for h, group in stats.groupby("H"):
        group = group.sort_values("jackpot")
        ax.errorbar(
            group["jackpot"],
            group["peak_acc_pct_mean"],
            yerr=group["peak_acc_pct_std"],
            marker="o",
            capsize=4,
            label=f"H={int(h)}",
        )
    ax.set_title("Phase D3 K-lock: peak accuracy vs K")
    ax.set_xlabel("K / jackpot candidates")
    ax.set_ylabel("peak accuracy (%)")
    ax.set_xticks([1, 3, 5, 9, 13, 18])
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "phase_d3_k_curve.png", dpi=160)
    plt.close(fig)


def write_report(
    paths: Paths,
    validation: dict,
    seed_stats: pd.DataFrame,
    full_anchor_stats: pd.DataFrame,
    winner_df: pd.DataFrame,
    verdict: str,
    notes: list[str],
    lock_margin: float,
    expected_d3_steps: int,
) -> None:
    lines = [
        "# Phase D3 Verdict: Search Aperture Function K-Lock",
        "",
        "D3 tests the K axis of the Search Aperture Function under strict acceptance (`tau=0`, `s=0`).",
        "",
        "## Integrity",
        "",
        f"- D3 runs: `{validation['runs']}`",
        f"- D3 candidate rows: `{validation['candidate_rows']}`",
        f"- D3 checkpoints: `{validation['checkpoints']}`",
        f"- D3 panel summaries: `{validation['panel_summaries']}`",
        f"- D3 panel timeseries files: `{validation['panel_timeseries_files']}`",
        "",
        "## Verdict",
        "",
        f"- Result: **{verdict}**",
        f"- Lock margin: `{lock_margin:.2f}pp` mean peak over K=9",
    ]
    lines.extend(f"- {note}" for note in notes)
    lines.extend([
        "",
        "## Seed-Matched K Grid",
        "",
        "Primary comparison uses seeds `42,1042,2042` for every K, because D3 new K values are n=3.",
        "",
        "| H | K | n | peak mean | peak std | final mean | accept mean | alive mean | C_K mean | wall/candidate ms |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in seed_stats.sort_values(["H", "jackpot"]).iterrows():
        wall_ms = float(row.get("wall_clock_s_mean", float("nan"))) * 1000.0 / (expected_d3_steps * int(row["jackpot"]))
        lines.append(
            f"| {int(row['H'])} | {int(row['jackpot'])} | {int(row['n'])} | "
            f"{row.get('peak_acc_pct_mean', float('nan')):.2f} | "
            f"{row.get('peak_acc_pct_std', float('nan')):.2f} | "
            f"{row.get('final_acc_pct_mean', float('nan')):.2f} | "
            f"{row.get('accept_rate_pct_mean', float('nan')):.2f} | "
            f"{row.get('alive_frac_mean_mean', float('nan')):.3f} | "
            f"{row.get('C_K_window_ratio_mean', float('nan')):.3e} | "
            f"{wall_ms:.3f} |"
        )
    lines.extend([
        "",
        "## Winner Table",
        "",
        "| H | best K | best peak mean | K9 peak mean | delta vs K9 |",
        "|---:|---:|---:|---:|---:|",
    ])
    for _, row in winner_df.iterrows():
        lines.append(
            f"| {int(row['H'])} | {int(row['best_K'])} | {row['best_peak_mean']:.2f} | "
            f"{row['K9_peak_mean']:.2f} | {row['delta_best_vs_K9_pp']:.2f} |"
        )
    if not full_anchor_stats.empty:
        lines.extend([
            "",
            "## Full Anchor Context",
            "",
            "Existing K={1,3,9} anchors are also retained at full n=5 where available.",
            "",
            "| H | K | n | peak mean | peak std |",
            "|---:|---:|---:|---:|---:|",
        ])
        for _, row in full_anchor_stats.sort_values(["H", "jackpot"]).iterrows():
            lines.append(
                f"| {int(row['H'])} | {int(row['jackpot'])} | {int(row['n'])} | "
                f"{row.get('peak_acc_pct_mean', float('nan')):.2f} | {row.get('peak_acc_pct_std', float('nan')):.2f} |"
            )
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- Machine summary: `{paths.analysis_dir / 'phase_d3_klock_verdict.json'}`",
        f"- Seed-matched stats: `{paths.analysis_dir / 'phase_d3_seed_matched_stats.csv'}`",
        f"- Full anchor stats: `{paths.analysis_dir / 'phase_d3_full_anchor_stats.csv'}`",
        f"- Figure: `{paths.figures_dir / 'phase_d3_k_curve.png'}`",
    ])
    paths.report_path.write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    paths = ensure_dirs(args)
    expected_h = [int(x) for x in args.expected_H.split(",") if x.strip()]
    anchor_seeds = [int(x) for x in args.anchor_seeds.split(",") if x.strip()]

    d3_results, d3_construct = load_d3_results(paths.d3_root)
    validation = validate_d3(paths, d3_results, d3_construct, expected_h, args.expected_d3_seeds)

    d2_results, d2_construct = load_anchor_results(paths.d2_root, "D2", [128, 256])
    d1_results, d1_construct = load_anchor_results(paths.d1_root, "D1", [384])

    anchors = pd.concat([d2_results, d1_results], ignore_index=True)
    anchor_construct = pd.concat([d2_construct, d1_construct], ignore_index=True)
    combined = pd.concat([anchors, d3_results], ignore_index=True)
    combined_construct = pd.concat([anchor_construct, d3_construct], ignore_index=True)

    seed_matched = combined[combined["seed"].isin(anchor_seeds)].copy()
    seed_matched_construct = combined_construct[combined_construct["seed"].isin(anchor_seeds)].copy()
    result_metrics = ["peak_acc_pct", "final_acc_pct", "peak_final_gap_pp", "accept_rate_pct", "alive_frac_mean", "wall_clock_s"]
    construct_metrics = ["C_K_window_ratio", "cost_eval_ms", "accepted_positive_steps", "accepted_nonpositive_steps"]
    seed_stats = summarize(seed_matched, ["H", "jackpot"], result_metrics)
    construct_stats = summarize(seed_matched_construct, ["H", "jackpot"], construct_metrics)
    seed_stats = merge_construct_stats(seed_stats, construct_stats)
    full_anchor_stats = summarize(anchors, ["H", "jackpot"], result_metrics)

    verdict, notes, winner_df = classify_verdict(seed_stats, args.lock_margin_pp)
    plot_k_curve(seed_stats, paths.figures_dir)

    seed_stats.to_csv(paths.analysis_dir / "phase_d3_seed_matched_stats.csv", index=False)
    full_anchor_stats.to_csv(paths.analysis_dir / "phase_d3_full_anchor_stats.csv", index=False)
    winner_df.to_csv(paths.analysis_dir / "phase_d3_winners.csv", index=False)
    payload = {
        "d3_root": str(paths.d3_root),
        "d2_root": str(paths.d2_root),
        "d1_root": str(paths.d1_root),
        "validation": validation,
        "anchor_seeds": anchor_seeds,
        "lock_margin_pp": args.lock_margin_pp,
        "verdict": verdict,
        "notes": notes,
        "winners": json.loads(winner_df.to_json(orient="records")),
        "seed_matched_stats": json.loads(seed_stats.to_json(orient="records")),
        "full_anchor_stats": json.loads(full_anchor_stats.to_json(orient="records")),
    }
    (paths.analysis_dir / "phase_d3_klock_verdict.json").write_text(json.dumps(payload, indent=2))
    write_report(
        paths,
        validation,
        seed_stats,
        full_anchor_stats,
        winner_df,
        verdict,
        notes,
        args.lock_margin_pp,
        args.expected_d3_steps,
    )

    print(json.dumps({
        "status": "PASS",
        "verdict": verdict,
        "runs": validation["runs"],
        "candidate_rows": validation["candidate_rows"],
        "winners": json.loads(winner_df.to_json(orient="records")),
        "report": str(paths.report_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

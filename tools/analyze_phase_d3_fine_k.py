"""Analyze Phase D3.1 fine K-lock artifacts.

D3.1 resolves the H=256 region opened by Phase D3:
  SAF(K, tau=0, s=0), strict policy

New D3.1 samples:
  - H=256
  - K={15,18,21,24}
  - 5 seeds

The analyzer also merges prior strict K-axis anchors and writes a compact
formula report for the empirical K(H) lock.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_D3F_ROOT = Path("output/phase_d3_fine_k_20260426")
DEFAULT_D3_ROOT = Path("output/phase_d3_klock_coarse_20260426")
DEFAULT_D2_ROOT = Path("output/phase_d2_cross_h_activation_20260426")
DEFAULT_D1_ROOT = Path("output/phase_d1_activation_20260425")
DEFAULT_FINE_REPORT = Path("docs/research/PHASE_D3_FINE_K_VERDICT.md")
DEFAULT_FORMULA_REPORT = Path("docs/research/SAF_K_FORMULA_LOCK.md")
EXPECTED_D3F_ARMS = ["D3F_K15_STRICT", "D3F_K18_STRICT", "D3F_K21_STRICT", "D3F_K24_STRICT"]


@dataclass
class Paths:
    d3f_root: Path
    d3_root: Path
    d2_root: Path
    d1_root: Path
    analysis_dir: Path
    figures_dir: Path
    fine_report: Path
    formula_report: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d3f-root", type=Path, default=DEFAULT_D3F_ROOT)
    parser.add_argument("--d3-root", type=Path, default=DEFAULT_D3_ROOT)
    parser.add_argument("--d2-root", type=Path, default=DEFAULT_D2_ROOT)
    parser.add_argument("--d1-root", type=Path, default=DEFAULT_D1_ROOT)
    parser.add_argument("--fine-report", type=Path, default=DEFAULT_FINE_REPORT)
    parser.add_argument("--formula-report", type=Path, default=DEFAULT_FORMULA_REPORT)
    parser.add_argument("--expected-H", type=int, default=256)
    parser.add_argument("--expected-d3f-seeds", type=int, default=5)
    parser.add_argument("--expected-d3f-steps", type=int, default=40_000)
    parser.add_argument("--lock-margin-pp", type=float, default=0.5)
    parser.add_argument("--collapse-threshold-pp", type=float, default=1.0)
    parser.add_argument("--variance-penalty-pp", type=float, default=1.0)
    return parser.parse_args()


def ensure_dirs(args: argparse.Namespace) -> Paths:
    analysis_dir = args.d3f_root / "analysis"
    figures_dir = analysis_dir / "figures"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    args.fine_report.parent.mkdir(parents=True, exist_ok=True)
    args.formula_report.parent.mkdir(parents=True, exist_ok=True)
    return Paths(args.d3f_root, args.d3_root, args.d2_root, args.d1_root, analysis_dir, figures_dir, args.fine_report, args.formula_report)


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


def load_run_pair(root: Path, source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = strict_only(normalize_results(read_csv_required(root / "results.csv")))
    construct = strict_only(normalize_construct(read_csv_required(root / "constructability_summary.csv")))
    results["source"] = source
    construct["source"] = source
    return results, construct


def validate_d3f(paths: Paths, d3f_results: pd.DataFrame, d3f_construct: pd.DataFrame, expected_h: int, expected_seeds: int) -> dict:
    expected_runs = len(EXPECTED_D3F_ARMS) * expected_seeds
    candidates = list(paths.d3f_root.glob("H_*/D3F_*/seed_*/candidates.csv"))
    checkpoints = list(paths.d3f_root.glob("H_*/D3F_*/seed_*/final.ckpt"))
    panel_summaries = list(paths.d3f_root.glob("H_*/D3F_*/seed_*/panel_summary.json"))
    panel_timeseries = list(paths.d3f_root.glob("H_*/D3F_*/seed_*/panel_timeseries.csv"))
    failures = []
    d3f_results = d3f_results[d3f_results["H"] == expected_h]
    d3f_construct = d3f_construct[d3f_construct["H"] == expected_h]
    if len(d3f_results) != expected_runs:
        failures.append(f"expected {expected_runs} D3.1 result rows, got {len(d3f_results)}")
    if len(d3f_construct) != expected_runs:
        failures.append(f"expected {expected_runs} D3.1 constructability rows, got {len(d3f_construct)}")
    for label, files in [
        ("candidate logs", candidates),
        ("checkpoints", checkpoints),
        ("panel summaries", panel_summaries),
        ("panel timeseries", panel_timeseries),
    ]:
        if len(files) != expected_runs:
            failures.append(f"expected {expected_runs} {label}, got {len(files)}")
    missing_arms = sorted(set(EXPECTED_D3F_ARMS) - set(d3f_results["arm"]))
    if missing_arms:
        failures.append(f"missing D3.1 arms: {missing_arms}")
    expected_candidate_rows = int(pd.to_numeric(d3f_results["expected_candidate_rows"], errors="coerce").sum())
    actual_candidate_rows = int(pd.to_numeric(d3f_construct["candidate_rows"], errors="coerce").sum())
    if expected_candidate_rows != actual_candidate_rows:
        failures.append(f"candidate rows mismatch expected={expected_candidate_rows} actual={actual_candidate_rows}")
    if failures:
        raise SystemExit(f"Phase D3.1 artifact validation failed: {failures}")
    return {
        "passed": True,
        "runs": int(len(d3f_results)),
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


def add_collapse_counts(stats: pd.DataFrame, raw: pd.DataFrame, threshold_pp: float) -> pd.DataFrame:
    counts = []
    for key, group in raw.groupby(["H", "jackpot"]):
        h, jackpot = key
        counts.append({
            "H": h,
            "jackpot": jackpot,
            "collapse_count": int((pd.to_numeric(group["peak_acc_pct"], errors="coerce") < threshold_pp).sum()),
        })
    return stats.merge(pd.DataFrame(counts), on=["H", "jackpot"], how="left")


def merge_construct_stats(result_stats: pd.DataFrame, construct_stats: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "H", "jackpot", "C_K_window_ratio_mean", "cost_eval_ms_mean", "V_raw_mean", "M_pos_mean", "R_neg_mean",
        "accepted_positive_steps_mean", "accepted_nonpositive_steps_mean", "positive_delta_rows_mean",
    ]
    available = [col for col in keep if col in construct_stats.columns]
    return result_stats.merge(construct_stats[available], on=["H", "jackpot"], how="left")


def classify_fine(fine_stats: pd.DataFrame, margin: float, variance_penalty: float) -> tuple[str, list[str]]:
    notes = []
    k18 = fine_stats[fine_stats["jackpot"] == 18]
    if k18.empty:
        return "H256_UNSTABLE_NO_LOCK", ["K=18 missing from D3.1 stats."]
    k18 = k18.iloc[0]
    k18_mean = float(k18["peak_acc_pct_mean"])
    k18_std = float(k18.get("peak_acc_pct_std", 0.0))
    k18_collapse = int(k18.get("collapse_count", 0))
    competitors = fine_stats[fine_stats["jackpot"] != 18].copy()
    competitors["delta_vs_K18_pp"] = competitors["peak_acc_pct_mean"] - k18_mean
    beating = competitors[competitors["delta_vs_K18_pp"] >= margin].sort_values("peak_acc_pct_mean", ascending=False)
    if beating.empty:
        notes.append(f"No D3.1 K beats K=18 by the lock margin ({margin:.2f}pp).")
        return "H256_K18_LOCK", notes
    best = beating.iloc[0]
    best_k = int(best["jackpot"])
    best_std = float(best.get("peak_acc_pct_std", 0.0))
    best_collapse = int(best.get("collapse_count", 0))
    notes.append(f"K={best_k} beats K=18 by {float(best['delta_vs_K18_pp']):.2f}pp.")
    if best_collapse > k18_collapse or best_std > k18_std + variance_penalty:
        notes.append("The better-mean candidate also increases collapse count or variance materially.")
        return "H256_UNSTABLE_NO_LOCK", notes
    if best_k < 18:
        return "H256_FINE_LOW_REQUIRED", notes
    return "H256_FINE_HIGH_REQUIRED", notes


def plot_h256_curve(stats: pd.DataFrame, figures_dir: Path) -> None:
    h256 = stats[stats["H"] == 256].sort_values("jackpot")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.errorbar(
        h256["jackpot"],
        h256["peak_acc_pct_mean"],
        yerr=h256["peak_acc_pct_std"],
        marker="o",
        capsize=4,
    )
    ax.set_title("Phase D3.1 H=256 fine K sweep")
    ax.set_xlabel("K / jackpot candidates")
    ax.set_ylabel("peak accuracy (%)")
    ax.set_xticks(sorted(h256["jackpot"].astype(int).unique()))
    fig.tight_layout()
    fig.savefig(figures_dir / "phase_d3_fine_k_h256_curve.png", dpi=160)
    plt.close(fig)


def build_formula_table(all_construct: pd.DataFrame, all_results: pd.DataFrame, collapse_threshold_pp: float) -> pd.DataFrame:
    construct_metrics = ["C_K_window_ratio", "V_raw", "M_pos", "R_neg", "cost_eval_ms", "accepted_positive_steps"]
    result_metrics = ["peak_acc_pct", "final_acc_pct", "accept_rate_pct", "alive_frac_mean", "wall_clock_s"]
    result_stats = summarize(all_results, ["H", "jackpot"], result_metrics)
    result_stats = add_collapse_counts(result_stats, all_results, collapse_threshold_pp)
    construct_stats = summarize(all_construct, ["H", "jackpot"], construct_metrics)
    table = merge_construct_stats(result_stats, construct_stats)

    rows = []
    for h, group in table.groupby("H"):
        group = group.copy()
        k1 = group[group["jackpot"] == 1]
        p_pos = float(k1.iloc[0]["V_raw_mean"]) if not k1.empty and "V_raw_mean" in k1 else float("nan")
        best_peak = float(group["peak_acc_pct_mean"].max())
        for _, row in group.iterrows():
            k = int(row["jackpot"])
            predicted_hit = 1.0 - (1.0 - p_pos) ** k if np.isfinite(p_pos) else float("nan")
            rows.append({
                **row.to_dict(),
                "p_pos_from_K1": p_pos,
                "P_hit_model": predicted_hit,
                "near_best_0_5pp": bool(float(row["peak_acc_pct_mean"]) >= best_peak - 0.5),
            })
    return pd.DataFrame(rows).sort_values(["H", "jackpot"])


def write_fine_report(paths: Paths, validation: dict, fine_stats: pd.DataFrame, context_stats: pd.DataFrame, verdict: str, notes: list[str], lock_margin: float, expected_steps: int) -> None:
    lines = [
        "# Phase D3.1 Verdict: H=256 Fine K-Lock",
        "",
        "D3.1 resolves the H=256 K-axis region opened by D3 under strict SAF (`tau=0`, `s=0`).",
        "",
        "## Integrity",
        "",
        f"- D3.1 runs: `{validation['runs']}`",
        f"- D3.1 candidate rows: `{validation['candidate_rows']}`",
        f"- D3.1 checkpoints: `{validation['checkpoints']}`",
        f"- D3.1 panel summaries: `{validation['panel_summaries']}`",
        f"- D3.1 panel timeseries files: `{validation['panel_timeseries_files']}`",
        "",
        "## Verdict",
        "",
        f"- Result: **{verdict}**",
        f"- Lock margin: `{lock_margin:.2f}pp` mean peak over K=18",
    ]
    lines.extend(f"- {note}" for note in notes)
    lines.extend([
        "",
        "## Fine K Grid",
        "",
        "| K | n | peak mean | peak std | final mean | accept mean | collapse | C_K mean | wall/candidate ms |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in fine_stats.sort_values("jackpot").iterrows():
        wall_ms = float(row.get("wall_clock_s_mean", float("nan"))) * 1000.0 / (expected_steps * int(row["jackpot"]))
        lines.append(
            f"| {int(row['jackpot'])} | {int(row['n'])} | "
            f"{row.get('peak_acc_pct_mean', float('nan')):.2f} | "
            f"{row.get('peak_acc_pct_std', float('nan')):.2f} | "
            f"{row.get('final_acc_pct_mean', float('nan')):.2f} | "
            f"{row.get('accept_rate_pct_mean', float('nan')):.2f} | "
            f"{int(row.get('collapse_count', 0))} | "
            f"{row.get('C_K_window_ratio_mean', float('nan')):.3e} | "
            f"{wall_ms:.3f} |"
        )
    lines.extend([
        "",
        "## H=256 Context",
        "",
        "| source | K | n | peak mean | peak std | C_K mean |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for _, row in context_stats.sort_values(["source_rank", "jackpot"]).iterrows():
        lines.append(
            f"| {row['source']} | {int(row['jackpot'])} | {int(row['n'])} | "
            f"{row.get('peak_acc_pct_mean', float('nan')):.2f} | "
            f"{row.get('peak_acc_pct_std', float('nan')):.2f} | "
            f"{row.get('C_K_window_ratio_mean', float('nan')):.3e} |"
        )
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- Machine summary: `{paths.analysis_dir / 'phase_d3_fine_k_verdict.json'}`",
        f"- Fine stats: `{paths.analysis_dir / 'phase_d3_fine_k_stats.csv'}`",
        f"- Formula stats: `{paths.analysis_dir / 'saf_k_formula_stats.csv'}`",
        f"- Figure: `{paths.figures_dir / 'phase_d3_fine_k_h256_curve.png'}`",
    ])
    paths.fine_report.write_text("\n".join(lines))


def write_formula_report(paths: Paths, formula_stats: pd.DataFrame, fine_verdict: str, d3_winners: dict[int, int]) -> None:
    lock_rows = []
    for h, group in formula_stats.groupby("H"):
        if int(h) == 256 and fine_verdict in {"H256_FINE_HIGH_REQUIRED", "H256_FINE_LOW_REQUIRED", "H256_UNSTABLE_NO_LOCK"}:
            lock = 18
            status = "unresolved"
        elif int(h) == 256 and fine_verdict == "H256_K18_LOCK":
            lock = 18
            status = "locked"
        elif int(h) in d3_winners:
            lock = d3_winners[int(h)]
            status = "provisional_lock"
        else:
            group = group.sort_values("jackpot")
            near = group[group["near_best_0_5pp"]].copy()
            lock = int(near.iloc[0]["jackpot"]) if not near.empty else int(group.sort_values("peak_acc_pct_mean", ascending=False).iloc[0]["jackpot"])
            status = "provisional_lock"
        lock_rows.append((int(h), lock, status))

    lines = [
        "# SAF K Formula Lock",
        "",
        "This report keeps `tau=0` and `s=0` fixed and summarizes the empirical sampling aperture rule.",
        "",
        "The tested null model is:",
        "",
        "```text",
        "P_hit(K,H) = 1 - (1 - p_pos(H))^K",
        "```",
        "",
        "`P_hit` is diagnostic only. The lock rule also uses peak accuracy, C_K, variance, collapse count, and cost.",
        "",
        "## Provisional K(H)",
        "",
        "| H | K_lock | status |",
        "|---:|---:|---|",
    ]
    for h, lock, status in lock_rows:
        lines.append(f"| {h} | {lock} | {status} |")
    lines.extend([
        "",
        "The lock table uses the seed-matched D3 verdict for H=128/H=384 and the D3.1 fine verdict for H=256. The diagnostics table below merges broader context and is not used by itself as a winner table.",
        "",
        "## Formula Diagnostics",
        "",
        "| H | K | peak mean | C_K mean | V_raw | P_hit model | near best | collapse |",
        "|---:|---:|---:|---:|---:|---:|---|---:|",
    ])
    for _, row in formula_stats.sort_values(["H", "jackpot"]).iterrows():
        lines.append(
            f"| {int(row['H'])} | {int(row['jackpot'])} | "
            f"{row.get('peak_acc_pct_mean', float('nan')):.2f} | "
            f"{row.get('C_K_window_ratio_mean', float('nan')):.3e} | "
            f"{row.get('V_raw_mean', float('nan')):.4f} | "
            f"{row.get('P_hit_model', float('nan')):.3f} | "
            f"{str(bool(row.get('near_best_0_5pp', False))).lower()} | "
            f"{int(row.get('collapse_count', 0))} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `K` is not a universal constant in the current grower substrate.",
        "- `P_hit` explains the sampling funnel pressure, but it is not sufficient by itself: H=384 shows that larger K can increase variance/collapse risk.",
        "- The practical SAF lock remains empirical: choose the smallest near-best K after penalizing instability and cost.",
    ])
    paths.formula_report.write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    paths = ensure_dirs(args)

    d3f_results, d3f_construct = load_run_pair(paths.d3f_root, "D3.1")
    d3_results, d3_construct = load_run_pair(paths.d3_root, "D3")
    d2_results, d2_construct = load_run_pair(paths.d2_root, "D2")
    d1_results, d1_construct = load_run_pair(paths.d1_root, "D1")
    validation = validate_d3f(paths, d3f_results, d3f_construct, args.expected_H, args.expected_d3f_seeds)

    d3f_results = d3f_results[d3f_results["H"] == args.expected_H].copy()
    d3f_construct = d3f_construct[d3f_construct["H"] == args.expected_H].copy()
    result_metrics = ["peak_acc_pct", "final_acc_pct", "peak_final_gap_pp", "accept_rate_pct", "alive_frac_mean", "wall_clock_s"]
    construct_metrics = ["C_K_window_ratio", "cost_eval_ms", "V_raw", "M_pos", "R_neg", "accepted_positive_steps", "accepted_nonpositive_steps", "positive_delta_rows"]

    fine_stats = summarize(d3f_results, ["H", "jackpot"], result_metrics)
    fine_stats = add_collapse_counts(fine_stats, d3f_results, args.collapse_threshold_pp)
    fine_construct_stats = summarize(d3f_construct, ["H", "jackpot"], construct_metrics)
    fine_stats = merge_construct_stats(fine_stats, fine_construct_stats)
    verdict, notes = classify_fine(fine_stats, args.lock_margin_pp, args.variance_penalty_pp)

    context_results = pd.concat([
        d2_results[(d2_results["H"] == 256) & d2_results["jackpot"].isin([1, 3, 9])],
        d3_results[(d3_results["H"] == 256) & d3_results["jackpot"].isin([5, 13, 18])],
        d3f_results,
    ], ignore_index=True)
    context_construct = pd.concat([
        d2_construct[(d2_construct["H"] == 256) & d2_construct["jackpot"].isin([1, 3, 9])],
        d3_construct[(d3_construct["H"] == 256) & d3_construct["jackpot"].isin([5, 13, 18])],
        d3f_construct,
    ], ignore_index=True)
    context_stats = summarize(context_results, ["source", "H", "jackpot"], result_metrics)
    context_construct_stats = summarize(context_construct, ["source", "H", "jackpot"], construct_metrics)
    context_stats = context_stats.merge(
        context_construct_stats[["source", "H", "jackpot", "C_K_window_ratio_mean"]],
        on=["source", "H", "jackpot"],
        how="left",
    )
    source_rank = {"D2": 0, "D3": 1, "D3.1": 2}
    context_stats["source_rank"] = context_stats["source"].map(source_rank).fillna(9)

    d3_formula_results = d3_results[~((d3_results["H"] == 256) & (d3_results["jackpot"] == 18))]
    d3_formula_construct = d3_construct[~((d3_construct["H"] == 256) & (d3_construct["jackpot"] == 18))]
    formula_results = pd.concat([
        d1_results[d1_results["jackpot"].isin([1, 3, 9])],
        d2_results[d2_results["jackpot"].isin([1, 3, 9])],
        d3_formula_results[d3_formula_results["jackpot"].isin([5, 13, 18])],
        d3f_results,
    ], ignore_index=True)
    formula_construct = pd.concat([
        d1_construct[d1_construct["jackpot"].isin([1, 3, 9])],
        d2_construct[d2_construct["jackpot"].isin([1, 3, 9])],
        d3_formula_construct[d3_formula_construct["jackpot"].isin([5, 13, 18])],
        d3f_construct,
    ], ignore_index=True)
    formula_stats = build_formula_table(formula_construct, formula_results, args.collapse_threshold_pp)
    d3_winners = {}
    d3_verdict_path = paths.d3_root / "analysis" / "phase_d3_klock_verdict.json"
    if d3_verdict_path.exists():
        d3_payload = json.loads(d3_verdict_path.read_text())
        d3_winners = {int(row["H"]): int(row["best_K"]) for row in d3_payload.get("winners", [])}

    plot_h256_curve(fine_stats, paths.figures_dir)
    fine_stats.to_csv(paths.analysis_dir / "phase_d3_fine_k_stats.csv", index=False)
    context_stats.to_csv(paths.analysis_dir / "phase_d3_fine_k_context_stats.csv", index=False)
    formula_stats.to_csv(paths.analysis_dir / "saf_k_formula_stats.csv", index=False)

    payload = {
        "d3f_root": str(paths.d3f_root),
        "d3_root": str(paths.d3_root),
        "d2_root": str(paths.d2_root),
        "validation": validation,
        "lock_margin_pp": args.lock_margin_pp,
        "collapse_threshold_pp": args.collapse_threshold_pp,
        "verdict": verdict,
        "notes": notes,
        "fine_stats": json.loads(fine_stats.to_json(orient="records")),
        "context_stats": json.loads(context_stats.to_json(orient="records")),
        "formula_stats": json.loads(formula_stats.to_json(orient="records")),
    }
    (paths.analysis_dir / "phase_d3_fine_k_verdict.json").write_text(json.dumps(payload, indent=2))
    write_fine_report(paths, validation, fine_stats, context_stats, verdict, notes, args.lock_margin_pp, args.expected_d3f_steps)
    write_formula_report(paths, formula_stats, verdict, d3_winners)

    print(json.dumps({
        "status": "PASS",
        "verdict": verdict,
        "runs": validation["runs"],
        "candidate_rows": validation["candidate_rows"],
        "report": str(paths.fine_report),
        "formula_report": str(paths.formula_report),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

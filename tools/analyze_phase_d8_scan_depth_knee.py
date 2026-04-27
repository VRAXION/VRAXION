"""Phase D8.1.1: deterministic scan-depth knee audit.

Offline-only audit for the question:
How many samples per spherical archive cell are needed before the cell's
future_gain estimate becomes stable enough to use as a confidence signal?

This is intentionally narrower than D8.1 cell coherence: it does not prove
that the cell map is predictive. It only estimates the scan depth needed to
stabilize the empirical mean of an already-defined cell. Cells below the
threshold remain low-confidence.

This does not launch Rust, does not read raw candidates.csv, and does not
change live search behavior.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_INPUT = Path("output/phase_d8_archive_psi_replay_20260427/analysis/panel_state_dataset.csv")
DEFAULT_OUT = Path("output/phase_d8_scan_depth_knee_20260427")
DEFAULT_REPORT = Path("docs/research/PHASE_D8_SCAN_DEPTH_KNEE_AUDIT.md")
CORE_FEATURES = [
    "stable_rank",
    "kernel_rank",
    "separation_sp",
    "collision_rate",
    "f_active",
    "unique_predictions",
    "edges",
    "accept_rate_window",
]
DEFAULT_ANCHORS = [16, 32, 64, 128]
DEFAULT_ANCHOR_SEEDS = [11, 23, 37]
DEFAULT_SAMPLE_SIZES = [1, 2, 3, 5, 8, 13, 21, 34, 55]
EPS = 1e-12


def parse_int_list(value: str, default: list[int]) -> list[int]:
    if value is None or value == "":
        return default
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--anchor-counts", type=str, default=",".join(str(x) for x in DEFAULT_ANCHORS))
    parser.add_argument("--anchor-seeds", type=str, default=",".join(str(x) for x in DEFAULT_ANCHOR_SEEDS))
    parser.add_argument("--sample-sizes", type=str, default=",".join(str(x) for x in DEFAULT_SAMPLE_SIZES))
    parser.add_argument("--bootstrap-iters", type=int, default=200)
    parser.add_argument("--basin-delta", type=float, default=0.005)
    parser.add_argument("--rank-threshold", type=float, default=0.70)
    parser.add_argument("--top-overlap-threshold", type=float, default=0.50)
    parser.add_argument("--relative-error-threshold", type=float, default=0.50)
    parser.add_argument("--min-eligible-cells", type=int, default=8)
    return parser.parse_args()


def json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_ready(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_ready(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.std(a[ok]) <= EPS or np.std(b[ok]) <= EPS:
        return 0.0
    val = stats.spearmanr(a[ok], b[ok], nan_policy="omit").correlation
    return float(0.0 if not np.isfinite(val) else val)


def load_dataset(path: Path, max_rows: int | None, basin_delta: float) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing panel dataset: {path}")
    df = pd.read_csv(path)
    required = {"H", "future_gain_final"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"missing required columns: {missing}")
    if max_rows is not None:
        df = df.sort_values(["H", "source", "run_id", "panel_index"], na_position="last").head(max_rows).copy()
    df["H"] = pd.to_numeric(df["H"], errors="coerce").astype("Int64")
    df = df[df["H"].notna()].copy()
    df["H"] = df["H"].astype(int)
    df["future_gain_final"] = pd.to_numeric(df["future_gain_final"], errors="coerce").fillna(0.0)
    if "basin_hit" in df:
        df["basin_hit"] = pd.to_numeric(df["basin_hit"], errors="coerce").fillna(0).astype(int)
    else:
        df["basin_hit"] = (df["future_gain_final"] >= basin_delta).astype(int)
    return df.reset_index(drop=True)


def robust_sphere_coords(sub: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = [c for c in CORE_FEATURES if c in sub and pd.to_numeric(sub[c], errors="coerce").notna().sum() > 2]
    if not cols:
        return np.empty((len(sub), 0)), []
    x = sub[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    med = np.nanmedian(x, axis=0)
    inds = np.where(np.isnan(x))
    x[inds] = np.take(med, inds[1])
    q75 = np.nanpercentile(x, 75, axis=0)
    q25 = np.nanpercentile(x, 25, axis=0)
    scale = q75 - q25
    scale[scale <= EPS] = 1.0
    z = (x - med) / scale
    norm = np.linalg.norm(z, axis=1)
    norm[norm <= EPS] = 1.0
    return z / norm[:, None], cols


def deterministic_anchors(dim: int, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    anchors = rng.normal(size=(count, dim))
    norm = np.linalg.norm(anchors, axis=1)
    norm[norm <= EPS] = 1.0
    return anchors / norm[:, None]


def assign_cells(coords: np.ndarray, anchor_count: int, anchor_seed: int) -> np.ndarray:
    if coords.shape[1] == 0:
        return np.zeros(coords.shape[0], dtype=int)
    anchors = deterministic_anchors(coords.shape[1], anchor_count, anchor_seed)
    return np.argmax(coords @ anchors.T, axis=1).astype(int)


def top_overlap(true_means: np.ndarray, est_means: np.ndarray) -> float:
    n = len(true_means)
    if n < 5:
        return math.nan
    k = max(1, int(math.ceil(0.20 * n)))
    true_top = set(np.argsort(true_means)[-k:].tolist())
    est_top = set(np.argsort(est_means)[-k:].tolist())
    return float(len(true_top & est_top) / k)


def scan_one_config(
    df: pd.DataFrame,
    h: int,
    anchor_count: int,
    anchor_seed: int,
    sample_sizes: list[int],
    bootstrap_iters: int,
) -> tuple[list[dict], pd.DataFrame]:
    sub = df[df["H"] == h].copy().reset_index(drop=True)
    coords, used_features = robust_sphere_coords(sub)
    cells = assign_cells(coords, anchor_count, anchor_seed)
    sub["cell_id"] = cells
    cell_groups = {
        int(cell): group.index.to_numpy(dtype=int)
        for cell, group in sub.groupby("cell_id", dropna=False)
    }
    target = sub["future_gain_final"].to_numpy(float)
    basin = sub["basin_hit"].to_numpy(int)
    cell_rows = []
    for cell, idx in cell_groups.items():
        cell_rows.append({
            "H": h,
            "anchor_count": anchor_count,
            "anchor_seed": anchor_seed,
            "cell_id": cell,
            "n": int(len(idx)),
            "true_mean_future_gain": float(np.mean(target[idx])),
            "true_basin_rate": float(np.mean(basin[idx])),
        })

    rows = []
    rng_base = 100000 + h * 10 + anchor_count * 1000 + anchor_seed
    for n_sample in sample_sizes:
        eligible = {cell: idx for cell, idx in cell_groups.items() if len(idx) >= n_sample}
        if len(eligible) < 2:
            rows.append(empty_metric_row(h, anchor_count, anchor_seed, n_sample, len(cell_groups), len(eligible), used_features))
            continue
        true_cells = np.array(sorted(eligible.keys()), dtype=int)
        true_mean = np.array([float(np.mean(target[eligible[cell]])) for cell in true_cells])
        true_basin = np.array([float(np.mean(basin[eligible[cell]])) for cell in true_cells])
        errors = []
        basin_errors = []
        ranks = []
        overlaps = []
        for b in range(bootstrap_iters):
            rng = np.random.default_rng(rng_base + n_sample * 10000 + b)
            est_mean = []
            est_basin = []
            for cell in true_cells:
                idx = eligible[int(cell)]
                sample_idx = rng.choice(idx, size=n_sample, replace=False)
                est_mean.append(float(np.mean(target[sample_idx])))
                est_basin.append(float(np.mean(basin[sample_idx])))
            est_mean_arr = np.array(est_mean)
            est_basin_arr = np.array(est_basin)
            errors.append(float(np.mean(np.abs(est_mean_arr - true_mean))))
            basin_errors.append(float(np.mean(np.abs(est_basin_arr - true_basin))))
            ranks.append(safe_spearman(true_mean, est_mean_arr))
            overlaps.append(top_overlap(true_mean, est_mean_arr))
        n1_error = float(np.mean(errors)) if n_sample == 1 else math.nan
        rows.append({
            "H": h,
            "anchor_count": anchor_count,
            "anchor_seed": anchor_seed,
            "sample_n": n_sample,
            "used_features": ",".join(used_features),
            "total_cells": int(len(cell_groups)),
            "eligible_cells": int(len(eligible)),
            "eligible_cell_rate": float(len(eligible) / max(len(cell_groups), 1)),
            "median_cell_count": float(np.median([len(idx) for idx in cell_groups.values()])),
            "mean_abs_cell_mean_error": float(np.mean(errors)),
            "std_abs_cell_mean_error": float(np.std(errors)),
            "mean_abs_basin_rate_error": float(np.mean(basin_errors)),
            "cell_rank_spearman": float(np.mean(ranks)),
            "top20_cell_overlap": float(np.nanmean(overlaps)),
            "n1_error_reference": n1_error,
        })
    return rows, pd.DataFrame(cell_rows)


def empty_metric_row(h: int, anchor_count: int, anchor_seed: int, n_sample: int, total_cells: int, eligible_cells: int, used_features: list[str]) -> dict:
    return {
        "H": h,
        "anchor_count": anchor_count,
        "anchor_seed": anchor_seed,
        "sample_n": n_sample,
        "used_features": ",".join(used_features),
        "total_cells": int(total_cells),
        "eligible_cells": int(eligible_cells),
        "eligible_cell_rate": float(eligible_cells / max(total_cells, 1)),
        "median_cell_count": math.nan,
        "mean_abs_cell_mean_error": math.nan,
        "std_abs_cell_mean_error": math.nan,
        "mean_abs_basin_rate_error": math.nan,
        "cell_rank_spearman": math.nan,
        "top20_cell_overlap": math.nan,
        "n1_error_reference": math.nan,
    }


def run_scan(df: pd.DataFrame, anchor_counts: list[int], anchor_seeds: list[int], sample_sizes: list[int], bootstrap_iters: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    occupancy = []
    for h in sorted(df["H"].dropna().unique().tolist()):
        for anchor_count in anchor_counts:
            for anchor_seed in anchor_seeds:
                rows, cells = scan_one_config(df, int(h), anchor_count, anchor_seed, sample_sizes, bootstrap_iters)
                metric_rows.extend(rows)
                occupancy.append(cells)
    metrics = pd.DataFrame(metric_rows)
    occupancy_df = pd.concat(occupancy, ignore_index=True) if occupancy else pd.DataFrame()
    metrics["n1_error_reference"] = metrics.groupby(["H", "anchor_count", "anchor_seed"])["mean_abs_cell_mean_error"].transform("first")
    metrics["relative_error_vs_n1"] = metrics["mean_abs_cell_mean_error"] / metrics["n1_error_reference"].replace(0, np.nan)
    return metrics, occupancy_df


def find_knees(metrics: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for (h, anchor_count, anchor_seed), sub in metrics.groupby(["H", "anchor_count", "anchor_seed"], dropna=False):
        sub = sub.sort_values("sample_n")
        knee = None
        for row in sub.to_dict("records"):
            if not np.isfinite(row.get("mean_abs_cell_mean_error", math.nan)):
                continue
            ok = (
                row["eligible_cells"] >= args.min_eligible_cells
                and row["cell_rank_spearman"] >= args.rank_threshold
                and row["top20_cell_overlap"] >= args.top_overlap_threshold
                and row["relative_error_vs_n1"] <= args.relative_error_threshold
            )
            if ok:
                knee = int(row["sample_n"])
                break
        rows.append({
            "H": int(h),
            "anchor_count": int(anchor_count),
            "anchor_seed": int(anchor_seed),
            "knee_sample_n": knee,
            "knee_found": knee is not None,
            "max_rank_spearman": float(sub["cell_rank_spearman"].max(skipna=True)),
            "max_top20_overlap": float(sub["top20_cell_overlap"].max(skipna=True)),
            "min_relative_error_vs_n1": float(sub["relative_error_vs_n1"].min(skipna=True)),
            "max_feasible_sample_n": int(sub[sub["eligible_cells"] >= args.min_eligible_cells]["sample_n"].max()) if not sub[sub["eligible_cells"] >= args.min_eligible_cells].empty else 0,
        })
    return pd.DataFrame(rows)


def decide(knees: pd.DataFrame, metrics: pd.DataFrame) -> tuple[str, dict]:
    by_h = {}
    found_h = 0
    weak_h = 0
    data_fail_h = 0
    for h, sub in knees.groupby("H"):
        configs = len(sub)
        found_rate = float(sub["knee_found"].mean()) if configs else 0.0
        median_knee = float(sub["knee_sample_n"].dropna().median()) if sub["knee_sample_n"].notna().any() else math.nan
        max_feasible = float(sub["max_feasible_sample_n"].median()) if configs else 0.0
        best_rank = float(sub["max_rank_spearman"].max(skipna=True)) if configs else math.nan
        best_overlap = float(sub["max_top20_overlap"].max(skipna=True)) if configs else math.nan
        h_found = found_rate >= 2 / 3
        h_weak = (found_rate > 0.0) or (best_rank >= 0.50 and best_overlap >= 0.35)
        h_data_fail = max_feasible < 5
        found_h += int(h_found)
        weak_h += int(h_weak)
        data_fail_h += int(h_data_fail)
        by_h[str(int(h))] = {
            "configs": configs,
            "found_rate": found_rate,
            "median_knee_sample_n": median_knee,
            "median_max_feasible_sample_n": max_feasible,
            "best_rank_spearman": best_rank,
            "best_top20_overlap": best_overlap,
            "h_found": h_found,
            "h_weak": h_weak,
            "h_data_fail": h_data_fail,
        }
    total_h = knees["H"].nunique()
    required_h = math.ceil((2 / 3) * total_h) if total_h else 999
    if data_fail_h >= required_h:
        verdict = "CELL_SCAN_NEEDS_MORE_DATA"
    elif found_h >= required_h:
        verdict = "CELL_SCAN_KNEE_FOUND"
    elif weak_h >= required_h:
        verdict = "CELL_SCAN_WEAK_KNEE"
    else:
        verdict = "CELL_SCAN_TOO_NOISY"
    decision = {
        "required_h": required_h,
        "found_h": found_h,
        "weak_h": weak_h,
        "data_fail_h": data_fail_h,
        "by_h": by_h,
    }
    return verdict, decision


def plot_outputs(out: Path, metrics: pd.DataFrame, knees: pd.DataFrame) -> None:
    fig_dir = out / "analysis" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if metrics.empty:
        return
    agg = metrics.groupby(["H", "sample_n"], dropna=False).agg(
        rank=("cell_rank_spearman", "mean"),
        error=("mean_abs_cell_mean_error", "mean"),
        overlap=("top20_cell_overlap", "mean"),
    ).reset_index()
    for value, ylabel, name in [
        ("rank", "cell rank Spearman higher is better", "rank_vs_samples.png"),
        ("error", "mean abs cell mean error lower is better", "error_vs_samples.png"),
        ("overlap", "top-20% cell overlap higher is better", "top_overlap_vs_samples.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4))
        for h, sub in agg.groupby("H"):
            ax.plot(sub["sample_n"], sub[value], marker="o", label=f"H={int(h)}")
        ax.set_xscale("log")
        ax.set_xlabel("samples per cell")
        ax.set_ylabel(ylabel)
        ax.legend()
        plt.tight_layout()
        fig.savefig(fig_dir / name, dpi=160)
        plt.close(fig)


def write_report(path: Path, verdict: str, decision: dict, coverage: dict, metrics: pd.DataFrame, knees: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    knee_view = knees.sort_values(["H", "anchor_count", "anchor_seed"])
    metric_view = (
        metrics.groupby(["H", "sample_n"], dropna=False)
        .agg(
            mean_rank=("cell_rank_spearman", "mean"),
            mean_error=("mean_abs_cell_mean_error", "mean"),
            mean_overlap=("top20_cell_overlap", "mean"),
            mean_eligible_cells=("eligible_cells", "mean"),
        )
        .reset_index()
    )
    lines = [
        "# Phase D8.1.1 Scan-Depth Knee Audit",
        "",
        f"Verdict: **{verdict}**",
        "",
        "## Summary",
        "",
        "- D8.1.1 is offline-only and estimates how many samples per spherical cell are needed for reliable cell-level future_gain estimates.",
        "- It uses deterministic bootstrap subsampling within each cell and never reads raw candidates or launches live Rust runs.",
        "- A knee requires rank stability, top-cell overlap, and a large error drop from the one-sample estimate.",
        "- This does not prove that spherical cells are predictive by themselves; it only estimates when an already-defined cell has been sampled deeply enough to trust its empirical mean.",
        "- Knee thresholds are occupancy-conditional: cells with fewer observations than the knee remain low-confidence and should not drive a pointer alone.",
        "",
        "## Coverage",
        "",
        "```json",
        json.dumps(json_ready(coverage), indent=2),
        "```",
        "",
        "## Decision",
        "",
        "```json",
        json.dumps(json_ready(decision), indent=2),
        "```",
        "",
        "## Knee By Config",
        "",
        "```text",
        knee_view.to_string(index=False, max_rows=120),
        "```",
        "",
        "## Aggregate Scan Curve",
        "",
        "```text",
        metric_view.to_string(index=False, max_rows=120),
        "```",
        "",
        "## Interpretation",
        "",
    ]
    if verdict == "CELL_SCAN_KNEE_FOUND":
        lines.append("- A practical scan-depth knee exists for most H regimes; use it as a confidence gate in future pointer replay, not as a live pointer proof.")
    elif verdict == "CELL_SCAN_WEAK_KNEE":
        lines.append("- Scan depth improves cell estimates, but the knee is not strong enough to lock a universal per-cell sample count.")
    elif verdict == "CELL_SCAN_NEEDS_MORE_DATA":
        lines.append("- Current cells do not contain enough samples to estimate a knee reliably at the requested anchor densities.")
    else:
        lines.append("- Cell estimates remain too noisy even with deeper sampling; improve `phi` or cellization before pointer work.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    anchor_counts = parse_int_list(args.anchor_counts, DEFAULT_ANCHORS)
    anchor_seeds = parse_int_list(args.anchor_seeds, DEFAULT_ANCHOR_SEEDS)
    sample_sizes = parse_int_list(args.sample_sizes, DEFAULT_SAMPLE_SIZES)
    analysis_dir = args.out / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(args.input, args.max_rows, args.basin_delta)
    metrics, occupancy = run_scan(df, anchor_counts, anchor_seeds, sample_sizes, args.bootstrap_iters)
    knees = find_knees(metrics, args)
    verdict, decision = decide(knees, metrics)
    coverage = {
        "input": str(args.input),
        "rows": int(len(df)),
        "H_values": sorted([int(x) for x in df["H"].dropna().unique().tolist()]),
        "anchor_counts": anchor_counts,
        "anchor_seeds": anchor_seeds,
        "sample_sizes": sample_sizes,
        "bootstrap_iters": args.bootstrap_iters,
        "features": CORE_FEATURES,
        "rank_threshold": args.rank_threshold,
        "top_overlap_threshold": args.top_overlap_threshold,
        "relative_error_threshold": args.relative_error_threshold,
        "min_eligible_cells": args.min_eligible_cells,
    }
    summary = {"verdict": verdict, "coverage": coverage, "decision": decision}
    (analysis_dir / "summary.json").write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")
    metrics.to_csv(analysis_dir / "scan_depth_summary.csv", index=False)
    knees.to_csv(analysis_dir / "knee_by_config.csv", index=False)
    occupancy.to_csv(analysis_dir / "cell_occupancy.csv", index=False)
    plot_outputs(args.out, metrics, knees)
    write_report(args.report, verdict, decision, coverage, metrics, knees)
    print(f"Verdict: {verdict}")
    print(f"Rows: {len(df)}")
    print(f"Wrote: {analysis_dir / 'summary.json'}")
    print(f"Wrote: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Phase D8.1: Spherical cell coherence audit.

Offline-only test for whether hypersphere archive cells are meaningful.
It asks whether same-cell neighbors predict future_gain better than
global, time-bucket, and random-cell baselines.
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
DEFAULT_OUT = Path("output/phase_d8_cell_coherence_20260427")
DEFAULT_REPORT = Path("docs/research/PHASE_D8_CELL_COHERENCE_AUDIT.md")

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
PLUS_FEATURES = CORE_FEATURES + ["main_peak_acc", "panel_probe_acc", "time_pct"]
DEFAULT_ANCHORS = [16, 32, 64, 128]
DEFAULT_SEEDS = [11, 23, 37]
EPS = 1e-12


def json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_ready(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_ready(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


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
    parser.add_argument("--anchor-seeds", type=str, default=",".join(str(x) for x in DEFAULT_SEEDS))
    parser.add_argument("--time-buckets", type=int, default=10)
    parser.add_argument("--singleton-max-rate", type=float, default=0.60)
    parser.add_argument("--min-nonempty-rate", type=float, default=0.25)
    parser.add_argument("--basin-delta", type=float, default=0.005)
    return parser.parse_args()


def safe_spearman(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5 or np.std(x[ok]) <= EPS or np.std(y[ok]) <= EPS:
        return 0.0
    val = stats.spearmanr(x[ok], y[ok], nan_policy="omit").correlation
    return float(0.0 if not np.isfinite(val) else val)


def mae(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() == 0:
        return math.nan
    return float(np.mean(np.abs(x[ok] - y[ok])))


def load_dataset(path: Path, max_rows: int | None, basin_delta: float, time_buckets: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing D8 panel dataset: {path}")
    df = pd.read_csv(path)
    required = {"H", "future_gain_final", "time_pct"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"missing required columns in panel dataset: {missing}")
    if max_rows is not None:
        df = df.sort_values(["H", "source", "run_id", "panel_index"], na_position="last").head(max_rows).copy()
    df["H"] = pd.to_numeric(df["H"], errors="coerce").astype("Int64")
    df = df[df["H"].notna()].copy()
    df["H"] = df["H"].astype(int)
    df["future_gain_final"] = pd.to_numeric(df["future_gain_final"], errors="coerce").fillna(0.0)
    if "basin_hit" not in df:
        df["basin_hit"] = (df["future_gain_final"] >= basin_delta).astype(int)
    else:
        df["basin_hit"] = pd.to_numeric(df["basin_hit"], errors="coerce").fillna(0).astype(int)
    time_pct = pd.to_numeric(df["time_pct"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    df["time_bucket"] = np.floor(np.minimum(time_pct * time_buckets, time_buckets - 1)).astype(int)
    return df.reset_index(drop=True)


def robust_sphere_coords(sub: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, list[str]]:
    cols = [c for c in features if c in sub and pd.to_numeric(sub[c], errors="coerce").notna().sum() > 2]
    if not cols:
        return np.empty((len(sub), 0)), []
    x = sub[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    med = np.nanmedian(x, axis=0)
    inds = np.where(np.isnan(x))
    x[inds] = np.take(med, inds[1])
    q75 = np.nanpercentile(x, 75, axis=0)
    q25 = np.nanpercentile(x, 25, axis=0)
    iqr = q75 - q25
    iqr[iqr <= EPS] = 1.0
    z = (x - med) / iqr
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
    sim = coords @ anchors.T
    return np.argmax(sim, axis=1).astype(int)


def leave_one_out_cell_mean(target: np.ndarray, cells: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    pred = np.asarray(fallback, dtype=float).copy()
    df = pd.DataFrame({"target": target, "cell": cells})
    sums = df.groupby("cell")["target"].transform("sum").to_numpy(float)
    counts = df.groupby("cell")["target"].transform("count").to_numpy(float)
    ok = counts > 1
    pred[ok] = (sums[ok] - target[ok]) / (counts[ok] - 1.0)
    return pred


def random_cell_control(sub: pd.DataFrame, cells: np.ndarray, anchor_seed: int) -> np.ndarray:
    rng = np.random.default_rng(9000 + anchor_seed)
    shuffled = cells.copy()
    for _, idx in sub.groupby(["H", "time_bucket"], dropna=False).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        vals = shuffled[idx_arr].copy()
        rng.shuffle(vals)
        shuffled[idx_arr] = vals
    return shuffled


def baseline_predictions(sub: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    target = sub["future_gain_final"].to_numpy(float)
    global_mean = np.full(len(sub), float(np.mean(target)))
    time_mean = (
        sub.groupby(["H", "time_bucket"], dropna=False)["future_gain_final"]
        .transform("mean")
        .to_numpy(float)
    )
    return global_mean, time_mean


def variance_ratio(target: np.ndarray, cells: np.ndarray) -> float:
    total = float(np.var(target))
    if total <= EPS:
        return 0.0
    means = pd.DataFrame({"target": target, "cell": cells}).groupby("cell")["target"].transform("mean").to_numpy(float)
    return float(np.var(means) / total)


def enrichment(sub: pd.DataFrame, cells: np.ndarray) -> tuple[float, float, int]:
    work = sub.copy()
    work["cell"] = cells
    cell_stats = (
        work.groupby("cell", dropna=False)
        .agg(n=("future_gain_final", "size"), mean_future_gain=("future_gain_final", "mean"), basin_rate=("basin_hit", "mean"))
        .reset_index()
    )
    if cell_stats.empty:
        return math.nan, math.nan, 0
    cutoff = cell_stats["mean_future_gain"].quantile(0.80)
    top = cell_stats[cell_stats["mean_future_gain"] >= cutoff]
    global_basin = float(work["basin_hit"].mean())
    top_basin = float(np.average(top["basin_rate"], weights=top["n"])) if not top.empty else math.nan
    return top_basin, global_basin, int(len(top))


def evaluate_one(
    df: pd.DataFrame,
    h: int,
    feature_set: str,
    features: list[str],
    anchor_count: int,
    anchor_seed: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    sub = df[df["H"] == h].copy().reset_index(drop=True)
    coords, used_features = robust_sphere_coords(sub, features)
    cells = assign_cells(coords, anchor_count, anchor_seed)
    target = sub["future_gain_final"].to_numpy(float)
    global_mean, time_mean = baseline_predictions(sub)
    same_pred = leave_one_out_cell_mean(target, cells, time_mean)
    random_cells = random_cell_control(sub, cells, anchor_seed)
    random_pred = leave_one_out_cell_mean(target, random_cells, time_mean)
    top_basin, global_basin, top_cells = enrichment(sub, cells)

    counts = pd.Series(cells).value_counts()
    nonempty = int(len(counts))
    singleton = int((counts == 1).sum())
    row = {
        "H": h,
        "feature_set": feature_set,
        "anchor_count": anchor_count,
        "anchor_seed": anchor_seed,
        "n": int(len(sub)),
        "used_features": ",".join(used_features),
        "nonempty_cells": nonempty,
        "empty_cells": int(anchor_count - nonempty),
        "nonempty_rate": float(nonempty / anchor_count),
        "singleton_cells": singleton,
        "singleton_rate": float(singleton / max(nonempty, 1)),
        "median_count_nonempty": float(counts.median()) if nonempty else 0.0,
        "same_cell_mae": mae(target, same_pred),
        "global_mae": mae(target, global_mean),
        "time_bucket_mae": mae(target, time_mean),
        "random_cell_mae": mae(target, random_pred),
        "same_cell_spearman": safe_spearman(target, same_pred),
        "global_spearman": safe_spearman(target, global_mean),
        "time_bucket_spearman": safe_spearman(target, time_mean),
        "random_cell_spearman": safe_spearman(target, random_pred),
        "between_cell_variance_ratio": variance_ratio(target, cells),
        "top_cell_basin_rate": top_basin,
        "global_basin_rate": global_basin,
        "top_cell_basin_lift": float(top_basin / global_basin) if global_basin > EPS and np.isfinite(top_basin) else math.nan,
        "top_cells": top_cells,
    }
    row["beats_time_mae"] = bool(row["same_cell_mae"] < row["time_bucket_mae"])
    row["beats_random_mae"] = bool(row["same_cell_mae"] < row["random_cell_mae"])
    row["beats_time_spearman"] = bool(row["same_cell_spearman"] > row["time_bucket_spearman"])
    row["beats_random_spearman"] = bool(row["same_cell_spearman"] > row["random_cell_spearman"])
    row["positive_enrichment"] = bool(np.isfinite(row["top_cell_basin_lift"]) and row["top_cell_basin_lift"] > 1.0)

    pred = sub[["state_id", "H", "source", "run_id", "panel_index", "time_bucket", "future_gain_final", "basin_hit"]].copy()
    pred["feature_set"] = feature_set
    pred["anchor_count"] = anchor_count
    pred["anchor_seed"] = anchor_seed
    pred["cell_id"] = cells
    pred["random_cell_id"] = random_cells
    pred["same_cell_pred"] = same_pred
    pred["time_bucket_pred"] = time_mean
    pred["global_pred"] = global_mean
    pred["random_cell_pred"] = random_pred

    cell_df = sub[["H", "future_gain_final", "basin_hit"]].copy()
    cell_df["feature_set"] = feature_set
    cell_df["anchor_count"] = anchor_count
    cell_df["anchor_seed"] = anchor_seed
    cell_df["cell_id"] = cells
    cell_summary = (
        cell_df.groupby(["H", "feature_set", "anchor_count", "anchor_seed", "cell_id"], dropna=False)
        .agg(n=("future_gain_final", "size"), mean_future_gain=("future_gain_final", "mean"), basin_rate=("basin_hit", "mean"))
        .reset_index()
    )
    return row, pred, cell_summary


def run_audit(df: pd.DataFrame, anchor_counts: list[int], anchor_seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    preds = []
    cells = []
    feature_sets = {
        "behavior_core_no_score_time": CORE_FEATURES,
        "behavior_plus_score_time": PLUS_FEATURES,
    }
    for feature_set, features in feature_sets.items():
        for h in sorted(df["H"].dropna().unique().tolist()):
            for count in anchor_counts:
                for seed in anchor_seeds:
                    row, pred, cell_summary = evaluate_one(df, int(h), feature_set, features, count, seed)
                    rows.append(row)
                    preds.append(pred)
                    cells.append(cell_summary)
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True), pd.concat(cells, ignore_index=True)


def decide(summary: pd.DataFrame, singleton_max_rate: float, min_nonempty_rate: float) -> tuple[str, dict]:
    core = summary[summary["feature_set"] == "behavior_core_no_score_time"].copy()
    plus = summary[summary["feature_set"] == "behavior_plus_score_time"].copy()
    if core.empty:
        return "CELL_MAP_NEEDS_MORE_DATA", {"reason": "no_core_rows"}

    def aggregate(work: pd.DataFrame) -> dict:
        by_h = {}
        pass_h = 0
        weak_h = 0
        data_fail_h = 0
        for h, sub in work.groupby("H"):
            per_count = []
            for count, csub in sub.groupby("anchor_count"):
                reliable = (
                    (csub["singleton_rate"].median() <= singleton_max_rate)
                    and (csub["nonempty_rate"].median() >= min_nonempty_rate)
                )
                strong = (
                    reliable
                    and (csub["beats_time_mae"].mean() >= 2 / 3)
                    and (csub["beats_random_mae"].mean() >= 2 / 3)
                    and (csub["positive_enrichment"].mean() >= 2 / 3)
                )
                weak = reliable and (csub["beats_random_mae"].mean() >= 2 / 3)
                per_count.append({"anchor_count": int(count), "reliable": reliable, "strong": strong, "weak": weak})
            strong_counts = sum(1 for x in per_count if x["strong"])
            weak_counts = sum(1 for x in per_count if x["weak"])
            reliable_counts = sum(1 for x in per_count if x["reliable"])
            required = math.ceil((2 / 3) * len(per_count)) if per_count else 999
            h_strong = strong_counts >= required
            h_weak = weak_counts >= required
            h_data_fail = reliable_counts < required
            pass_h += int(h_strong)
            weak_h += int(h_weak)
            data_fail_h += int(h_data_fail)
            by_h[str(int(h))] = {
                "strong_counts": strong_counts,
                "weak_counts": weak_counts,
                "reliable_counts": reliable_counts,
                "required_counts": required,
                "anchor_results": per_count,
                "median_same_mae": float(sub["same_cell_mae"].median()),
                "median_time_mae": float(sub["time_bucket_mae"].median()),
                "median_random_mae": float(sub["random_cell_mae"].median()),
                "median_variance_ratio": float(sub["between_cell_variance_ratio"].median()),
                "median_basin_lift": float(sub["top_cell_basin_lift"].replace([np.inf, -np.inf], np.nan).median()),
            }
        return {"pass_h": pass_h, "weak_h": weak_h, "data_fail_h": data_fail_h, "by_h": by_h}

    core_agg = aggregate(core)
    plus_agg = aggregate(plus)
    h_total = int(core["H"].nunique())
    required_h = math.ceil((2 / 3) * h_total) if h_total else 999
    if core_agg["data_fail_h"] >= required_h:
        verdict = "CELL_MAP_NEEDS_MORE_DATA"
    elif core_agg["pass_h"] >= required_h:
        verdict = "CELL_MAP_VALID"
    elif core_agg["weak_h"] >= 1 and plus_agg["pass_h"] >= required_h:
        verdict = "CELL_MAP_TIME_CONFOUNDED"
    elif core_agg["weak_h"] >= 1:
        verdict = "CELL_MAP_WEAK"
    else:
        verdict = "CELL_MAP_INVALID"
    return verdict, {"required_h": required_h, "core": core_agg, "plus": plus_agg}


def plot_outputs(out: Path, summary: pd.DataFrame) -> None:
    fig_dir = out / "analysis" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    core = summary[summary["feature_set"] == "behavior_core_no_score_time"].copy()
    if core.empty:
        return
    pivot = core.groupby(["anchor_count", "H"])["between_cell_variance_ratio"].median().unstack("H")
    ax = pivot.plot(marker="o", figsize=(8, 4))
    ax.set_ylabel("between-cell / total variance")
    ax.set_title("Cell coherence by anchor count")
    plt.tight_layout()
    plt.savefig(fig_dir / "coherence_by_anchor_count.png", dpi=160)
    plt.close()

    mae_df = core.groupby("anchor_count")[["same_cell_mae", "time_bucket_mae", "random_cell_mae", "global_mae"]].median()
    ax = mae_df.plot(marker="o", figsize=(8, 4))
    ax.set_ylabel("MAE lower is better")
    ax.set_title("Same-cell prediction vs baselines")
    plt.tight_layout()
    plt.savefig(fig_dir / "same_cell_vs_baseline.png", dpi=160)
    plt.close()

    lift = core.groupby(["anchor_count", "H"])["top_cell_basin_lift"].median().unstack("H")
    ax = lift.plot(kind="bar", figsize=(8, 4))
    ax.axhline(1.0, color="black", linewidth=1)
    ax.set_ylabel("top-cell basin enrichment")
    plt.tight_layout()
    plt.savefig(fig_dir / "cell_basin_enrichment.png", dpi=160)
    plt.close()


def write_report(path: Path, verdict: str, decision: dict, coverage: dict, summary: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    core = summary[summary["feature_set"] == "behavior_core_no_score_time"].copy()
    lines = [
        "# Phase D8.1 Spherical Cell Coherence Audit",
        "",
        f"Verdict: **{verdict}**",
        "",
        "## Summary",
        "",
        "- D8.1 is offline-only and tests whether hypersphere cells have predictive meaning.",
        "- The primary verdict uses behavior features only; score and time are excluded from cell assignment.",
        "- Same-cell leave-one-out prediction is compared against global, same-time-bucket, and random-cell controls.",
        "",
        "## Coverage",
        "",
        "```json",
        json.dumps(coverage, indent=2),
        "```",
        "",
        "## Decision",
        "",
        "```json",
        json.dumps(json_ready(decision), indent=2),
        "```",
        "",
        "## Core Feature Summary",
        "",
        "```text",
        core.to_string(index=False, max_rows=120),
        "```",
        "",
        "## Interpretation",
        "",
    ]
    if verdict == "CELL_MAP_VALID":
        lines.append("- The behavior hypersphere cell map is coherent enough to justify Frontier Pointer replay.")
    elif verdict == "CELL_MAP_WEAK":
        lines.append("- The cell map has weak signal, but not enough to treat cells as a reliable pointer substrate.")
    elif verdict == "CELL_MAP_TIME_CONFOUNDED":
        lines.append("- Cell coherence appears only when score/time features are allowed; improve behavior-only `phi` before pointer work.")
    elif verdict == "CELL_MAP_NEEDS_MORE_DATA":
        lines.append("- Cell occupancy is too sparse/singleton-heavy for reliable leave-one-out prediction.")
    else:
        lines.append("- Same-cell neighbors do not predict future gain better than controls; do not build Frontier Pointer on this map.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    anchor_counts = parse_int_list(args.anchor_counts, DEFAULT_ANCHORS)
    anchor_seeds = parse_int_list(args.anchor_seeds, DEFAULT_SEEDS)
    analysis_dir = args.out / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.input, args.max_rows, args.basin_delta, args.time_buckets)
    summary, preds, cell_summary = run_audit(df, anchor_counts, anchor_seeds)
    verdict, decision = decide(summary, args.singleton_max_rate, args.min_nonempty_rate)
    coverage = {
        "input": str(args.input),
        "rows": int(len(df)),
        "H_values": sorted([int(x) for x in df["H"].dropna().unique().tolist()]),
        "anchor_counts": anchor_counts,
        "anchor_seeds": anchor_seeds,
        "time_buckets": args.time_buckets,
        "primary_features": CORE_FEATURES,
        "diagnostic_features": PLUS_FEATURES,
    }
    out_summary = {"verdict": verdict, "coverage": coverage, "decision": decision}

    (analysis_dir / "summary.json").write_text(json.dumps(json_ready(out_summary), indent=2), encoding="utf-8")
    summary.to_csv(analysis_dir / "cell_coherence_summary.csv", index=False)
    preds.to_csv(analysis_dir / "same_cell_predictions.csv", index=False)
    summary.to_csv(analysis_dir / "per_H_anchor_sweep.csv", index=False)
    cell_summary.to_csv(analysis_dir / "cell_enrichment.csv", index=False)
    plot_outputs(args.out, summary)
    write_report(args.report, verdict, decision, coverage, summary)
    print(f"Verdict: {verdict}")
    print(f"Rows: {len(df)}")
    print(f"Wrote: {analysis_dir / 'summary.json'}")
    print(f"Wrote: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

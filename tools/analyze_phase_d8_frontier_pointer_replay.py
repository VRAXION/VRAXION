"""Phase D8.2: offline Frontier Pointer replay audit.

This tests whether a deterministic pointer over behavior-sphere archive cells
would have selected more future-useful states than score-only, time-balanced,
random-cell, or plain Psi ranking.

Offline only: no Rust run, no SAF change, no raw candidates.csv scan.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("output/phase_d8_archive_psi_replay_20260427/analysis/panel_state_dataset.csv")
DEFAULT_KNEE_SUMMARY = Path("output/phase_d8_scan_depth_knee_20260427/analysis/summary.json")
DEFAULT_OUT = Path("output/phase_d8_frontier_pointer_replay_20260427")
DEFAULT_REPORT = Path("docs/research/PHASE_D8_FRONTIER_POINTER_REPLAY_AUDIT.md")
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
DEFAULT_KNEES = {128: 8, 256: 13, 384: 5}
SELECTORS = [
    "S0_SCORE_TOPN",
    "S1_TIME_BUCKET_TOPN",
    "R0_RANDOM_CELL",
    "P1_PSI_ONLY",
    "P2_PSI_CONF",
    "P3_PSI_NOVELTY",
    "P4_FRONTIER_POINTER",
]
EPS = 1e-12


def parse_int_list(value: str, default: list[int]) -> list[int]:
    if value is None or value == "":
        return default
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--knee-summary", type=Path, default=DEFAULT_KNEE_SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--anchor-counts", type=str, default=",".join(str(x) for x in DEFAULT_ANCHORS))
    parser.add_argument("--anchor-seeds", type=str, default=",".join(str(x) for x in DEFAULT_ANCHOR_SEEDS))
    parser.add_argument("--archive-size", type=int, default=64)
    parser.add_argument("--time-buckets", type=int, default=10)
    parser.add_argument("--basin-delta", type=float, default=0.005)
    parser.add_argument("--material-regression", type=float, default=-0.005)
    parser.add_argument("--tie-margin", type=float, default=0.002)
    parser.add_argument("--random-seed", type=int, default=8128)
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


def zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.zeros_like(arr, dtype=float)
    ok = np.isfinite(arr)
    if ok.sum() < 2:
        return out
    mu = float(np.mean(arr[ok]))
    sd = float(np.std(arr[ok]))
    if sd <= EPS:
        return out
    out[ok] = (arr[ok] - mu) / sd
    return out


def load_dataset(path: Path, max_rows: int | None, basin_delta: float, time_buckets: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing panel dataset: {path}")
    df = pd.read_csv(path)
    if "current_peak" not in df and "main_peak_acc" in df:
        df["current_peak"] = df["main_peak_acc"]
    required = {"H", "state_id", "future_gain_final", "current_peak", "psi_pred_seed_cv", "time_pct"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"missing required columns: {missing}")
    df["H"] = pd.to_numeric(df["H"], errors="coerce").astype("Int64")
    df = df[df["H"].notna()].copy()
    df["H"] = df["H"].astype(int)
    for col in ["future_gain_final", "current_peak", "psi_pred_seed_cv", "time_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["psi_pred_seed_cv"].notna()].copy()
    df["future_gain_final"] = df["future_gain_final"].fillna(0.0)
    df["current_peak"] = df["current_peak"].fillna(0.0)
    df["time_pct"] = df["time_pct"].fillna(0.5).clip(0.0, 1.0)
    if "basin_hit" in df:
        df["basin_hit"] = pd.to_numeric(df["basin_hit"], errors="coerce").fillna(0).astype(int)
    else:
        df["basin_hit"] = (df["future_gain_final"] >= basin_delta).astype(int)
    df["time_bucket"] = np.floor(np.minimum(df["time_pct"] * time_buckets, time_buckets - 1)).astype(int)
    if max_rows is not None:
        sort_cols = [c for c in ["H", "source", "run_id", "panel_index", "state_id"] if c in df]
        df = df.sort_values(sort_cols, na_position="last").head(max_rows).copy()
    return df.reset_index(drop=True)


def load_knees(path: Path) -> dict[int, int]:
    if not path.exists():
        return dict(DEFAULT_KNEES)
    data = json.loads(path.read_text(encoding="utf-8"))
    out = dict(DEFAULT_KNEES)
    by_h = data.get("decision", {}).get("by_h", {})
    for key, row in by_h.items():
        try:
            h = int(key)
            knee = row.get("median_knee_sample_n")
            if knee is not None and math.isfinite(float(knee)):
                out[h] = max(1, int(math.ceil(float(knee))))
        except (TypeError, ValueError):
            continue
    return out


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


def normalize_centroid(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= EPS:
        return vec
    return vec / norm


def build_cell_table(sub: pd.DataFrame, coords: np.ndarray, cells: np.ndarray, knee_n: int) -> pd.DataFrame:
    work = sub.copy()
    work["_local_idx"] = np.arange(len(work), dtype=int)
    work["cell_id"] = cells
    rows = []
    for cell_id, group in work.groupby("cell_id", dropna=False):
        idx = group["_local_idx"].to_numpy(dtype=int)
        centroid = normalize_centroid(np.mean(coords[idx], axis=0)) if len(idx) else np.array([])
        best_psi = (
            group.sort_values(["psi_pred_seed_cv", "current_peak", "state_id"], ascending=[False, False, True])
            .iloc[0]
            .to_dict()
        )
        best_current = (
            group.sort_values(["current_peak", "psi_pred_seed_cv", "state_id"], ascending=[False, False, True])
            .iloc[0]
            .to_dict()
        )
        count = int(len(group))
        rows.append({
            "cell_id": int(cell_id),
            "cell_count": count,
            "cell_psi_mean": float(group["psi_pred_seed_cv"].mean()),
            "cell_future_gain_mean": float(group["future_gain_final"].mean()),
            "cell_basin_rate": float(group["basin_hit"].mean()),
            "cell_current_score_mean": float(group["current_peak"].mean()),
            "confidence": float(min(1.0, count / max(knee_n, 1))),
            "centroid": centroid,
            "best_psi_idx": int(best_psi["_local_idx"]),
            "best_current_idx": int(best_current["_local_idx"]),
        })
    return pd.DataFrame(rows)


def state_rows(sub: pd.DataFrame, local_indices: list[int], selector: str, h: int, anchor_count: int, anchor_seed: int, knee_n: int, cells: np.ndarray, cell_table: pd.DataFrame) -> pd.DataFrame:
    if not local_indices:
        return pd.DataFrame()
    selected = sub.iloc[local_indices].copy()
    selected["selector"] = selector
    selected["H"] = h
    selected["anchor_count"] = anchor_count
    selected["anchor_seed"] = anchor_seed
    selected["knee_n"] = knee_n
    selected["cell_id"] = cells[local_indices]
    conf = cell_table.set_index("cell_id")["confidence"].to_dict()
    counts = cell_table.set_index("cell_id")["cell_count"].to_dict()
    selected["cell_confidence"] = selected["cell_id"].map(conf).fillna(0.0)
    selected["cell_count"] = selected["cell_id"].map(counts).fillna(0).astype(int)
    selected["selected_rank"] = np.arange(len(selected), dtype=int)
    keep = [
        "selector",
        "H",
        "anchor_count",
        "anchor_seed",
        "selected_rank",
        "state_id",
        "source",
        "run_id",
        "phase",
        "arm",
        "seed",
        "panel_index",
        "time_pct",
        "time_bucket",
        "cell_id",
        "cell_count",
        "cell_confidence",
        "current_peak",
        "psi_pred_seed_cv",
        "future_gain_final",
        "basin_hit",
    ]
    for col in keep:
        if col not in selected:
            selected[col] = np.nan
    return selected[keep]


def select_score_topn(sub: pd.DataFrame, archive_size: int) -> list[int]:
    order = sub.assign(_idx=np.arange(len(sub), dtype=int)).sort_values(
        ["current_peak", "psi_pred_seed_cv", "state_id"], ascending=[False, False, True]
    )
    return order["_idx"].head(archive_size).astype(int).tolist()


def select_time_bucket_topn(sub: pd.DataFrame, archive_size: int) -> list[int]:
    work = sub.assign(_idx=np.arange(len(sub), dtype=int)).copy()
    bucket_orders = {}
    for bucket, group in work.groupby("time_bucket", dropna=False):
        bucket_orders[int(bucket)] = group.sort_values(
            ["current_peak", "psi_pred_seed_cv", "state_id"], ascending=[False, False, True]
        )["_idx"].astype(int).tolist()
    selected = []
    cursors = {bucket: 0 for bucket in bucket_orders}
    buckets = sorted(bucket_orders)
    while len(selected) < archive_size:
        progressed = False
        for bucket in buckets:
            cursor = cursors[bucket]
            values = bucket_orders[bucket]
            if cursor < len(values):
                selected.append(values[cursor])
                cursors[bucket] += 1
                progressed = True
                if len(selected) >= archive_size:
                    break
        if not progressed:
            break
    return selected


def select_random_cells(cell_table: pd.DataFrame, archive_size: int, random_seed: int, h: int, anchor_count: int, anchor_seed: int) -> list[int]:
    rng = np.random.default_rng(random_seed + h * 100000 + anchor_count * 100 + anchor_seed)
    cells = cell_table["cell_id"].to_numpy(dtype=int)
    if len(cells) == 0:
        return []
    order = rng.permutation(cells)
    chosen = order[: min(archive_size, len(order))]
    idx_by_cell = cell_table.set_index("cell_id")["best_current_idx"].to_dict()
    return [int(idx_by_cell[int(cell)]) for cell in chosen]


def select_static_cells(cell_table: pd.DataFrame, archive_size: int, selector: str) -> tuple[list[int], pd.DataFrame]:
    work = cell_table.copy()
    if selector == "P1_PSI_ONLY":
        work["pointer_score"] = work["cell_psi_mean"]
    elif selector == "P2_PSI_CONF":
        work["pointer_score"] = work["cell_psi_mean"] * work["confidence"]
    else:
        raise ValueError(f"unknown static cell selector: {selector}")
    work = work.sort_values(["pointer_score", "cell_psi_mean", "cell_id"], ascending=[False, False, True])
    chosen = work.head(archive_size)
    return chosen["best_psi_idx"].astype(int).tolist(), chosen


def novelty_against_selected(remaining: pd.DataFrame, selected_centroids: list[np.ndarray]) -> np.ndarray:
    if not selected_centroids:
        return np.zeros(len(remaining), dtype=float)
    selected = np.vstack(selected_centroids)
    vals = []
    for centroid in remaining["centroid"]:
        if len(centroid) == 0:
            vals.append(0.0)
            continue
        sim = selected @ centroid
        vals.append(float(1.0 - np.max(sim)))
    return np.asarray(vals, dtype=float)


def select_dynamic_cells(cell_table: pd.DataFrame, archive_size: int, selector: str) -> tuple[list[int], pd.DataFrame]:
    work = cell_table.copy().reset_index(drop=True)
    work["z_psi"] = zscore(work["cell_psi_mean"].to_numpy(float))
    selected_rows = []
    selected_centroids: list[np.ndarray] = []
    remaining = set(work.index.tolist())
    while remaining and len(selected_rows) < archive_size:
        rem = work.loc[sorted(remaining)].copy()
        novelty = novelty_against_selected(rem, selected_centroids)
        z_novelty = zscore(novelty)
        if selector == "P3_PSI_NOVELTY":
            rem["pointer_score"] = rem["z_psi"].to_numpy(float) + 0.35 * z_novelty
        elif selector == "P4_FRONTIER_POINTER":
            crowding = np.full(len(rem), len(selected_rows) / max(archive_size, 1), dtype=float)
            rem["pointer_score"] = (
                rem["z_psi"].to_numpy(float)
                + 0.35 * z_novelty
                + 0.35 * rem["confidence"].to_numpy(float)
                - 0.20 * crowding
            )
        else:
            raise ValueError(f"unknown dynamic selector: {selector}")
        rem["novelty"] = novelty
        rem["z_novelty"] = z_novelty
        chosen = rem.sort_values(["pointer_score", "cell_psi_mean", "cell_id"], ascending=[False, False, True]).iloc[0]
        selected_rows.append(chosen.to_dict())
        selected_centroids.append(chosen["centroid"])
        remaining.remove(int(chosen.name))
    chosen_df = pd.DataFrame(selected_rows)
    return chosen_df["best_psi_idx"].astype(int).tolist() if not chosen_df.empty else [], chosen_df


def summarize_selection(selected: pd.DataFrame) -> dict:
    if selected.empty:
        return {
            "n_selected": 0,
            "mean_future_gain": math.nan,
            "median_future_gain": math.nan,
            "topk_basin_precision": math.nan,
            "mean_current_peak": math.nan,
            "mean_psi": math.nan,
            "unique_cells": 0,
            "confident_cell_rate": math.nan,
            "mean_cell_confidence": math.nan,
        }
    return {
        "n_selected": int(len(selected)),
        "mean_future_gain": float(selected["future_gain_final"].mean()),
        "median_future_gain": float(selected["future_gain_final"].median()),
        "topk_basin_precision": float(selected["basin_hit"].mean()),
        "mean_current_peak": float(selected["current_peak"].mean()),
        "mean_psi": float(selected["psi_pred_seed_cv"].mean()),
        "unique_cells": int(selected["cell_id"].nunique()),
        "confident_cell_rate": float((selected["cell_confidence"] >= 1.0).mean()),
        "mean_cell_confidence": float(selected["cell_confidence"].mean()),
    }


def replay_one_config(
    df: pd.DataFrame,
    h: int,
    anchor_count: int,
    anchor_seed: int,
    archive_size: int,
    knee_n: int,
    random_seed: int,
) -> tuple[list[dict], list[pd.DataFrame], pd.DataFrame]:
    sub = df[df["H"] == h].copy().reset_index(drop=True)
    if sub.empty:
        return [], [], pd.DataFrame()
    coords, _features = robust_sphere_coords(sub)
    cells = assign_cells(coords, anchor_count, anchor_seed)
    cell_table = build_cell_table(sub, coords, cells, knee_n)
    summary_rows = []
    selected_frames = []
    cell_score_frames = []

    selector_indices: dict[str, list[int]] = {
        "S0_SCORE_TOPN": select_score_topn(sub, archive_size),
        "S1_TIME_BUCKET_TOPN": select_time_bucket_topn(sub, archive_size),
        "R0_RANDOM_CELL": select_random_cells(cell_table, archive_size, random_seed, h, anchor_count, anchor_seed),
    }
    p1_idx, p1_cells = select_static_cells(cell_table, archive_size, "P1_PSI_ONLY")
    p2_idx, p2_cells = select_static_cells(cell_table, archive_size, "P2_PSI_CONF")
    p3_idx, p3_cells = select_dynamic_cells(cell_table, archive_size, "P3_PSI_NOVELTY")
    p4_idx, p4_cells = select_dynamic_cells(cell_table, archive_size, "P4_FRONTIER_POINTER")
    selector_indices.update({
        "P1_PSI_ONLY": p1_idx,
        "P2_PSI_CONF": p2_idx,
        "P3_PSI_NOVELTY": p3_idx,
        "P4_FRONTIER_POINTER": p4_idx,
    })
    for name, cells_df in [
        ("P1_PSI_ONLY", p1_cells),
        ("P2_PSI_CONF", p2_cells),
        ("P3_PSI_NOVELTY", p3_cells),
        ("P4_FRONTIER_POINTER", p4_cells),
    ]:
        if cells_df.empty:
            continue
        out_cells = cells_df.drop(columns=["centroid"], errors="ignore").copy()
        out_cells["selector"] = name
        out_cells["H"] = h
        out_cells["anchor_count"] = anchor_count
        out_cells["anchor_seed"] = anchor_seed
        out_cells["knee_n"] = knee_n
        cell_score_frames.append(out_cells)

    for selector in SELECTORS:
        selected = state_rows(sub, selector_indices.get(selector, []), selector, h, anchor_count, anchor_seed, knee_n, cells, cell_table)
        row = summarize_selection(selected)
        row.update({
            "H": h,
            "anchor_count": anchor_count,
            "anchor_seed": anchor_seed,
            "selector": selector,
            "archive_size": archive_size,
            "knee_n": knee_n,
            "total_states": int(len(sub)),
            "total_cells": int(len(cell_table)),
            "oof_only": True,
        })
        summary_rows.append(row)
        selected_frames.append(selected)
    cell_scores = pd.concat(cell_score_frames, ignore_index=True) if cell_score_frames else pd.DataFrame()
    return summary_rows, selected_frames, cell_scores


def run_replay(df: pd.DataFrame, anchor_counts: list[int], anchor_seeds: list[int], archive_size: int, knees: dict[int, int], random_seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    selected_frames = []
    cell_score_frames = []
    for h in sorted(df["H"].dropna().unique().tolist()):
        for anchor_count in anchor_counts:
            for anchor_seed in anchor_seeds:
                rows, selected, cell_scores = replay_one_config(
                    df,
                    int(h),
                    anchor_count,
                    anchor_seed,
                    archive_size,
                    knees.get(int(h), DEFAULT_KNEES.get(int(h), 8)),
                    random_seed,
                )
                summary_rows.extend(rows)
                selected_frames.extend(selected)
                if not cell_scores.empty:
                    cell_score_frames.append(cell_scores)
    summary = pd.DataFrame(summary_rows)
    selected_df = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    cell_scores_df = pd.concat(cell_score_frames, ignore_index=True) if cell_score_frames else pd.DataFrame()
    return summary, selected_df, cell_scores_df


def compute_deltas(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_cols = ["H", "anchor_count", "anchor_seed"]
    base = summary[summary["selector"] == "S0_SCORE_TOPN"][base_cols + ["mean_future_gain", "topk_basin_precision"]].rename(
        columns={"mean_future_gain": "score_mean_future_gain", "topk_basin_precision": "score_basin_precision"}
    )
    time = summary[summary["selector"] == "S1_TIME_BUCKET_TOPN"][base_cols + ["mean_future_gain", "topk_basin_precision"]].rename(
        columns={"mean_future_gain": "time_mean_future_gain", "topk_basin_precision": "time_basin_precision"}
    )
    rand = summary[summary["selector"] == "R0_RANDOM_CELL"][base_cols + ["mean_future_gain"]].rename(
        columns={"mean_future_gain": "random_mean_future_gain"}
    )
    merged = summary.merge(base, on=base_cols, how="left").merge(time, on=base_cols, how="left").merge(rand, on=base_cols, how="left")
    merged["delta_vs_score"] = merged["mean_future_gain"] - merged["score_mean_future_gain"]
    merged["delta_vs_time"] = merged["mean_future_gain"] - merged["time_mean_future_gain"]
    merged["delta_vs_random"] = merged["mean_future_gain"] - merged["random_mean_future_gain"]
    merged["delta_basin_vs_score"] = merged["topk_basin_precision"] - merged["score_basin_precision"]
    merged["delta_basin_vs_time"] = merged["topk_basin_precision"] - merged["time_basin_precision"]
    per_h = (
        merged.groupby(["H", "selector"], dropna=False)
        .agg(
            configs=("mean_future_gain", "size"),
            mean_future_gain=("mean_future_gain", "mean"),
            mean_delta_vs_score=("delta_vs_score", "mean"),
            mean_delta_vs_time=("delta_vs_time", "mean"),
            mean_delta_vs_random=("delta_vs_random", "mean"),
            mean_basin_precision=("topk_basin_precision", "mean"),
            mean_confident_cell_rate=("confident_cell_rate", "mean"),
            mean_unique_cells=("unique_cells", "mean"),
            mean_current_peak=("mean_current_peak", "mean"),
            mean_psi=("mean_psi", "mean"),
        )
        .reset_index()
    )
    return merged, per_h


def decide(summary: pd.DataFrame, per_h: pd.DataFrame, args: argparse.Namespace) -> tuple[str, dict]:
    h_values = sorted(summary["H"].dropna().unique().tolist())
    required_h = math.ceil((2 / 3) * len(h_values)) if h_values else 999
    p4 = per_h[per_h["selector"] == "P4_FRONTIER_POINTER"].copy()
    p1 = per_h[per_h["selector"] == "P1_PSI_ONLY"].copy()
    p2 = per_h[per_h["selector"] == "P2_PSI_CONF"].copy()
    p3 = per_h[per_h["selector"] == "P3_PSI_NOVELTY"].copy()
    random = per_h[per_h["selector"] == "R0_RANDOM_CELL"].copy()

    if summary.empty or p4.empty or len(h_values) == 0:
        return "D8_POINTER_NEEDS_MORE_DATA", {"reason": "missing_pointer_rows"}
    if int(summary["total_states"].min()) < max(8, args.archive_size // 2):
        return "D8_POINTER_NEEDS_MORE_DATA", {"reason": "too_few_oof_states"}

    p4_by_h = p4.set_index("H")
    pass_baselines_h = int(((p4_by_h["mean_delta_vs_score"] > 0.0) & (p4_by_h["mean_delta_vs_time"] > 0.0)).sum())
    random_h = int((p4_by_h["mean_delta_vs_random"] > 0.0).sum())
    material_regression = bool((p4_by_h[["mean_delta_vs_score", "mean_delta_vs_time"]].min(axis=1) < args.material_regression).any())

    peer = pd.concat([p1, p2, p3], ignore_index=True)
    peer_best = peer.groupby("H", dropna=False).agg(
        best_peer_future_gain=("mean_future_gain", "max"),
        best_peer_confident_cell_rate=("mean_confident_cell_rate", "max"),
        best_peer_unique_cells=("mean_unique_cells", "max"),
    )
    joined = p4_by_h.join(peer_best, how="left")
    competitive_h = int((
        (joined["mean_future_gain"] >= joined["best_peer_future_gain"] - args.tie_margin)
        & (joined["mean_confident_cell_rate"] >= joined["best_peer_confident_cell_rate"] - 0.05)
        & (joined["mean_unique_cells"] >= joined["best_peer_unique_cells"] - 1.0)
    ).sum())

    p1_by_h = p1.set_index("H")
    p1_pass_h = int(((p1_by_h["mean_delta_vs_score"] > 0.0) & (p1_by_h["mean_delta_vs_time"] > 0.0)).sum()) if not p1_by_h.empty else 0
    p1_material_regression = bool((p1_by_h[["mean_delta_vs_score", "mean_delta_vs_time"]].min(axis=1) < args.material_regression).any()) if not p1_by_h.empty else True
    selector_means = per_h.groupby("selector", dropna=False)["mean_future_gain"].mean().sort_values(ascending=False)
    best_selector = str(selector_means.index[0]) if not selector_means.empty else ""
    best_selector_mean = float(selector_means.iloc[0]) if not selector_means.empty else math.nan
    p4_mean = float(selector_means.get("P4_FRONTIER_POINTER", math.nan))
    pointer_beats_random = random_h >= required_h
    pointer_pass = (
        pass_baselines_h >= required_h
        and not material_regression
        and competitive_h >= required_h
        and pointer_beats_random
    )
    psi_only_pass = p1_pass_h >= required_h and not p1_material_regression
    weak = random_h >= required_h and pass_baselines_h < required_h
    if pointer_pass:
        verdict = "D8_POINTER_REPLAY_PASS"
    elif psi_only_pass:
        verdict = "D8_PSI_ONLY_PASS"
    elif weak:
        verdict = "D8_POINTER_WEAK"
    else:
        verdict = "D8_POINTER_REJECT"

    decision = {
        "required_h": required_h,
        "h_values": [int(x) for x in h_values],
        "p4_pass_baselines_h": pass_baselines_h,
        "p4_beats_random_h": random_h,
        "p4_competitive_with_peer_h": competitive_h,
        "p4_material_regression": material_regression,
        "p1_pass_baselines_h": p1_pass_h,
        "p1_material_regression": p1_material_regression,
        "best_selector_overall": best_selector,
        "best_selector_mean_future_gain": best_selector_mean,
        "p4_gap_vs_best_selector": p4_mean - best_selector_mean if math.isfinite(p4_mean) and math.isfinite(best_selector_mean) else math.nan,
        "by_h": {},
    }
    for h in h_values:
        h_int = int(h)
        row = p4_by_h.loc[h] if h in p4_by_h.index else pd.Series(dtype=float)
        decision["by_h"][str(h_int)] = {
            "p4_mean_future_gain": float(row.get("mean_future_gain", math.nan)),
            "p4_delta_vs_score": float(row.get("mean_delta_vs_score", math.nan)),
            "p4_delta_vs_time": float(row.get("mean_delta_vs_time", math.nan)),
            "p4_delta_vs_random": float(row.get("mean_delta_vs_random", math.nan)),
            "p4_confident_cell_rate": float(row.get("mean_confident_cell_rate", math.nan)),
            "p4_unique_cells": float(row.get("mean_unique_cells", math.nan)),
        }
    return verdict, decision


def plot_outputs(out: Path, per_h: pd.DataFrame, selected: pd.DataFrame) -> None:
    fig_dir = out / "analysis" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if per_h.empty:
        return
    pivot = per_h[per_h["selector"].isin(["P1_PSI_ONLY", "P2_PSI_CONF", "P3_PSI_NOVELTY", "P4_FRONTIER_POINTER"])].pivot(
        index="H", columns="selector", values="mean_delta_vs_time"
    )
    ax = pivot.plot(kind="bar", figsize=(9, 4))
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel("delta mean future_gain vs time baseline")
    ax.set_title("Pointer delta by H")
    plt.tight_layout()
    plt.savefig(fig_dir / "pointer_delta_by_H.png", dpi=160)
    plt.close()

    fg = per_h.pivot(index="H", columns="selector", values="mean_future_gain")
    ax = fg.plot(marker="o", figsize=(9, 4))
    ax.set_ylabel("mean selected future_gain")
    ax.set_title("Selected future_gain by selector")
    plt.tight_layout()
    plt.savefig(fig_dir / "pointer_selected_future_gain.png", dpi=160)
    plt.close()

    conf = per_h.pivot(index="H", columns="selector", values="mean_confident_cell_rate")
    ax = conf.plot(marker="o", figsize=(9, 4))
    ax.set_ylabel("selected confident-cell rate")
    ax.set_title("Pointer confidence coverage")
    plt.tight_layout()
    plt.savefig(fig_dir / "pointer_confidence_coverage.png", dpi=160)
    plt.close()


def write_report(path: Path, verdict: str, coverage: dict, decision: dict, per_h: pd.DataFrame, summary: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    view = per_h.sort_values(["H", "selector"])
    agg = (
        summary.groupby("selector", dropna=False)
        .agg(
            configs=("mean_future_gain", "size"),
            mean_future_gain=("mean_future_gain", "mean"),
            mean_delta_vs_score=("delta_vs_score", "mean"),
            mean_delta_vs_time=("delta_vs_time", "mean"),
            mean_delta_vs_random=("delta_vs_random", "mean"),
            mean_basin_precision=("topk_basin_precision", "mean"),
            mean_confident_cell_rate=("confident_cell_rate", "mean"),
            mean_unique_cells=("unique_cells", "mean"),
        )
        .reset_index()
        .sort_values("mean_future_gain", ascending=False)
    )
    lines = [
        "# Phase D8.2 Frontier Pointer Replay Audit",
        "",
        f"Verdict: **{verdict}**",
        "",
        "## Summary",
        "",
        "- D8.2 is offline-only: no live Rust run, no SAF change, no K change, no acceptance change.",
        "- The primary replay uses OOF-only `psi_pred_seed_cv`; in-sample `psi_pred` fallback is not used.",
        "- Behavior-sphere cell assignment excludes score and time features.",
        "- D8.1.1 scan-depth knees are used only as confidence gates, not as proof that a cell is predictive.",
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
        "## Aggregate Selector Summary",
        "",
        "```text",
        agg.to_string(index=False, max_rows=120),
        "```",
        "",
        "## Per-H Selector Summary",
        "",
        "```text",
        view.to_string(index=False, max_rows=160),
        "```",
        "",
        "## Interpretation",
        "",
    ]
    if verdict == "D8_POINTER_REPLAY_PASS":
        best = decision.get("best_selector_overall", "")
        gap = decision.get("p4_gap_vs_best_selector", math.nan)
        lines.append("- `P4_FRONTIER_POINTER` beats score/time/random baselines offline and is competitive with simpler Psi selectors.")
        if best and best != "P4_FRONTIER_POINTER":
            lines.append(f"- Numeric best selector is `{best}`; P4 gap vs best is `{gap:.6f}` mean future_gain. Treat novelty as optional until live-tested.")
        lines.append("- Next step is D8.3 instrumentation-only, not live steering yet.")
    elif verdict == "D8_PSI_ONLY_PASS":
        lines.append("- Plain OOF Psi selection is useful, but novelty/confidence pointer composition does not add enough value yet.")
    elif verdict == "D8_POINTER_WEAK":
        lines.append("- Pointer beats random but not the score/time baselines consistently. Improve `phi` or pointer formula before instrumentation.")
    elif verdict == "D8_POINTER_NEEDS_MORE_DATA":
        lines.append("- OOF Psi or cell occupancy is too sparse for reliable pointer replay.")
    else:
        lines.append("- Pointer replay does not beat baselines; do not build live pointer on this signal.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    anchor_counts = parse_int_list(args.anchor_counts, DEFAULT_ANCHORS)
    anchor_seeds = parse_int_list(args.anchor_seeds, DEFAULT_ANCHOR_SEEDS)
    analysis_dir = args.out / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.input, args.max_rows, args.basin_delta, args.time_buckets)
    knees = load_knees(args.knee_summary)
    summary, selected, cell_scores = run_replay(df, anchor_counts, anchor_seeds, args.archive_size, knees, args.random_seed)
    detailed, per_h = compute_deltas(summary)
    verdict, decision = decide(detailed, per_h, args)
    coverage = {
        "input": str(args.input),
        "knee_summary": str(args.knee_summary),
        "rows_oof": int(len(df)),
        "H_values": sorted([int(x) for x in df["H"].dropna().unique().tolist()]),
        "anchor_counts": anchor_counts,
        "anchor_seeds": anchor_seeds,
        "archive_size": args.archive_size,
        "time_buckets": args.time_buckets,
        "random_seed": args.random_seed,
        "features": CORE_FEATURES,
        "knee_by_H": {str(k): int(v) for k, v in sorted(knees.items())},
        "psi_column": "psi_pred_seed_cv",
        "psi_policy": "OOF-only; drop missing predictions",
    }
    out_summary = {"verdict": verdict, "coverage": coverage, "decision": decision}
    (analysis_dir / "summary.json").write_text(json.dumps(json_ready(out_summary), indent=2), encoding="utf-8")
    detailed.to_csv(analysis_dir / "pointer_replay_summary.csv", index=False)
    per_h.to_csv(analysis_dir / "per_H_pointer_deltas.csv", index=False)
    selected.to_csv(analysis_dir / "selected_states.csv", index=False)
    cell_scores.to_csv(analysis_dir / "pointer_cell_scores.csv", index=False)
    plot_outputs(args.out, per_h, selected)
    write_report(args.report, verdict, coverage, decision, per_h, detailed)
    print(f"Verdict: {verdict}")
    print(f"Rows OOF: {len(df)}")
    print(f"Wrote: {analysis_dir / 'summary.json'}")
    print(f"Wrote: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

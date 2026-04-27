"""Phase D8.6: Cell Atlas / Basin Map Dashboard.

Builds a standalone HTML "mission control" dashboard for D8 behavior-sphere
archive cells. This is analysis/visualization only: no Rust run, no SAF
change, no archive parent switching.

The geometry intentionally uses only behavior-core features. Score and time
features are displayed as metrics, but they are not used to place or assign
cells.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("output/phase_d8_archive_psi_replay_20260427/analysis/panel_state_dataset.csv")
DEFAULT_KNEE_SUMMARY = Path("output/phase_d8_scan_depth_knee_20260427/analysis/summary.json")
DEFAULT_OUT = Path("output/phase_d8_cell_atlas_20260427")
DEFAULT_REPORT = Path("docs/research/PHASE_D8_CELL_ATLAS.md")

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
DEFAULT_KNEES = {128: 8, 256: 13, 384: 5}
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--knee-summary", type=Path, default=DEFAULT_KNEE_SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--anchor-count", type=int, default=64)
    parser.add_argument("--anchor-seed", type=int, default=11)
    parser.add_argument("--knn-k", type=int, default=6)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--sample-limit-per-cell", type=int, default=20)
    parser.add_argument("--time-buckets", type=int, default=10)
    parser.add_argument("--basin-delta", type=float, default=0.005)
    return parser.parse_args()


def json_ready(obj: Any) -> Any:
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


def zscore(values: pd.Series | np.ndarray) -> np.ndarray:
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


def minmax(values: list[float], default: tuple[float, float] = (0.0, 1.0)) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return default
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if abs(hi - lo) <= EPS:
        hi = lo + 1.0
    return lo, hi


def load_knees(path: Path) -> dict[int, int]:
    out = dict(DEFAULT_KNEES)
    if not path.exists():
        return out
    data = json.loads(path.read_text(encoding="utf-8"))
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


def load_dataset(path: Path, max_rows: int | None, basin_delta: float, time_buckets: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing panel-state dataset: {path}")
    df = pd.read_csv(path)
    required = {"H", "state_id", "future_gain_final", "time_pct"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"missing required panel columns: {missing}")
    if "current_peak" not in df and "main_peak_acc" in df:
        df["current_peak"] = df["main_peak_acc"]
    if "current_peak" not in df:
        df["current_peak"] = 0.0
    if "psi_pred_seed_cv" in df:
        df["psi_atlas"] = pd.to_numeric(df["psi_pred_seed_cv"], errors="coerce")
    elif "psi_pred" in df:
        df["psi_atlas"] = pd.to_numeric(df["psi_pred"], errors="coerce")
    else:
        raise SystemExit("missing Ψ prediction column: expected psi_pred_seed_cv or psi_pred")
    df["H"] = pd.to_numeric(df["H"], errors="coerce").astype("Int64")
    df = df[df["H"].notna()].copy()
    df["H"] = df["H"].astype(int)
    for col in ["future_gain_final", "current_peak", "time_pct", "psi_atlas"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["psi_atlas"].notna()].copy()
    df["future_gain_final"] = df["future_gain_final"].fillna(0.0)
    df["current_peak"] = df["current_peak"].fillna(0.0)
    df["time_pct"] = df["time_pct"].fillna(0.5).clip(0.0, 1.0)
    if "basin_hit" in df:
        df["basin_hit"] = pd.to_numeric(df["basin_hit"], errors="coerce").fillna(0).astype(int)
    else:
        df["basin_hit"] = (df["future_gain_final"] >= basin_delta).astype(int)
    df["time_bucket"] = np.floor(np.minimum(df["time_pct"] * time_buckets, time_buckets - 1)).astype(int)
    sort_cols = [c for c in ["H", "source", "run_id", "panel_index", "state_id"] if c in df.columns]
    df = df.sort_values(sort_cols, na_position="last").reset_index(drop=True)
    if max_rows is not None and max_rows > 0 and len(df) > max_rows:
        per_h = max(1, int(math.ceil(max_rows / max(df["H"].nunique(), 1))))
        df = (
            df.groupby("H", group_keys=False, sort=True)
            .head(per_h)
            .sort_values(sort_cols, na_position="last")
            .head(max_rows)
            .reset_index(drop=True)
        )
    return df


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


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= EPS:
        return vec
    return vec / norm


def pca_2d(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return np.empty((0, 2))
    if x.shape[1] == 0:
        return np.zeros((len(x), 2))
    centered = x - np.mean(x, axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        basis = vt[: min(2, len(vt))]
        projected = centered @ basis.T
    except np.linalg.LinAlgError:
        projected = centered[:, : min(2, centered.shape[1])]
    if projected.shape[1] == 1:
        projected = np.column_stack([projected[:, 0], np.zeros(len(projected))])
    elif projected.shape[1] == 0:
        projected = np.zeros((len(x), 2))
    return projected[:, :2]


def normalized_plot_positions(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    out = points.astype(float).copy()
    for col in range(2):
        lo = float(np.min(out[:, col]))
        hi = float(np.max(out[:, col]))
        if abs(hi - lo) <= EPS:
            out[:, col] = 0.5
        else:
            out[:, col] = (out[:, col] - lo) / (hi - lo)
    out[:, 1] = 1.0 - out[:, 1]
    return out


def iqr(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)
    if len(vals) == 0:
        return math.nan
    return float(np.percentile(vals, 75) - np.percentile(vals, 25))


def safe_std(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)
    if len(vals) < 2:
        return 0.0
    return float(np.std(vals, ddof=0))


def diversity(series: pd.Series) -> int:
    return int(series.dropna().astype(str).nunique())


def build_cells_for_h(
    sub: pd.DataFrame,
    coords: np.ndarray,
    cells: np.ndarray,
    knee_n: int,
    h: int,
    anchor_count: int,
    anchor_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    work = sub.copy().reset_index(drop=True)
    work["cell_id"] = cells
    work["_local_idx"] = np.arange(len(work), dtype=int)
    rows = []
    centroids = []
    for cell_id, group in work.groupby("cell_id", sort=True, dropna=False):
        idx = group["_local_idx"].to_numpy(dtype=int)
        centroid = normalize_vector(np.mean(coords[idx], axis=0)) if len(idx) and coords.shape[1] else np.array([])
        centroids.append(centroid)
        n = int(len(group))
        rows.append({
            "H": h,
            "cell_id": int(cell_id),
            "atlas_cell_key": f"H{h}_C{int(cell_id)}",
            "anchor_count": anchor_count,
            "anchor_seed": anchor_seed,
            "n_samples": n,
            "knee_H": int(knee_n),
            "confidence": float(min(1.0, n / max(knee_n, 1))),
            "mean_future_gain": float(group["future_gain_final"].mean()),
            "median_future_gain": float(group["future_gain_final"].median()),
            "std_future_gain": safe_std(group["future_gain_final"]),
            "iqr_future_gain": iqr(group["future_gain_final"]),
            "mean_psi": float(group["psi_atlas"].mean()),
            "median_psi": float(group["psi_atlas"].median()),
            "basin_precision": float(group["basin_hit"].mean()),
            "mean_current_peak": float(group["current_peak"].mean()),
            "mean_time_pct": float(group["time_pct"].mean()),
            "source_diversity": diversity(group["source"]) if "source" in group else 0,
            "run_diversity": diversity(group["global_run_id"] if "global_run_id" in group else group.get("run_id", pd.Series(dtype=object))),
            "_local_indices": idx.tolist(),
            "_centroid": centroid,
        })
    cell_df = pd.DataFrame(rows)
    if cell_df.empty:
        return cell_df, work, np.empty((0, coords.shape[1] if coords.ndim == 2 else 0))

    for metric, out_col in [
        ("mean_psi", "_z_mean_psi"),
        ("std_future_gain", "_z_std_future_gain"),
        ("iqr_future_gain", "_z_iqr_future_gain"),
        ("basin_precision", "_z_basin_precision"),
        ("confidence", "_z_confidence"),
        ("mean_future_gain", "_z_mean_future_gain"),
    ]:
        cell_df[out_col] = zscore(cell_df[metric])
    mismatch = np.abs(cell_df["mean_psi"].to_numpy(float) - cell_df["mean_future_gain"].to_numpy(float))
    cell_df["_z_mismatch"] = zscore(mismatch)
    under_penalty = np.where(cell_df["n_samples"].to_numpy(float) < 2.0 * knee_n, 1.0, 0.0)
    cell_df["scan_priority"] = (
        0.45 * cell_df["_z_mean_psi"]
        + 0.25 * cell_df["_z_std_future_gain"]
        + 0.20 * (1.0 - cell_df["confidence"])
        + 0.10 * cell_df["_z_basin_precision"]
    )
    cell_df["split_score"] = (
        cell_df["_z_std_future_gain"]
        + cell_df["_z_iqr_future_gain"]
        + cell_df["_z_mismatch"]
        - under_penalty
    )
    cell_df["branch_trial_score"] = (
        cell_df["_z_mean_psi"]
        + cell_df["_z_confidence"]
        + 0.5 * cell_df["_z_mean_future_gain"]
        - 0.25 * cell_df["_z_std_future_gain"]
    )
    cell_df["retire_score"] = (
        -cell_df["_z_mean_psi"]
        -cell_df["_z_mean_future_gain"]
        -0.25 * cell_df["_z_std_future_gain"]
        + cell_df["_z_confidence"]
    )

    q75_std = float(cell_df["std_future_gain"].quantile(0.75))
    q75_iqr = float(cell_df["iqr_future_gain"].quantile(0.75))
    q75_psi = float(cell_df["mean_psi"].quantile(0.75))
    q25_psi = float(cell_df["mean_psi"].quantile(0.25))
    q25_gain = float(cell_df["mean_future_gain"].quantile(0.25))
    med_std = float(cell_df["std_future_gain"].median())
    cell_df["split_candidate"] = (
        (cell_df["n_samples"] >= 2 * knee_n)
        & (cell_df["std_future_gain"] >= q75_std)
        & (cell_df["iqr_future_gain"] >= q75_iqr)
    )
    cell_df["sample_more_candidate"] = (
        (cell_df["confidence"] < 1.0)
        & ((cell_df["mean_psi"] >= q75_psi) | (cell_df["std_future_gain"] >= q75_std))
    )
    cell_df["retire_candidate"] = (
        (cell_df["confidence"] >= 1.0)
        & (cell_df["mean_psi"] <= q25_psi)
        & (cell_df["mean_future_gain"] <= q25_gain)
        & (cell_df["std_future_gain"] <= med_std)
    )
    centroids = np.vstack(cell_df["_centroid"].to_numpy()) if len(cell_df) else np.empty((0, coords.shape[1]))
    pca = normalized_plot_positions(pca_2d(centroids))
    cell_df["pca_x"] = pca[:, 0] if len(pca) else []
    cell_df["pca_y"] = pca[:, 1] if len(pca) else []
    keep = [c for c in cell_df.columns if not c.startswith("_")]
    return cell_df[keep + ["_local_indices"]], work, centroids


def build_neighbors(cell_df: pd.DataFrame, centroids: np.ndarray, k: int) -> pd.DataFrame:
    if len(cell_df) == 0 or len(centroids) == 0:
        return pd.DataFrame(columns=["H", "source_cell_id", "target_cell_id", "cosine_similarity", "rank"])
    sim = centroids @ centroids.T
    np.fill_diagonal(sim, -np.inf)
    rows = []
    h = int(cell_df["H"].iloc[0])
    cell_ids = cell_df["cell_id"].to_numpy(dtype=int)
    for i, source in enumerate(cell_ids):
        order = np.argsort(-sim[i])[: min(k, len(cell_ids) - 1)]
        for rank, j in enumerate(order, start=1):
            val = float(sim[i, j])
            if math.isfinite(val):
                rows.append({
                    "H": h,
                    "source_cell_id": int(source),
                    "target_cell_id": int(cell_ids[j]),
                    "cosine_similarity": val,
                    "rank": rank,
                })
    return pd.DataFrame(rows)


def sample_states(work: pd.DataFrame, cell_df: pd.DataFrame, limit: int) -> pd.DataFrame:
    rows = []
    sort_cols = ["psi_atlas", "future_gain_final", "current_peak", "state_id"]
    for _, cell in cell_df.iterrows():
        idx = cell["_local_indices"]
        group = work.iloc[idx].copy()
        group = group.sort_values(sort_cols, ascending=[False, False, False, True]).head(limit)
        for _, state in group.iterrows():
            rows.append({
                "H": int(cell["H"]),
                "cell_id": int(cell["cell_id"]),
                "state_id": state.get("state_id"),
                "source": state.get("source"),
                "run_id": state.get("run_id"),
                "phase": state.get("phase"),
                "arm": state.get("arm"),
                "seed": state.get("seed"),
                "panel_index": state.get("panel_index"),
                "time_pct": state.get("time_pct"),
                "current_peak": state.get("current_peak"),
                "future_gain_final": state.get("future_gain_final"),
                "psi_pred": state.get("psi_atlas"),
                "basin_hit": state.get("basin_hit"),
            })
    return pd.DataFrame(rows)


def build_atlas(
    df: pd.DataFrame,
    knees: dict[int, int],
    anchor_count: int,
    anchor_seed: int,
    knn_k: int,
    sample_limit_per_cell: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_cells = []
    all_neighbors = []
    all_samples = []
    for h in sorted(df["H"].dropna().unique()):
        sub = df[df["H"] == h].copy().reset_index(drop=True)
        coords, used = robust_sphere_coords(sub)
        if coords.shape[1] == 0:
            continue
        cells = assign_cells(coords, anchor_count, anchor_seed)
        knee_n = int(knees.get(int(h), DEFAULT_KNEES.get(int(h), 8)))
        cell_df, work, centroids = build_cells_for_h(sub, coords, cells, knee_n, int(h), anchor_count, anchor_seed)
        cell_df["used_features"] = ",".join(used)
        neighbors = build_neighbors(cell_df, centroids, knn_k)
        samples = sample_states(work, cell_df, sample_limit_per_cell)
        all_cells.append(cell_df.drop(columns=["_local_indices"], errors="ignore"))
        all_neighbors.append(neighbors)
        all_samples.append(samples)
    cell_table = pd.concat(all_cells, ignore_index=True) if all_cells else pd.DataFrame()
    neighbors = pd.concat(all_neighbors, ignore_index=True) if all_neighbors else pd.DataFrame()
    samples = pd.concat(all_samples, ignore_index=True) if all_samples else pd.DataFrame()
    return cell_table, neighbors, samples


def top_candidates(cell_table: pd.DataFrame, kind: str, n: int = 20) -> pd.DataFrame:
    if cell_table.empty:
        return cell_table.copy()
    if kind == "sample_more":
        mask = cell_table["sample_more_candidate"].astype(bool)
        sort = ["scan_priority", "mean_psi", "std_future_gain", "cell_id"]
        asc = [False, False, False, True]
    elif kind == "split":
        mask = cell_table["split_candidate"].astype(bool)
        sort = ["split_score", "std_future_gain", "iqr_future_gain", "cell_id"]
        asc = [False, False, False, True]
    elif kind == "branch_trial":
        mask = pd.Series(True, index=cell_table.index)
        sort = ["branch_trial_score", "confidence", "mean_psi", "cell_id"]
        asc = [False, False, False, True]
    elif kind == "retire":
        mask = cell_table["retire_candidate"].astype(bool)
        sort = ["retire_score", "confidence", "cell_id"]
        asc = [False, False, True]
    else:
        raise ValueError(kind)
    return (
        cell_table[mask]
        .sort_values(sort, ascending=asc)
        .groupby("H", group_keys=False)
        .head(n)
        .reset_index(drop=True)
    )


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def html_template(cell_rows: list[dict], edge_rows: list[dict], sample_rows: list[dict], summary: dict) -> str:
    data = {
        "cells": cell_rows,
        "edges": edge_rows,
        "samples": sample_rows,
        "summary": summary,
    }
    data_json = json.dumps(json_ready(data), ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Phase D8.6 Cell Atlas / Basin Pokedex</title>
<style>
:root {{
  --bg-deep: #0a1018;
  --bg-mid: #0f1722;
  --bg-soft: #15202e;
  --panel: #18243380;
  --panel-solid: #182433;
  --panel-edge: #243349;
  --panel-edge-strong: #324663;
  --ink: #e8eef7;
  --ink-soft: #c4cfdf;
  --muted: #8896aa;
  --muted-dim: #5e6b80;
  --gold: #f4c95d;
  --gold-soft: #f4c95d33;
  --mint: #4ee6c8;
  --mint-soft: #4ee6c822;
  --teal: #38b6c2;
  --rose: #ff6b8a;
  --rose-soft: #ff6b8a22;
  --amber: #ffb547;
  --amber-soft: #ffb54722;
  --gray-pill: #6f7a8b;
  --bar-track: #0d1622;
  --bar-track-edge: #1d2a3d;
  --h128: #6aa9ff;
  --h128-soft: #6aa9ff22;
  --h128-deep: #2c5fb8;
  --h256: #ffb547;
  --h256-soft: #ffb54722;
  --h256-deep: #c08020;
  --h384: #c69bff;
  --h384-soft: #c69bff22;
  --h384-deep: #7a4cc4;
  --good-1: #16a37a;
  --good-2: #4ee6c8;
  --warn-1: #f4c95d;
  --warn-2: #ffa64a;
  --bad-1: #ff6b8a;
  --bad-2: #ff4870;
  --shadow: 0 18px 42px rgba(0,0,0,.42);
  --shadow-soft: 0 6px 18px rgba(0,0,0,.28);
}}
* {{ box-sizing: border-box; }}
html, body {{ height: 100%; }}
body {{
  margin: 0;
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", "Inter", sans-serif;
  background:
    radial-gradient(1200px 600px at 0% 0%, #1a2a44 0%, transparent 60%),
    radial-gradient(900px 500px at 100% 0%, #2a1c3a 0%, transparent 55%),
    linear-gradient(180deg, var(--bg-mid), var(--bg-deep) 70%);
  color: var(--ink);
  font-size: 13px;
  line-height: 1.45;
  min-height: 100vh;
}}
button, input, select {{ font-family: inherit; }}
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: #283549; border-radius: 999px; border: 2px solid transparent; background-clip: padding-box; }}
::-webkit-scrollbar-thumb:hover {{ background: #3a4a66; background-clip: padding-box; border: 2px solid transparent; }}

/* ---------- Header ---------- */
header.app-header {{
  position: sticky;
  top: 0;
  z-index: 30;
  padding: 14px 22px 10px;
  border-bottom: 1px solid var(--panel-edge);
  background: linear-gradient(180deg, rgba(10,16,24,.96), rgba(10,16,24,.84));
  backdrop-filter: blur(8px);
}}
.header-row {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
}}
.brand {{
  display: flex;
  align-items: center;
  gap: 12px;
  min-width: 0;
}}
.brand-mark {{
  width: 38px;
  height: 38px;
  border-radius: 10px;
  background: linear-gradient(135deg, var(--gold), #b8870b);
  display: grid;
  place-items: center;
  color: #1b1305;
  font-weight: 800;
  font-size: 16px;
  letter-spacing: 0.04em;
  box-shadow: 0 6px 18px rgba(244,201,93,.35), inset 0 1px 0 rgba(255,255,255,.4);
}}
.brand h1 {{
  margin: 0;
  font-size: 17px;
  letter-spacing: 0.02em;
  font-weight: 700;
}}
.brand .tagline {{
  color: var(--muted);
  font-size: 11.5px;
  margin-top: 2px;
}}
.verdict-pill {{
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  background: var(--mint-soft);
  color: var(--mint);
  border: 1px solid #4ee6c855;
}}
.verdict-pill.bad {{ background: var(--rose-soft); color: var(--rose); border-color: #ff6b8a55; }}
.verdict-pill.warn {{ background: var(--amber-soft); color: var(--amber); border-color: #ffb54755; }}

.summary-strip {{
  margin-top: 10px;
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}}
.summary-chip {{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 11px;
  border-radius: 10px;
  background: rgba(24,36,51,.85);
  border: 1px solid var(--panel-edge);
  font-size: 11.5px;
  color: var(--ink-soft);
}}
.summary-chip .h-dot {{
  width: 10px;
  height: 10px;
  border-radius: 4px;
}}
.summary-chip .label {{ color: var(--muted); margin-right: 2px; }}
.summary-chip .num {{ color: var(--ink); font-weight: 700; font-feature-settings: "tnum" 1; }}
.summary-chip .sub {{ color: var(--muted-dim); font-size: 10.5px; }}

/* ---------- Toolbar ---------- */
.toolbar {{
  position: sticky;
  top: 64px;
  z-index: 25;
  background: linear-gradient(180deg, rgba(15,23,34,.95), rgba(15,23,34,.85));
  backdrop-filter: blur(8px);
  border-bottom: 1px solid var(--panel-edge);
  padding: 10px 22px;
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 14px 18px;
  align-items: center;
}}
.toolbar .group {{
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}}
.toolbar .group-label {{
  text-transform: uppercase;
  font-size: 10px;
  letter-spacing: 0.12em;
  color: var(--muted);
  margin-right: 4px;
  font-weight: 700;
}}
.chip {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 11px;
  border-radius: 999px;
  background: rgba(20,30,44,.85);
  color: var(--ink-soft);
  border: 1px solid var(--panel-edge);
  cursor: pointer;
  font-size: 11.5px;
  font-weight: 600;
  user-select: none;
  transition: background .12s ease, border-color .12s ease, color .12s ease, transform .08s ease;
}}
.chip:hover {{ border-color: var(--panel-edge-strong); }}
.chip:active {{ transform: translateY(1px); }}
.chip.active {{
  background: var(--gold-soft);
  color: var(--gold);
  border-color: #f4c95d99;
  box-shadow: 0 0 0 1px rgba(244,201,93,.2) inset;
}}
.chip .ico {{
  font-size: 12px;
  line-height: 1;
}}
.chip.h-128.active {{ background: var(--h128-soft); color: var(--h128); border-color: #6aa9ff99; }}
.chip.h-256.active {{ background: var(--h256-soft); color: var(--h256); border-color: #ffb54799; }}
.chip.h-384.active {{ background: var(--h384-soft); color: var(--h384); border-color: #c69bff99; }}
.chip.cand-split.active {{ background: var(--rose-soft); color: var(--rose); border-color: #ff6b8a99; }}
.chip.cand-sample.active {{ background: var(--amber-soft); color: var(--amber); border-color: #ffb54799; }}
.chip.cand-retire.active {{ background: rgba(150,160,180,.18); color: #a9b4c4; border-color: #6f7a8baa; }}
.chip.cand-branch.active {{ background: rgba(56,182,194,.18); color: var(--teal); border-color: #38b6c2aa; }}
.chip.sort-chip .arrow {{
  display: inline-block;
  width: 8px;
  text-align: center;
  font-size: 10px;
  color: currentColor;
  opacity: .8;
}}
.chip.sort-chip.active .arrow {{ font-weight: 800; }}

.slider-wrap {{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 5px 11px;
  border-radius: 999px;
  background: rgba(20,30,44,.85);
  border: 1px solid var(--panel-edge);
  font-size: 11.5px;
  color: var(--muted);
}}
.slider-wrap input[type="range"] {{
  width: 110px;
  accent-color: var(--gold);
}}
.slider-wrap .val {{ color: var(--ink); font-weight: 700; min-width: 26px; text-align: right; font-feature-settings: "tnum" 1; }}

.search-wrap {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 999px;
  background: rgba(20,30,44,.85);
  border: 1px solid var(--panel-edge);
  font-size: 11.5px;
}}
.search-wrap input {{
  background: transparent;
  border: none;
  color: var(--ink);
  outline: none;
  width: 130px;
  font-size: 11.5px;
}}
.search-wrap .ico {{ color: var(--muted); }}

.toolbar-right {{
  display: flex;
  gap: 18px;
  align-items: center;
  flex-wrap: wrap;
  justify-content: flex-end;
}}
.action-btn {{
  background: rgba(20,30,44,.85);
  color: var(--ink);
  border: 1px solid var(--panel-edge);
  padding: 6px 12px;
  border-radius: 999px;
  cursor: pointer;
  font-size: 11.5px;
  font-weight: 600;
}}
.action-btn:hover {{ border-color: var(--panel-edge-strong); }}
.action-btn.primary {{
  background: var(--gold);
  color: #1c1305;
  border-color: var(--gold);
}}
.action-btn.primary:hover {{ filter: brightness(1.05); }}

/* ---------- Main layout ---------- */
main {{
  padding: 16px 22px 28px;
  display: grid;
  grid-template-columns: minmax(0, 1fr) 360px;
  gap: 18px;
  align-items: start;
}}
.cards-region {{
  min-width: 0;
}}
.cards-meta {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
  color: var(--muted);
  font-size: 11.5px;
}}
.cards-meta .count strong {{ color: var(--ink); }}
.empty-state {{
  padding: 60px 20px;
  text-align: center;
  color: var(--muted);
  border: 1px dashed var(--panel-edge);
  border-radius: 16px;
  background: rgba(20,30,44,.4);
}}
.empty-state .big {{
  font-size: 24px;
  margin-bottom: 6px;
  color: var(--ink);
}}

.card-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(310px, 1fr));
  gap: 12px;
}}
.cell-card {{
  position: relative;
  background:
    linear-gradient(168deg, rgba(28,40,58,.96), rgba(18,26,38,.96) 70%);
  border: 1px solid var(--panel-edge);
  border-radius: 14px;
  padding: 12px 12px 10px;
  cursor: pointer;
  transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease;
  box-shadow: var(--shadow-soft);
  overflow: hidden;
}}
.cell-card::before {{
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  background:
    radial-gradient(120% 60% at 0% 0%, rgba(255,255,255,.04), transparent 55%),
    linear-gradient(180deg, transparent 70%, rgba(0,0,0,.18));
  border-radius: inherit;
}}
.cell-card:hover {{
  transform: translateY(-2px);
  box-shadow: 0 24px 48px rgba(0,0,0,.45);
  border-color: var(--panel-edge-strong);
}}
.cell-card.h-128 {{ box-shadow: 0 0 0 1px rgba(106,169,255,.06) inset, var(--shadow-soft); }}
.cell-card.h-256 {{ box-shadow: 0 0 0 1px rgba(255,181,71,.06) inset, var(--shadow-soft); }}
.cell-card.h-384 {{ box-shadow: 0 0 0 1px rgba(198,155,255,.06) inset, var(--shadow-soft); }}
.cell-card.selected {{
  border-width: 2px;
  padding: 11px 11px 9px;
  transform: translateY(-2px);
  box-shadow: 0 24px 48px rgba(0,0,0,.55);
}}
.cell-card.selected.h-128 {{ border-color: var(--h128); }}
.cell-card.selected.h-256 {{ border-color: var(--h256); }}
.cell-card.selected.h-384 {{ border-color: var(--h384); }}
.cell-card .h-bar {{
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  width: 4px;
}}
.cell-card.h-128 .h-bar {{ background: linear-gradient(180deg, var(--h128), var(--h128-deep)); }}
.cell-card.h-256 .h-bar {{ background: linear-gradient(180deg, var(--h256), var(--h256-deep)); }}
.cell-card.h-384 .h-bar {{ background: linear-gradient(180deg, var(--h384), var(--h384-deep)); }}

.card-head {{
  display: grid;
  grid-template-columns: auto 1fr auto;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
  position: relative;
}}
.h-badge {{
  display: grid;
  place-items: center;
  width: 50px;
  height: 50px;
  border-radius: 12px;
  font-weight: 800;
  font-size: 16px;
  letter-spacing: 0.04em;
  color: #0a1018;
  box-shadow: 0 4px 12px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.35);
  position: relative;
}}
.h-badge .h-num {{ font-size: 17px; line-height: 1; font-weight: 800; }}
.h-badge .h-tag {{ font-size: 8.5px; line-height: 1; opacity: 0.78; margin-top: 3px; letter-spacing: 0.12em; font-weight: 700; }}
.h-badge.h-128 {{ background: linear-gradient(160deg, #84baff, var(--h128) 50%, var(--h128-deep)); }}
.h-badge.h-256 {{ background: linear-gradient(160deg, #ffc875, var(--h256) 50%, var(--h256-deep)); }}
.h-badge.h-384 {{ background: linear-gradient(160deg, #d8b8ff, var(--h384) 50%, var(--h384-deep)); }}

.card-id {{
  display: flex;
  flex-direction: column;
  min-width: 0;
}}
.card-id .cell-key {{
  font-size: 17px;
  font-weight: 800;
  letter-spacing: 0.01em;
  color: var(--ink);
  font-feature-settings: "tnum" 1;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.card-id .cell-key .dim {{ color: var(--muted); font-weight: 600; }}
.card-id .sub {{
  font-size: 10.5px;
  color: var(--muted);
  margin-top: 2px;
  display: flex;
  gap: 8px;
  align-items: center;
}}
.card-id .sub .rank {{
  display: inline-flex;
  align-items: center;
  gap: 3px;
  padding: 1px 6px;
  border-radius: 6px;
  background: rgba(244,201,93,.12);
  color: var(--gold);
  font-weight: 700;
  font-size: 10px;
}}

.flag-cluster {{
  display: flex;
  flex-direction: column;
  gap: 3px;
  align-items: flex-end;
}}
.flag {{
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 7px;
  border-radius: 999px;
  font-size: 9.5px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  border: 1px solid transparent;
}}
.flag .dot {{
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: currentColor;
}}
.flag.split {{ background: var(--rose-soft); color: var(--rose); border-color: #ff6b8a55; }}
.flag.sample {{ background: var(--amber-soft); color: var(--amber); border-color: #ffb54755; }}
.flag.retire {{ background: rgba(150,160,180,.18); color: #a9b4c4; border-color: #6f7a8b66; }}
.flag.branch {{ background: rgba(56,182,194,.18); color: var(--teal); border-color: #38b6c266; }}

.bars {{
  display: grid;
  gap: 6px;
  margin-bottom: 8px;
}}
.bar-row {{
  display: grid;
  grid-template-columns: 88px 1fr 56px;
  gap: 8px;
  align-items: center;
  font-size: 10.5px;
}}
.bar-row .label {{
  color: var(--muted);
  font-weight: 600;
  font-size: 10.5px;
  letter-spacing: 0.02em;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.bar-row .label .icon {{
  display: inline-block;
  width: 10px;
  margin-right: 3px;
  text-align: center;
  color: var(--muted-dim);
}}
.bar-track {{
  position: relative;
  height: 9px;
  border-radius: 999px;
  background: var(--bar-track);
  border: 1px solid var(--bar-track-edge);
  overflow: hidden;
}}
.bar-fill {{
  position: absolute;
  inset: 0 auto 0 0;
  width: 0%;
  border-radius: inherit;
  transition: width .25s ease;
  background: linear-gradient(90deg, var(--good-1), var(--good-2));
}}
.bar-fill.warm {{ background: linear-gradient(90deg, var(--warn-1), var(--warn-2)); }}
.bar-fill.bad {{ background: linear-gradient(90deg, var(--bad-1), var(--bad-2)); }}
.bar-fill.cool {{ background: linear-gradient(90deg, #2a6fd1, #6aa9ff); }}
.bar-fill.gold {{ background: linear-gradient(90deg, #b88210, var(--gold)); }}
.bar-fill.violet {{ background: linear-gradient(90deg, #6b3fb1, var(--h384)); }}
.bar-fill.flip-warn {{ background: linear-gradient(90deg, var(--good-2), var(--warn-2), var(--bad-2)); }}
.bar-fill::after {{
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(180deg, rgba(255,255,255,.18), transparent 60%);
}}
.bar-track .midline {{
  position: absolute;
  top: -1px;
  bottom: -1px;
  width: 1px;
  background: rgba(255,255,255,.2);
  pointer-events: none;
}}
.bar-row .val {{
  color: var(--ink);
  text-align: right;
  font-weight: 700;
  font-feature-settings: "tnum" 1;
  font-size: 10.5px;
  white-space: nowrap;
}}
.bar-row .val.neg {{ color: var(--rose); }}
.bar-row .val.pos {{ color: var(--mint); }}

.card-foot {{
  margin-top: 6px;
  padding-top: 7px;
  border-top: 1px dashed var(--panel-edge);
  display: flex;
  justify-content: space-between;
  font-size: 10px;
  color: var(--muted);
  letter-spacing: 0.02em;
}}
.card-foot .stat {{ display: inline-flex; gap: 4px; align-items: baseline; }}
.card-foot .stat .v {{ color: var(--ink-soft); font-weight: 700; font-feature-settings: "tnum" 1; }}
.card-foot .pri {{
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 1px 7px;
  border-radius: 999px;
  background: rgba(244,201,93,.1);
  color: var(--gold);
  font-weight: 700;
  font-size: 10px;
}}

/* ---------- Sidebar ---------- */
aside.side {{
  display: flex;
  flex-direction: column;
  gap: 14px;
  position: sticky;
  top: 132px;
}}
.side-card {{
  background: linear-gradient(180deg, rgba(24,36,51,.96), rgba(18,28,40,.96));
  border: 1px solid var(--panel-edge);
  border-radius: 14px;
  overflow: hidden;
  box-shadow: var(--shadow-soft);
}}
.side-card h3 {{
  margin: 0;
  padding: 10px 12px;
  font-size: 12px;
  font-weight: 700;
  color: var(--ink-soft);
  border-bottom: 1px solid var(--panel-edge);
  display: flex;
  justify-content: space-between;
  align-items: center;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}}
.side-card h3 .hint {{
  color: var(--muted-dim);
  font-size: 10px;
  font-weight: 500;
  text-transform: none;
  letter-spacing: 0;
}}
.svg-wrap {{
  position: relative;
  background:
    radial-gradient(60% 60% at 50% 50%, rgba(106,169,255,.05), transparent 70%),
    #0c1320;
}}
.svg-wrap svg {{
  display: block;
  width: 100%;
  height: 240px;
}}
.svg-wrap.atlas svg {{ height: 220px; }}
.tip {{
  position: absolute;
  pointer-events: none;
  background: rgba(8,14,22,.97);
  border: 1px solid var(--panel-edge-strong);
  border-radius: 6px;
  padding: 6px 8px;
  font-size: 10.5px;
  color: var(--ink);
  z-index: 8;
  white-space: nowrap;
  box-shadow: 0 6px 16px rgba(0,0,0,.5);
  display: none;
  font-feature-settings: "tnum" 1;
}}
.tip .name {{ font-weight: 800; margin-bottom: 1px; }}
.tip .row {{ color: var(--muted); }}
.tip .row b {{ color: var(--ink); font-weight: 700; }}
.svg-edge {{ stroke: rgba(170,200,220,.16); stroke-width: 1; }}
.svg-edge.hot {{ stroke: rgba(244,201,93,.55); stroke-width: 1.4; }}
.svg-node {{ stroke-width: 1.4; cursor: pointer; transition: r .12s ease, stroke-width .12s ease; }}
.svg-node.h-128 {{ stroke: rgba(106,169,255,.7); }}
.svg-node.h-256 {{ stroke: rgba(255,181,71,.7); }}
.svg-node.h-384 {{ stroke: rgba(198,155,255,.7); }}
.svg-node.flag-split {{ stroke: var(--rose); stroke-width: 2.4; }}
.svg-node.flag-sample {{ stroke: var(--amber); stroke-width: 2; }}
.svg-node.flag-retire {{ stroke: #a9b4c4; stroke-width: 2; }}
.svg-node.dim {{ opacity: .25; }}
.svg-node.selected {{ stroke: #ffffff; stroke-width: 3.2; filter: drop-shadow(0 0 6px rgba(255,255,255,.6)); }}
.svg-node.match-card {{ stroke: var(--gold); stroke-width: 2.6; filter: drop-shadow(0 0 6px rgba(244,201,93,.6)); }}
.svg-label {{
  font-size: 8.5px;
  fill: #d6e0f0;
  paint-order: stroke;
  stroke: #0a1018;
  stroke-width: 2.5px;
  pointer-events: none;
  font-weight: 600;
}}

.detail-card {{
  padding: 0;
}}
.detail-head {{
  padding: 12px 12px 10px;
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 10px;
  align-items: center;
  border-bottom: 1px solid var(--panel-edge);
}}
.detail-head .h-badge {{ width: 44px; height: 44px; }}
.detail-head .h-badge .h-num {{ font-size: 14px; }}
.detail-head .h-badge .h-tag {{ font-size: 7.5px; }}
.detail-head .title {{
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}}
.detail-head .title .key {{ font-size: 15px; font-weight: 800; color: var(--ink); }}
.detail-head .title .key .dim {{ color: var(--muted); font-weight: 600; }}
.detail-head .title .flags {{ display: flex; gap: 4px; flex-wrap: wrap; }}
.detail-body {{
  padding: 10px 12px 12px;
  display: grid;
  gap: 8px;
}}
.kv-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px 10px;
  font-size: 11px;
}}
.kv-grid .kv-cell {{
  padding: 5px 7px;
  background: rgba(13,22,34,.6);
  border: 1px solid var(--bar-track-edge);
  border-radius: 6px;
}}
.kv-grid .kv-cell .label {{ color: var(--muted); font-size: 9.5px; text-transform: uppercase; letter-spacing: 0.04em; }}
.kv-grid .kv-cell .val {{ color: var(--ink); font-weight: 700; font-feature-settings: "tnum" 1; font-size: 12px; }}
.kv-grid .kv-cell .val.pos {{ color: var(--mint); }}
.kv-grid .kv-cell .val.neg {{ color: var(--rose); }}
.kv-grid .kv-cell.full {{ grid-column: 1 / -1; }}
.kv-grid .kv-cell.action {{
  background: rgba(244,201,93,.08);
  border-color: rgba(244,201,93,.3);
}}

/* ---------- Bottom panel ---------- */
.collapsible {{
  margin: 22px 22px 0;
  background: linear-gradient(180deg, rgba(24,36,51,.92), rgba(18,28,40,.92));
  border: 1px solid var(--panel-edge);
  border-radius: 14px;
  overflow: hidden;
}}
.collapsible-head {{
  padding: 10px 14px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  cursor: pointer;
  user-select: none;
  border-bottom: 1px solid var(--panel-edge);
  background: rgba(15,23,34,.4);
}}
.collapsible-head h3 {{
  margin: 0;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--ink-soft);
}}
.collapsible-head .right {{
  display: flex;
  align-items: center;
  gap: 10px;
  color: var(--muted);
  font-size: 11px;
}}
.collapsible-head .twirl {{
  display: inline-block;
  transition: transform .18s ease;
  font-size: 10px;
  color: var(--muted);
}}
.collapsible.collapsed .twirl {{ transform: rotate(-90deg); }}
.collapsible-body {{
  max-height: 360px;
  overflow: auto;
  transition: max-height .25s ease;
}}
.collapsible.collapsed .collapsible-body {{ max-height: 0; }}
.sample-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 11px;
}}
.sample-table thead {{
  position: sticky;
  top: 0;
  background: rgba(20,30,44,.96);
  backdrop-filter: blur(6px);
}}
.sample-table th {{
  text-align: left;
  font-weight: 700;
  color: var(--muted);
  padding: 8px 10px;
  border-bottom: 1px solid var(--panel-edge);
  font-size: 10.5px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}}
.sample-table td {{
  padding: 6px 10px;
  border-bottom: 1px solid rgba(255,255,255,.04);
  color: var(--ink-soft);
  font-feature-settings: "tnum" 1;
}}
.sample-table tr:hover td {{ background: rgba(244,201,93,.05); }}
.sample-table .basin-y {{ color: var(--mint); font-weight: 700; }}
.sample-table .basin-n {{ color: var(--muted-dim); }}
.sample-table .h-tag {{
  display: inline-block;
  padding: 1px 6px;
  border-radius: 4px;
  font-weight: 700;
  font-size: 10px;
  letter-spacing: 0.04em;
}}
.sample-table .h-tag.h-128 {{ background: var(--h128-soft); color: var(--h128); }}
.sample-table .h-tag.h-256 {{ background: var(--h256-soft); color: var(--h256); }}
.sample-table .h-tag.h-384 {{ background: var(--h384-soft); color: var(--h384); }}

/* ---------- Responsive ---------- */
@media (max-width: 1180px) {{
  main {{ grid-template-columns: 1fr; }}
  aside.side {{ position: static; flex-direction: row; flex-wrap: wrap; }}
  aside.side > .side-card {{ flex: 1 1 320px; min-width: 280px; }}
}}
@media (max-width: 720px) {{
  .toolbar {{ grid-template-columns: 1fr; }}
  .toolbar-right {{ justify-content: flex-start; }}
  .card-grid {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<header class="app-header">
  <div class="header-row">
    <div class="brand">
      <div class="brand-mark">D8</div>
      <div>
        <h1>Cell Atlas / Basin Pokedex</h1>
        <div class="tagline">Phase D8.6 — behavior-sphere archive cells, sortable + filterable</div>
      </div>
    </div>
    <div style="display:flex; gap:10px; align-items:center;">
      <span class="verdict-pill" id="verdictPill">--</span>
      <span class="summary-chip" id="rowsChip">
        <span class="label">cells</span><span class="num" id="totalCells">0</span>
        <span class="sub">/ <span id="filteredCount">0</span> shown</span>
      </span>
    </div>
  </div>
  <div class="summary-strip" id="summaryStrip"></div>
</header>

<div class="toolbar">
  <div class="group">
    <span class="group-label">H</span>
    <span class="chip h-128 active" data-h="128" data-toggle="h"><span class="ico">&#9679;</span>H128</span>
    <span class="chip h-256 active" data-h="256" data-toggle="h"><span class="ico">&#9679;</span>H256</span>
    <span class="chip h-384 active" data-h="384" data-toggle="h"><span class="ico">&#9679;</span>H384</span>
    <span style="width:8px;"></span>
    <span class="group-label">flags</span>
    <span class="chip cand-split" data-flag="split_candidate" data-toggle="flag">split</span>
    <span class="chip cand-sample" data-flag="sample_more_candidate" data-toggle="flag">sample more</span>
    <span class="chip cand-retire" data-flag="retire_candidate" data-toggle="flag">retire</span>
    <span class="chip cand-branch" data-flag="branch_trial_candidate" data-toggle="flag">branch trial</span>
    <span style="width:8px;"></span>
    <span class="slider-wrap">
      <span>min conf</span>
      <input type="range" id="confFilter" min="0" max="1" step="0.05" value="0">
      <span class="val" id="confVal">0.00</span>
    </span>
    <span class="search-wrap">
      <span class="ico">&#x1F50D;</span>
      <input type="text" id="searchBox" placeholder="cell id...">
    </span>
  </div>
  <div class="toolbar-right">
    <div class="group">
      <span class="group-label">sort</span>
      <span class="chip sort-chip active" data-sort="scan_priority">scan_priority<span class="arrow">&#x25BC;</span></span>
      <span class="chip sort-chip" data-sort="branch_trial_score">branch<span class="arrow"></span></span>
      <span class="chip sort-chip" data-sort="split_score">split<span class="arrow"></span></span>
      <span class="chip sort-chip" data-sort="mean_psi">mean_psi<span class="arrow"></span></span>
      <span class="chip sort-chip" data-sort="mean_future_gain">gain<span class="arrow"></span></span>
      <span class="chip sort-chip" data-sort="confidence">conf<span class="arrow"></span></span>
      <span class="chip sort-chip" data-sort="std_future_gain">std<span class="arrow"></span></span>
      <span class="chip sort-chip" data-sort="n_samples">n<span class="arrow"></span></span>
    </div>
    <button class="action-btn" id="resetBtn">reset</button>
    <button class="action-btn primary" id="exportBtn">export</button>
  </div>
</div>

<main>
  <section class="cards-region">
    <div class="cards-meta">
      <div class="count">Showing <strong id="metaShown">0</strong> of <strong id="metaTotal">0</strong> cells &middot; sorted by <strong id="metaSort">--</strong></div>
      <div id="metaTopHint" class="hint" style="color:var(--muted-dim); font-size:11px;"></div>
    </div>
    <div id="cardGrid" class="card-grid"></div>
    <div id="emptyState" class="empty-state" style="display:none;">
      <div class="big">No cells match the filter</div>
      <div>Try toggling more H slices, lowering min-confidence, or clearing flag filters.</div>
    </div>
  </section>

  <aside class="side">
    <div class="side-card">
      <h3>Constellation Graph <span class="hint">kNN, hi-D cosine</span></h3>
      <div class="svg-wrap">
        <svg id="graphView" viewBox="0 0 360 240" role="img" aria-label="kNN constellation"></svg>
        <div class="tip" id="graphTip"></div>
      </div>
    </div>
    <div class="side-card">
      <h3>PCA Atlas <span class="hint">2D projection</span></h3>
      <div class="svg-wrap atlas">
        <svg id="pcaView" viewBox="0 0 360 220" role="img" aria-label="PCA atlas"></svg>
        <div class="tip" id="pcaTip"></div>
      </div>
    </div>
    <div class="side-card detail-card">
      <h3>Selected Cell <span class="hint" id="selCellHint">--</span></h3>
      <div id="detailHead" class="detail-head" style="display:none;"></div>
      <div id="detailBody" class="detail-body"></div>
    </div>
  </aside>
</main>

<div id="samplesPanel" class="collapsible">
  <div class="collapsible-head" id="samplesToggle">
    <h3>Sample States <span class="hint" style="color:var(--muted-dim); font-weight:500; text-transform:none; letter-spacing:0; margin-left:6px;" id="samplesHint">--</span></h3>
    <div class="right">
      <span id="sampleCount">0 rows</span>
      <span class="twirl">&#x25BC;</span>
    </div>
  </div>
  <div class="collapsible-body">
    <table class="sample-table">
      <thead><tr>
        <th>H</th><th>cell</th><th>state_id</th><th>t%</th><th>peak</th><th>gain</th><th>&Psi;</th><th>basin</th>
      </tr></thead>
      <tbody id="sampleRows"></tbody>
    </table>
  </div>
</div>

<script>
const DATA = {data_json};

// ---------------- State ----------------
const STATE = {{
  hSet: new Set([128, 256, 384]),
  flags: new Set(),
  minConf: 0,
  sortKey: "scan_priority",
  sortDir: -1,
  search: "",
  selectedKey: null,
  hoveredKey: null,
}};

const H_COLORS = {{
  128: {{ main: "#6aa9ff", deep: "#2c5fb8", soft: "rgba(106,169,255,.18)" }},
  256: {{ main: "#ffb547", deep: "#c08020", soft: "rgba(255,181,71,.18)" }},
  384: {{ main: "#c69bff", deep: "#7a4cc4", soft: "rgba(198,155,255,.18)" }},
}};

const BARS = [
  {{ key: "confidence", label: "confidence", icon: "&#9679;", mode: "raw01", color: "good", domain: "self" }},
  {{ key: "basin_precision", label: "basin", icon: "&#9670;", mode: "raw01", color: "cool", domain: "self" }},
  {{ key: "scan_priority", label: "scan_pri", icon: "&#9650;", mode: "norm_signed", color: "gold", domain: "h" }},
  {{ key: "mean_psi", label: "mean &Psi;", icon: "&#9678;", mode: "norm", color: "violet", domain: "h" }},
  {{ key: "mean_future_gain", label: "fut_gain", icon: "&#8593;", mode: "norm", color: "good", domain: "h" }},
  {{ key: "std_future_gain", label: "std_gain", icon: "&#9633;", mode: "norm", color: "flip-warn", domain: "h", reverse: true }},
  {{ key: "_n_ratio", label: "n / knee", icon: "&#9881;", mode: "raw01_clamped", color: "warm", domain: "self" }},
];

const SORT_LABELS = {{
  scan_priority: "scan priority",
  branch_trial_score: "branch trial",
  split_score: "split score",
  mean_psi: "mean Ψ",
  mean_future_gain: "mean future gain",
  confidence: "confidence",
  std_future_gain: "std future gain",
  n_samples: "samples",
}};

// ---------------- Helpers ----------------
function fmt(v, digits) {{
  if (v === null || v === undefined || v === "" || (typeof v === "number" && !Number.isFinite(v))) return "--";
  const d = (digits === undefined) ? 3 : digits;
  const n = Number(v);
  if (!Number.isFinite(n)) return String(v);
  if (Math.abs(n) >= 100) return n.toFixed(0);
  if (Math.abs(n) >= 10) return n.toFixed(Math.max(1, d - 1));
  return n.toFixed(d);
}}
function fmtSigned(v, digits) {{
  const n = Number(v);
  if (!Number.isFinite(n)) return "--";
  const s = fmt(Math.abs(n), digits);
  return (n >= 0 ? "+" : "-") + s;
}}
function clamp(v, lo, hi) {{ return Math.max(lo, Math.min(hi, v)); }}
function cellKey(c) {{ return "H" + c.H + "_C" + c.cell_id; }}
function safeNum(v) {{
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}}
function uniqueHs() {{
  const set = new Set();
  for (const c of DATA.cells) set.add(Number(c.H));
  return [...set].sort((a, b) => a - b);
}}

// Pre-compute auxiliary fields and per-H min/max for normalization.
function preprocess() {{
  const hStats = {{}};
  for (const c of DATA.cells) {{
    const h = Number(c.H);
    c._H = h;
    c._key = cellKey(c);
    c._n_ratio = c.knee_H && c.knee_H > 0 ? Math.min(1, Number(c.n_samples || 0) / Number(c.knee_H)) : 0;
    if (!hStats[h]) hStats[h] = {{}};
  }}
  for (const bar of BARS) {{
    if (bar.domain !== "h") continue;
    for (const h of uniqueHs()) {{
      const vals = DATA.cells.filter(c => c._H === h).map(c => safeNum(c[bar.key])).filter(v => v !== null);
      if (vals.length === 0) {{ hStats[h][bar.key] = [0, 1]; continue; }}
      let lo = Math.min(...vals), hi = Math.max(...vals);
      if (Math.abs(hi - lo) < 1e-12) hi = lo + 1;
      hStats[h][bar.key] = [lo, hi];
    }}
  }}
  return hStats;
}}
let H_STATS = preprocess();

function normalizedBar(c, bar) {{
  const v = safeNum(c[bar.key]);
  if (v === null) return {{ pct: 0, raw: null }};
  if (bar.mode === "raw01") return {{ pct: clamp(v, 0, 1) * 100, raw: v }};
  if (bar.mode === "raw01_clamped") return {{ pct: clamp(v, 0, 1) * 100, raw: v }};
  const stats = H_STATS[c._H] && H_STATS[c._H][bar.key];
  if (!stats) return {{ pct: 0, raw: v }};
  const [lo, hi] = stats;
  let t = (v - lo) / (hi - lo);
  t = clamp(t, 0, 1);
  if (bar.reverse) t = 1 - t;
  return {{ pct: t * 100, raw: v, lo, hi }};
}}

function classForFlag(c) {{
  const out = [];
  if (c.split_candidate) out.push("split");
  if (c.sample_more_candidate) out.push("sample");
  if (c.retire_candidate) out.push("retire");
  // We don't have an explicit branch_trial flag in cell row; treat top branch_trial_score as "branch"
  return out;
}}
function isBranchTopForH(c, threshold) {{
  return Number(c.branch_trial_score || 0) >= threshold;
}}
let BRANCH_THRESH = {{ 128: -1e9, 256: -1e9, 384: -1e9 }};
function recomputeBranchThresh() {{
  for (const h of uniqueHs()) {{
    const sorted = DATA.cells.filter(c => c._H === h).map(c => Number(c.branch_trial_score || 0)).sort((a, b) => b - a);
    const cut = Math.min(20, sorted.length - 1);
    BRANCH_THRESH[h] = sorted.length ? sorted[cut] : -1e9;
  }}
}}
recomputeBranchThresh();

function flagsOf(c) {{
  const f = [];
  if (c.split_candidate) f.push({{ name: "split", cls: "split" }});
  if (c.sample_more_candidate) f.push({{ name: "sample+", cls: "sample" }});
  if (c.retire_candidate) f.push({{ name: "retire", cls: "retire" }});
  if (isBranchTopForH(c, BRANCH_THRESH[c._H])) f.push({{ name: "branch", cls: "branch" }});
  return f;
}}
function hasFlag(c, key) {{
  if (key === "branch_trial_candidate") return isBranchTopForH(c, BRANCH_THRESH[c._H]);
  return Boolean(c[key]);
}}

// ---------------- Filtering / Sorting ----------------
function applyFilters() {{
  let out = DATA.cells.filter(c => STATE.hSet.has(c._H));
  if (STATE.flags.size > 0) {{
    out = out.filter(c => {{
      for (const f of STATE.flags) {{
        if (!hasFlag(c, f)) return false;
      }}
      return true;
    }});
  }}
  if (STATE.minConf > 0) {{
    out = out.filter(c => Number(c.confidence || 0) >= STATE.minConf);
  }}
  if (STATE.search) {{
    const q = STATE.search.toLowerCase();
    out = out.filter(c => String(c._key).toLowerCase().includes(q) || String(c.cell_id).includes(q));
  }}
  return out;
}}
function sortCells(cells) {{
  const k = STATE.sortKey;
  const dir = STATE.sortDir;
  return cells.slice().sort((a, b) => {{
    const va = Number(a[k]); const vb = Number(b[k]);
    const aOk = Number.isFinite(va), bOk = Number.isFinite(vb);
    if (!aOk && !bOk) return Number(a.cell_id) - Number(b.cell_id);
    if (!aOk) return 1;
    if (!bOk) return -1;
    if (va === vb) return Number(a.cell_id) - Number(b.cell_id);
    return (va < vb ? 1 : -1) * (dir === -1 ? 1 : -1);
  }});
}}

// ---------------- Rendering: cards ----------------
function renderCards() {{
  const filtered = applyFilters();
  const sorted = sortCells(filtered);
  const grid = document.getElementById("cardGrid");
  const empty = document.getElementById("emptyState");
  document.getElementById("metaShown").textContent = String(sorted.length);
  document.getElementById("metaTotal").textContent = String(DATA.cells.length);
  document.getElementById("filteredCount").textContent = String(sorted.length);
  document.getElementById("metaSort").textContent = (SORT_LABELS[STATE.sortKey] || STATE.sortKey) + " " + (STATE.sortDir === -1 ? "high to low" : "low to high");
  if (sorted.length === 0) {{
    grid.innerHTML = "";
    empty.style.display = "block";
    return;
  }} else {{
    empty.style.display = "none";
  }}
  const html = sorted.map((c, idx) => renderCard(c, idx + 1)).join("");
  grid.innerHTML = html;
  attachCardEvents();
  // Scroll selected into view if needed
  if (STATE.selectedKey) {{
    const el = grid.querySelector('.cell-card[data-key="' + STATE.selectedKey + '"]');
    if (el && idx0(el) > 24) {{ /* skip auto-scroll for far items */ }}
  }}
}}
function idx0(el) {{
  let i = 0; let n = el;
  while ((n = n.previousElementSibling)) i++;
  return i;
}}

function renderCard(c, rank) {{
  const flags = flagsOf(c);
  const hClass = "h-" + c._H;
  const sel = c._key === STATE.selectedKey ? " selected" : "";
  const flagHtml = flags.map(f => '<span class="flag ' + f.cls + '"><span class="dot"></span>' + f.name + '</span>').join("");
  const barsHtml = BARS.map(bar => {{
    const norm = normalizedBar(c, bar);
    const colorCls = bar.color || "good";
    const valStr = bar.key === "_n_ratio"
      ? (Number(c.n_samples || 0) + " / " + Number(c.knee_H || 0))
      : (bar.mode === "norm_signed" ? fmtSigned(norm.raw, 2) : fmt(norm.raw, bar.key === "n_samples" ? 0 : 3));
    const valCls = (bar.mode === "norm_signed" && Number(norm.raw) < 0) ? "neg" : (bar.mode === "norm_signed" && Number(norm.raw) > 0 ? "pos" : "");
    const midline = bar.mode === "norm_signed" ? '<span class="midline" style="left:50%"></span>' : "";
    return '<div class="bar-row">'
      + '<div class="label"><span class="icon">' + bar.icon + '</span>' + bar.label + '</div>'
      + '<div class="bar-track">' + midline + '<div class="bar-fill ' + colorCls + '" style="width:' + norm.pct.toFixed(1) + '%"></div></div>'
      + '<div class="val ' + valCls + '">' + valStr + '</div>'
      + '</div>';
  }}).join("");
  const tagText = c._H === 128 ? "SHORT" : (c._H === 256 ? "MID" : "LONG");
  return ''
    + '<div class="cell-card ' + hClass + sel + '" data-key="' + c._key + '">'
    + '<div class="h-bar"></div>'
    + '<div class="card-head">'
    + '<div class="h-badge ' + hClass + '"><div class="h-num">' + c._H + '</div><div class="h-tag">' + tagText + '</div></div>'
    + '<div class="card-id">'
    + '<div class="cell-key">C' + c.cell_id + ' <span class="dim">/ H' + c._H + '</span></div>'
    + '<div class="sub"><span class="rank">#' + rank + '</span><span>' + (SORT_LABELS[STATE.sortKey] || STATE.sortKey) + ': <b style="color:var(--ink-soft)">' + fmt(c[STATE.sortKey], 3) + '</b></span></div>'
    + '</div>'
    + '<div class="flag-cluster">' + (flagHtml || '<span class="flag" style="background:rgba(120,135,160,.1);color:var(--muted-dim);border-color:#324663aa">stable</span>') + '</div>'
    + '</div>'
    + '<div class="bars">' + barsHtml + '</div>'
    + '<div class="card-foot">'
    + '<span class="stat">src&middot;<span class="v">' + (c.source_diversity || 0) + '</span></span>'
    + '<span class="stat">runs&middot;<span class="v">' + (c.run_diversity || 0) + '</span></span>'
    + '<span class="stat">t&middot;<span class="v">' + fmt(c.mean_time_pct, 2) + '</span></span>'
    + '<span class="stat">pk&middot;<span class="v">' + fmt(c.mean_current_peak, 3) + '</span></span>'
    + '</div>'
    + '</div>';
}}

function attachCardEvents() {{
  document.querySelectorAll(".cell-card").forEach(el => {{
    el.addEventListener("click", () => {{
      selectCell(el.dataset.key, true);
    }});
    el.addEventListener("mouseenter", () => {{
      STATE.hoveredKey = el.dataset.key;
      highlightSvgMatch(el.dataset.key);
    }});
    el.addEventListener("mouseleave", () => {{
      STATE.hoveredKey = null;
      highlightSvgMatch(null);
    }});
  }});
}}

// ---------------- Rendering: SVG views ----------------
function projectPoint(c, w, h, padding) {{
  const px = padding + Number(c.pca_x || 0.5) * (w - 2 * padding);
  const py = padding + Number(c.pca_y || 0.5) * (h - 2 * padding);
  return [px, py];
}}
function projectGraph(cells, w, h, padding) {{
  // For graph view, lay out per H in horizontal bands
  const hs = uniqueHs().filter(hh => STATE.hSet.has(hh));
  if (hs.length === 0) return new Map();
  const points = new Map();
  const bandHeight = (h - 2 * padding) / Math.max(1, hs.length);
  hs.forEach((hh, i) => {{
    const slice = cells.filter(c => c._H === hh);
    if (!slice.length) return;
    // position by pca_x within band, pca_y modulates within slice
    slice.forEach(c => {{
      const px = padding + Number(c.pca_x || 0.5) * (w - 2 * padding);
      const py = padding + i * bandHeight + (0.15 + 0.7 * Number(c.pca_y || 0.5)) * bandHeight;
      points.set(c._key, [px, py, c]);
    }});
  }});
  return points;
}}

function renderGraph() {{
  const svg = document.getElementById("graphView");
  const w = 360, h = 240, pad = 18;
  const filtered = applyFilters();
  const filteredKeys = new Set(filtered.map(c => c._key));
  // We still want to draw all cells in visible H slices, just dim those filtered out
  const visibleCells = DATA.cells.filter(c => STATE.hSet.has(c._H));
  const positions = projectGraph(visibleCells, w, h, pad);
  // edges
  const seen = new Set();
  let edgeHtml = "";
  for (const e of DATA.edges) {{
    if (!STATE.hSet.has(Number(e.H))) continue;
    const a = positions.get("H" + e.H + "_C" + e.source_cell_id);
    const b = positions.get("H" + e.H + "_C" + e.target_cell_id);
    if (!a || !b) continue;
    const k = [e.source_cell_id, e.target_cell_id].sort((x, y) => x - y).join("-") + "_" + e.H;
    if (seen.has(k)) continue;
    seen.add(k);
    const isSel = STATE.selectedKey && (a[2]._key === STATE.selectedKey || b[2]._key === STATE.selectedKey);
    const cls = isSel ? "svg-edge hot" : "svg-edge";
    edgeHtml += '<line class="' + cls + '" x1="' + a[0].toFixed(1) + '" y1="' + a[1].toFixed(1) + '" x2="' + b[0].toFixed(1) + '" y2="' + b[1].toFixed(1) + '"></line>';
  }}
  // band labels
  const hs = uniqueHs().filter(hh => STATE.hSet.has(hh));
  const bandHeight = (h - 2 * pad) / Math.max(1, hs.length);
  let bandHtml = "";
  hs.forEach((hh, i) => {{
    const y = pad + i * bandHeight + bandHeight / 2;
    const col = H_COLORS[hh] ? H_COLORS[hh].main : "#888";
    bandHtml += '<text x="6" y="' + (pad + i * bandHeight + 12).toFixed(1) + '" class="svg-label" fill="' + col + '" style="font-weight:800;font-size:9px">H' + hh + '</text>';
    bandHtml += '<line x1="0" y1="' + (pad + (i + 1) * bandHeight).toFixed(1) + '" x2="' + w + '" y2="' + (pad + (i + 1) * bandHeight).toFixed(1) + '" stroke="rgba(255,255,255,.04)" stroke-width="1"></line>';
  }});
  // nodes
  let nodeHtml = "";
  for (const c of visibleCells) {{
    const p = positions.get(c._key);
    if (!p) continue;
    const dim = !filteredKeys.has(c._key) ? " dim" : "";
    const col = H_COLORS[c._H] ? H_COLORS[c._H].main : "#888";
    const r = 3 + 4 * Math.max(0, Math.min(1, Number(c.confidence || 0)));
    let extraCls = "";
    if (c.split_candidate) extraCls = " flag-split";
    else if (c.sample_more_candidate) extraCls = " flag-sample";
    else if (c.retire_candidate) extraCls = " flag-retire";
    if (c._key === STATE.selectedKey) extraCls = " selected";
    nodeHtml += '<circle class="svg-node h-' + c._H + extraCls + dim + '" data-key="' + c._key + '" cx="' + p[0].toFixed(1) + '" cy="' + p[1].toFixed(1) + '" r="' + r.toFixed(1) + '" fill="' + col + '" fill-opacity="0.62"></circle>';
  }}
  svg.innerHTML = bandHtml + edgeHtml + nodeHtml;
  attachSvgEvents("graphView", "graphTip");
}}

function renderPca() {{
  const svg = document.getElementById("pcaView");
  const w = 360, h = 220, pad = 14;
  const filtered = applyFilters();
  const filteredKeys = new Set(filtered.map(c => c._key));
  const visibleCells = DATA.cells.filter(c => STATE.hSet.has(c._H));
  // background grid
  let bg = "";
  for (let i = 1; i < 4; i++) {{
    const x = pad + (i / 4) * (w - 2 * pad);
    const y = pad + (i / 4) * (h - 2 * pad);
    bg += '<line x1="' + x.toFixed(1) + '" y1="' + pad + '" x2="' + x.toFixed(1) + '" y2="' + (h - pad) + '" stroke="rgba(255,255,255,.04)" stroke-width="1"></line>';
    bg += '<line x1="' + pad + '" y1="' + y.toFixed(1) + '" x2="' + (w - pad) + '" y2="' + y.toFixed(1) + '" stroke="rgba(255,255,255,.04)" stroke-width="1"></line>';
  }}
  let nodes = "";
  for (const c of visibleCells) {{
    const [x, y] = projectPoint(c, w, h, pad);
    const dim = !filteredKeys.has(c._key) ? " dim" : "";
    const col = H_COLORS[c._H] ? H_COLORS[c._H].main : "#888";
    const r = 3.5 + 3.5 * Math.max(0, Math.min(1, Number(c.confidence || 0)));
    let extraCls = "";
    if (c.split_candidate) extraCls = " flag-split";
    else if (c.sample_more_candidate) extraCls = " flag-sample";
    else if (c.retire_candidate) extraCls = " flag-retire";
    if (c._key === STATE.selectedKey) extraCls = " selected";
    nodes += '<circle class="svg-node h-' + c._H + extraCls + dim + '" data-key="' + c._key + '" cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) + '" r="' + r.toFixed(1) + '" fill="' + col + '" fill-opacity="0.66"></circle>';
  }}
  // legend
  let legend = "";
  const hs = uniqueHs();
  hs.forEach((hh, i) => {{
    const col = H_COLORS[hh] ? H_COLORS[hh].main : "#888";
    const x = pad + i * 50;
    legend += '<circle cx="' + (x + 6) + '" cy="' + (h - 6) + '" r="3.5" fill="' + col + '"></circle>';
    legend += '<text class="svg-label" x="' + (x + 13) + '" y="' + (h - 3) + '">H' + hh + '</text>';
  }});
  svg.innerHTML = bg + nodes + legend;
  attachSvgEvents("pcaView", "pcaTip");
}}

function attachSvgEvents(svgId, tipId) {{
  const svg = document.getElementById(svgId);
  const tip = document.getElementById(tipId);
  svg.querySelectorAll(".svg-node").forEach(n => {{
    n.addEventListener("mouseenter", e => {{
      const key = n.dataset.key;
      const c = DATA.cells.find(x => x._key === key);
      if (!c) return;
      tip.style.display = "block";
      tip.innerHTML = '<div class="name" style="color:' + (H_COLORS[c._H] ? H_COLORS[c._H].main : "#fff") + '">H' + c._H + ' / C' + c.cell_id + '</div>'
        + '<div class="row">conf <b>' + fmt(c.confidence, 2) + '</b> &nbsp; basin <b>' + fmt(c.basin_precision, 2) + '</b></div>'
        + '<div class="row">&Psi; <b>' + fmt(c.mean_psi, 3) + '</b> &nbsp; gain <b>' + fmt(c.mean_future_gain, 3) + '</b></div>'
        + '<div class="row">scan_pri <b>' + fmtSigned(c.scan_priority, 2) + '</b> &nbsp; n <b>' + c.n_samples + '/' + c.knee_H + '</b></div>';
      highlightSvgMatch(key);
      const card = document.querySelector('.cell-card[data-key="' + key + '"]');
      if (card) card.classList.add("hover-from-svg");
    }});
    n.addEventListener("mousemove", e => {{
      const rect = svg.getBoundingClientRect();
      const wrapRect = svg.parentElement.getBoundingClientRect();
      const x = e.clientX - wrapRect.left + 12;
      const y = e.clientY - wrapRect.top + 12;
      tip.style.left = Math.min(x, wrapRect.width - 180) + "px";
      tip.style.top = Math.min(y, wrapRect.height - 80) + "px";
    }});
    n.addEventListener("mouseleave", () => {{
      tip.style.display = "none";
      highlightSvgMatch(null);
      document.querySelectorAll(".cell-card.hover-from-svg").forEach(el => el.classList.remove("hover-from-svg"));
    }});
    n.addEventListener("click", () => selectCell(n.dataset.key, true));
  }});
}}

function highlightSvgMatch(key) {{
  document.querySelectorAll(".svg-node").forEach(n => {{
    n.classList.remove("match-card");
  }});
  if (!key) return;
  document.querySelectorAll('.svg-node[data-key="' + key + '"]').forEach(n => {{
    n.classList.add("match-card");
  }});
}}

// ---------------- Selection / details ----------------
function selectCell(key, sticky) {{
  STATE.selectedKey = key;
  renderCards();
  renderGraph();
  renderPca();
  renderDetails();
  renderSamples();
  if (sticky) {{
    const card = document.querySelector('.cell-card[data-key="' + key + '"]');
    if (card) card.scrollIntoView({{ block: "nearest", behavior: "smooth" }});
  }}
}}

function renderDetails() {{
  const head = document.getElementById("detailHead");
  const body = document.getElementById("detailBody");
  const hint = document.getElementById("selCellHint");
  const c = DATA.cells.find(x => x._key === STATE.selectedKey);
  if (!c) {{
    head.style.display = "none";
    body.innerHTML = '<div style="color:var(--muted-dim); padding:14px 0; text-align:center;">click a card to inspect</div>';
    hint.textContent = "--";
    return;
  }}
  hint.textContent = c._key;
  const flags = flagsOf(c);
  const flagHtml = flags.map(f => '<span class="flag ' + f.cls + '"><span class="dot"></span>' + f.name + '</span>').join("");
  const tagText = c._H === 128 ? "SHORT" : (c._H === 256 ? "MID" : "LONG");
  head.style.display = "grid";
  head.innerHTML = ''
    + '<div class="h-badge h-' + c._H + '"><div class="h-num">' + c._H + '</div><div class="h-tag">' + tagText + '</div></div>'
    + '<div class="title">'
    + '<div class="key">C' + c.cell_id + ' <span class="dim">/ H' + c._H + '</span></div>'
    + '<div class="flags">' + (flagHtml || '<span class="flag" style="background:rgba(120,135,160,.1);color:var(--muted-dim);border-color:#324663aa">stable</span>') + '</div>'
    + '</div>';
  const cells = [
    {{ label: "n / knee", val: (c.n_samples + " / " + c.knee_H), cls: "" }},
    {{ label: "confidence", val: fmt(c.confidence, 3), cls: "" }},
    {{ label: "scan_priority", val: fmtSigned(c.scan_priority, 3), cls: Number(c.scan_priority) >= 0 ? "pos" : "neg" }},
    {{ label: "split_score", val: fmtSigned(c.split_score, 3), cls: "" }},
    {{ label: "branch_score", val: fmtSigned(c.branch_trial_score, 3), cls: "" }},
    {{ label: "retire_score", val: fmtSigned(c.retire_score, 3), cls: "" }},
    {{ label: "mean Ψ", val: fmt(c.mean_psi, 4), cls: "" }},
    {{ label: "median Ψ", val: fmt(c.median_psi, 4), cls: "" }},
    {{ label: "mean future gain", val: fmt(c.mean_future_gain, 4), cls: "" }},
    {{ label: "median gain", val: fmt(c.median_future_gain, 4), cls: "" }},
    {{ label: "std gain", val: fmt(c.std_future_gain, 4), cls: "" }},
    {{ label: "iqr gain", val: fmt(c.iqr_future_gain, 4), cls: "" }},
    {{ label: "basin precision", val: fmt(c.basin_precision, 3), cls: "" }},
    {{ label: "mean peak", val: fmt(c.mean_current_peak, 3), cls: "" }},
    {{ label: "mean t%", val: fmt(c.mean_time_pct, 2), cls: "" }},
    {{ label: "n_samples", val: String(c.n_samples), cls: "" }},
    {{ label: "src diversity", val: String(c.source_diversity || 0), cls: "" }},
    {{ label: "run diversity", val: String(c.run_diversity || 0), cls: "" }},
  ];
  let html = cells.map(cell => ''
    + '<div class="kv-cell">'
    + '<div class="label">' + cell.label + '</div>'
    + '<div class="val ' + cell.cls + '">' + cell.val + '</div>'
    + '</div>').join("");
  // Action recommendation
  let action = "no action — stable";
  let actionCls = "";
  if (c.split_candidate) {{ action = "SPLIT — variance is high, refine cell partition"; actionCls = "neg"; }}
  else if (c.sample_more_candidate) {{ action = "SAMPLE MORE — promising but undersampled"; actionCls = "pos"; }}
  else if (c.retire_candidate) {{ action = "RETIRE — saturated and low yield"; actionCls = ""; }}
  else if (isBranchTopForH(c, BRANCH_THRESH[c._H])) {{ action = "BRANCH TRIAL — strong, well-confident"; actionCls = "pos"; }}
  html += '<div class="kv-cell action full"><div class="label">recommended</div><div class="val ' + actionCls + '">' + action + '</div></div>';
  body.innerHTML = '<div class="kv-grid">' + html + '</div>';
}}

function renderSamples() {{
  const tbody = document.getElementById("sampleRows");
  const c = DATA.cells.find(x => x._key === STATE.selectedKey);
  const hint = document.getElementById("samplesHint");
  let rows = [];
  if (c) {{
    rows = DATA.samples.filter(s => Number(s.H) === c._H && Number(s.cell_id) === Number(c.cell_id));
    hint.textContent = "for " + c._key;
  }} else {{
    rows = DATA.samples.slice(0, 50);
    hint.textContent = "first 50 (no cell selected)";
  }}
  document.getElementById("sampleCount").textContent = rows.length + " rows";
  if (rows.length === 0) {{
    tbody.innerHTML = '<tr><td colspan="8" style="color:var(--muted-dim); padding: 16px; text-align:center;">no samples for this cell</td></tr>';
    return;
  }}
  tbody.innerHTML = rows.map(s => {{
    const sid = String(s.state_id || "");
    const sidShort = sid.length > 22 ? sid.slice(0, 10) + "..." + sid.slice(-8) : sid;
    const basin = Number(s.basin_hit || 0) > 0;
    return '<tr>'
      + '<td><span class="h-tag h-' + s.H + '">' + s.H + '</span></td>'
      + '<td>C' + s.cell_id + '</td>'
      + '<td title="' + sid + '" style="font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:10.5px;">' + sidShort + '</td>'
      + '<td>' + fmt(s.time_pct, 2) + '</td>'
      + '<td>' + fmt(s.current_peak, 3) + '</td>'
      + '<td>' + fmt(s.future_gain_final, 4) + '</td>'
      + '<td>' + fmt(s.psi_pred, 4) + '</td>'
      + '<td class="' + (basin ? "basin-y" : "basin-n") + '">' + (basin ? "&#x2714;" : "&#xb7;") + '</td>'
      + '</tr>';
  }}).join("");
}}

// ---------------- Header / summary strip ----------------
function renderSummaryStrip() {{
  const strip = document.getElementById("summaryStrip");
  const verdict = (DATA.summary && DATA.summary.verdict) || "--";
  const pill = document.getElementById("verdictPill");
  pill.textContent = verdict;
  if (verdict.includes("READY")) pill.classList.remove("bad", "warn");
  else if (verdict.includes("GAP")) {{ pill.classList.remove("bad"); pill.classList.add("warn"); }}
  else if (verdict.includes("FAIL")) {{ pill.classList.remove("warn"); pill.classList.add("bad"); }}

  const cov = (DATA.summary && DATA.summary.coverage) || {{}};
  const cands = (DATA.summary && DATA.summary.candidate_counts) || {{}};
  const knees = cov.knee_by_H || {{}};
  const hs = uniqueHs();
  let chips = "";
  hs.forEach(h => {{
    const cnt = DATA.cells.filter(c => c._H === h).length;
    const knee = knees[String(h)] || knees[h] || "?";
    const col = H_COLORS[h] ? H_COLORS[h].main : "#888";
    chips += '<span class="summary-chip">'
      + '<span class="h-dot" style="background:' + col + '"></span>'
      + '<span class="label">H' + h + '</span>'
      + '<span class="num">' + cnt + '</span>'
      + '<span class="sub">cells &middot; knee ' + knee + '</span>'
      + '</span>';
  }});
  const candKeys = [
    ["sample_more_candidates", "sample+", "var(--amber)"],
    ["split_candidates", "split", "var(--rose)"],
    ["branch_trial_candidates", "branch", "var(--teal)"],
    ["retire_candidates", "retire", "var(--gray-pill)"],
  ];
  candKeys.forEach(([k, label, col]) => {{
    const c = cands[k] || {{ rows: 0 }};
    chips += '<span class="summary-chip">'
      + '<span class="h-dot" style="background:' + col + '"></span>'
      + '<span class="label">' + label + '</span>'
      + '<span class="num">' + (c.rows || 0) + '</span>'
      + '</span>';
  }});
  document.getElementById("totalCells").textContent = String(DATA.cells.length);
  strip.innerHTML = chips;
}}

// ---------------- Wiring ----------------
function bindToolbar() {{
  document.querySelectorAll('.chip[data-toggle="h"]').forEach(el => {{
    el.addEventListener("click", () => {{
      const h = Number(el.dataset.h);
      if (STATE.hSet.has(h)) STATE.hSet.delete(h); else STATE.hSet.add(h);
      el.classList.toggle("active");
      renderAll();
    }});
  }});
  document.querySelectorAll('.chip[data-toggle="flag"]').forEach(el => {{
    el.addEventListener("click", () => {{
      const f = el.dataset.flag;
      if (STATE.flags.has(f)) STATE.flags.delete(f); else STATE.flags.add(f);
      el.classList.toggle("active");
      renderAll();
    }});
  }});
  document.querySelectorAll('.chip.sort-chip').forEach(el => {{
    el.addEventListener("click", () => {{
      const k = el.dataset.sort;
      if (STATE.sortKey === k) {{
        STATE.sortDir = -STATE.sortDir;
      }} else {{
        STATE.sortKey = k;
        STATE.sortDir = -1;
      }}
      // update arrows + active
      document.querySelectorAll(".chip.sort-chip").forEach(c2 => {{
        c2.classList.remove("active");
        const arr = c2.querySelector(".arrow");
        if (arr) arr.innerHTML = "";
      }});
      el.classList.add("active");
      const arr = el.querySelector(".arrow");
      if (arr) arr.innerHTML = STATE.sortDir === -1 ? "&#x25BC;" : "&#x25B2;";
      renderCards();
    }});
  }});
  const conf = document.getElementById("confFilter");
  const confVal = document.getElementById("confVal");
  conf.addEventListener("input", () => {{
    STATE.minConf = Number(conf.value);
    confVal.textContent = STATE.minConf.toFixed(2);
    renderAll();
  }});
  const search = document.getElementById("searchBox");
  search.addEventListener("input", () => {{
    STATE.search = search.value.trim();
    renderCards();
  }});
  document.getElementById("resetBtn").addEventListener("click", () => {{
    STATE.hSet = new Set([128, 256, 384]);
    STATE.flags.clear();
    STATE.minConf = 0;
    STATE.search = "";
    STATE.sortKey = "scan_priority";
    STATE.sortDir = -1;
    document.querySelectorAll('.chip[data-toggle="h"]').forEach(c => c.classList.add("active"));
    document.querySelectorAll('.chip[data-toggle="flag"]').forEach(c => c.classList.remove("active"));
    document.querySelectorAll(".chip.sort-chip").forEach(c => {{
      c.classList.remove("active");
      const arr = c.querySelector(".arrow");
      if (arr) arr.innerHTML = "";
    }});
    const def = document.querySelector('.chip.sort-chip[data-sort="scan_priority"]');
    if (def) {{ def.classList.add("active"); const arr = def.querySelector(".arrow"); if (arr) arr.innerHTML = "&#x25BC;"; }}
    conf.value = "0"; confVal.textContent = "0.00";
    search.value = "";
    renderAll();
  }});
  document.getElementById("exportBtn").addEventListener("click", () => {{
    const filtered = sortCells(applyFilters());
    const payload = {{
      generated_at: new Date().toISOString(),
      filter: {{
        H: [...STATE.hSet],
        flags: [...STATE.flags],
        min_confidence: STATE.minConf,
        search: STATE.search,
        sort_key: STATE.sortKey,
        sort_dir: STATE.sortDir,
      }},
      selected: STATE.selectedKey,
      cells: filtered.map(c => ({{
        H: c.H, cell_id: c.cell_id, atlas_cell_key: c._key,
        n_samples: c.n_samples, knee_H: c.knee_H, confidence: c.confidence,
        scan_priority: c.scan_priority, split_score: c.split_score,
        branch_trial_score: c.branch_trial_score, retire_score: c.retire_score,
        mean_psi: c.mean_psi, mean_future_gain: c.mean_future_gain,
        std_future_gain: c.std_future_gain, basin_precision: c.basin_precision,
        sample_more_candidate: !!c.sample_more_candidate,
        split_candidate: !!c.split_candidate,
        retire_candidate: !!c.retire_candidate,
      }})),
    }};
    const blob = new Blob([JSON.stringify(payload, null, 2)], {{ type: "application/json" }});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "cell_atlas_view.json";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(a.href);
  }});
  // Samples collapsible
  document.getElementById("samplesToggle").addEventListener("click", () => {{
    document.getElementById("samplesPanel").classList.toggle("collapsed");
  }});
}}

function renderAll() {{
  // ensure selected cell is still in filtered set; otherwise pick top
  const filtered = sortCells(applyFilters());
  if (!STATE.selectedKey || !filtered.some(c => c._key === STATE.selectedKey)) {{
    STATE.selectedKey = filtered.length ? filtered[0]._key : null;
  }}
  renderCards();
  renderGraph();
  renderPca();
  renderDetails();
  renderSamples();
}}

function init() {{
  renderSummaryStrip();
  bindToolbar();
  // initial sort indicator
  const def = document.querySelector('.chip.sort-chip[data-sort="scan_priority"]');
  if (def) {{ const arr = def.querySelector(".arrow"); if (arr) arr.innerHTML = "&#x25BC;"; }}
  // initial selection: top scan_priority
  const sorted = sortCells(applyFilters());
  STATE.selectedKey = sorted.length ? sorted[0]._key : null;
  renderAll();
}}
init();
</script>
</body>
</html>
"""

def write_report(path: Path, summary: dict, output_root: Path, candidates: dict[str, pd.DataFrame]) -> None:
    lines = [
        "# Phase D8.6 Cell Atlas / Basin Map Dashboard",
        "",
        f"Verdict: `{summary['verdict']}`",
        "",
        "D8.6 is analysis/visualization only. It does not launch a Rust run, change SAF, change K(H), switch archive parents, or alter acceptance.",
        "",
        "## What The Atlas Shows",
        "",
        "- `Constellation Graph`: cells are connected by k-nearest neighbors in the original high-D behavior-feature space.",
        "- `PCA/SVD Atlas`: a deterministic 2D projection for overview only.",
        "- `Command Grid`: a priority dashboard, not a spatial map.",
        "",
        "The 2D atlas is a visualization/projection, not exact high-D geometry.",
        "",
        "## Geometry Contract",
        "",
        "Cell geometry uses only behavior-core features:",
        "",
        "`stable_rank`, `kernel_rank`, `separation_sp`, `collision_rate`, `f_active`, `unique_predictions`, `edges`, `accept_rate_window`.",
        "",
        "Score/time fields are displayed as metrics but are not used for behavior-space cell assignment.",
        "",
        "## Outputs",
        "",
        f"- HTML dashboard: `{output_root / 'cell_atlas.html'}`",
        f"- Cell table: `{output_root / 'analysis' / 'cell_table.csv'}`",
        f"- Cell neighbors: `{output_root / 'analysis' / 'cell_neighbors.csv'}`",
        f"- Sample-more candidates: `{output_root / 'analysis' / 'sample_more_candidates.csv'}`",
        f"- Split candidates: `{output_root / 'analysis' / 'split_candidates.csv'}`",
        f"- Branch-trial candidates: `{output_root / 'analysis' / 'branch_trial_candidates.csv'}`",
        f"- Retire candidates: `{output_root / 'analysis' / 'retire_candidates.csv'}`",
        "",
        "## Coverage",
        "",
        f"- Input rows: `{summary['coverage']['input_rows']}`",
        f"- Atlas cells: `{summary['coverage']['cell_rows']}`",
        f"- Neighbor rows: `{summary['coverage']['neighbor_rows']}`",
        f"- H values: `{summary['coverage']['H_values']}`",
        f"- Anchor config: `{summary['coverage']['anchor_count']}` anchors, seed `{summary['coverage']['anchor_seed']}`",
        f"- Knees: `{summary['coverage']['knee_by_H']}`",
        "",
        "## Candidate Counts",
        "",
    ]
    for name, df in candidates.items():
        by_h = df.groupby("H").size().to_dict() if not df.empty and "H" in df else {}
        lines.append(f"- `{name}`: `{len(df)}` rows, by H `{by_h}`")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Use this dashboard as an operator-facing atlas: choose cells to sample more, split, retire, or branch-test. It does not prove live improvement by itself; it makes the next live tests less blind.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    analysis_dir = args.out / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.input, args.max_rows, args.basin_delta, args.time_buckets)
    knees = load_knees(args.knee_summary)
    cell_table, neighbors, samples = build_atlas(
        df,
        knees,
        args.anchor_count,
        args.anchor_seed,
        args.knn_k,
        args.sample_limit_per_cell,
    )

    verdict = "D8_CELL_ATLAS_READY"
    if df.empty or cell_table.empty or samples.empty:
        verdict = "D8_CELL_ATLAS_DATA_GAP"
    elif sorted(cell_table["H"].unique().tolist()) != sorted(df["H"].unique().tolist()):
        verdict = "D8_CELL_ATLAS_GEOMETRY_FAIL"

    sample_more = top_candidates(cell_table, "sample_more")
    split = top_candidates(cell_table, "split")
    branch = top_candidates(cell_table, "branch_trial")
    retire = top_candidates(cell_table, "retire")
    candidates = {
        "sample_more_candidates": sample_more,
        "split_candidates": split,
        "branch_trial_candidates": branch,
        "retire_candidates": retire,
    }

    write_csv(analysis_dir / "cell_table.csv", cell_table)
    write_csv(analysis_dir / "cell_neighbors.csv", neighbors)
    write_csv(analysis_dir / "cell_sample_states.csv", samples)
    write_csv(analysis_dir / "sample_more_candidates.csv", sample_more)
    write_csv(analysis_dir / "split_candidates.csv", split)
    write_csv(analysis_dir / "branch_trial_candidates.csv", branch)
    write_csv(analysis_dir / "retire_candidates.csv", retire)

    summary = {
        "verdict": verdict,
        "coverage": {
            "input": str(args.input),
            "knee_summary": str(args.knee_summary),
            "input_rows": int(len(df)),
            "cell_rows": int(len(cell_table)),
            "neighbor_rows": int(len(neighbors)),
            "sample_state_rows": int(len(samples)),
            "H_values": [int(x) for x in sorted(df["H"].unique().tolist())],
            "anchor_count": int(args.anchor_count),
            "anchor_seed": int(args.anchor_seed),
            "knn_k": int(args.knn_k),
            "knee_by_H": {int(k): int(v) for k, v in knees.items()},
            "geometry_features": CORE_FEATURES,
            "score_time_used_for_geometry": False,
        },
        "candidate_counts": {
            name: {
                "rows": int(len(frame)),
                "by_H": {str(k): int(v) for k, v in (frame.groupby("H").size().to_dict() if not frame.empty and "H" in frame else {}).items()},
            }
            for name, frame in candidates.items()
        },
    }
    (analysis_dir / "summary.json").write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")

    html = html_template(
        cell_table.drop(columns=["used_features"], errors="ignore").to_dict(orient="records"),
        neighbors.to_dict(orient="records"),
        samples.to_dict(orient="records"),
        summary,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cell_atlas.html").write_text(html, encoding="utf-8")
    write_report(args.report, summary, args.out, candidates)

    print(json.dumps(json_ready(summary), indent=2))
    return 0 if verdict == "D8_CELL_ATLAS_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())

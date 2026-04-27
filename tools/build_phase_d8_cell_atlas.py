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
<title>Phase D8.6 Cell Atlas / Basin Map</title>
<style>
:root {{
  --bg: #101417;
  --panel: #172026;
  --panel2: #1d2930;
  --ink: #e8f0ee;
  --muted: #9eb0ad;
  --line: #33454d;
  --hot: #f6c85f;
  --cold: #325d88;
  --good: #52d273;
  --bad: #ff5c7a;
  --focus: #ffffff;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
  background: radial-gradient(circle at 15% 0%, #25343a, #101417 42%, #0a0d0f);
  color: var(--ink);
}}
header {{
  padding: 18px 22px 12px;
  border-bottom: 1px solid var(--line);
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 12px;
  align-items: end;
}}
h1 {{ margin: 0; font-size: 22px; letter-spacing: 0.02em; }}
.subtitle {{ color: var(--muted); font-size: 13px; margin-top: 5px; }}
.controls {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  justify-content: flex-end;
}}
select, input, button {{
  background: #0d1215;
  color: var(--ink);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 7px 9px;
  font-size: 12px;
}}
button {{ cursor: pointer; }}
main {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) 360px;
  gap: 14px;
  padding: 14px;
}}
.views {{
  display: grid;
  grid-template-columns: minmax(0, 1.1fr) minmax(0, 0.9fr);
  grid-auto-rows: minmax(360px, auto);
  gap: 14px;
}}
.card {{
  background: linear-gradient(180deg, rgba(29,41,48,.96), rgba(17,23,27,.96));
  border: 1px solid var(--line);
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 20px 55px rgba(0,0,0,.24);
}}
.card h2 {{
  margin: 0;
  padding: 12px 14px;
  font-size: 14px;
  color: #cde0dc;
  border-bottom: 1px solid var(--line);
  display: flex;
  justify-content: space-between;
  gap: 12px;
}}
.hint {{ color: var(--muted); font-size: 11px; font-weight: 400; }}
svg {{ display: block; width: 100%; height: 360px; background: rgba(8,12,14,.45); }}
.grid {{
  grid-column: 1 / -1;
  padding: 10px;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(116px, 1fr));
  gap: 8px;
  max-height: 420px;
  overflow: auto;
}}
.tile {{
  border: 1px solid #2f444b;
  border-radius: 10px;
  padding: 8px;
  min-height: 84px;
  background: #132027;
  color: white;
  cursor: pointer;
  transition: transform .08s ease, border-color .08s ease;
}}
.tile:hover, .tile.selected {{ transform: translateY(-1px); border-color: var(--focus); }}
.tile .id {{ font-weight: 700; font-size: 12px; }}
.tile .metric {{ font-size: 11px; color: rgba(255,255,255,.86); margin-top: 3px; }}
.node {{ cursor: pointer; stroke-width: 1.5; }}
.node.selected {{ stroke: var(--focus); stroke-width: 4; }}
.edge {{ stroke: rgba(180,210,210,.18); stroke-width: 1; }}
.label {{ fill: #eaf5f2; font-size: 9px; pointer-events: none; paint-order: stroke; stroke: #071012; stroke-width: 3px; }}
aside {{
  display: flex;
  flex-direction: column;
  gap: 14px;
}}
.sidebox {{
  background: rgba(23,32,38,.97);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 14px;
}}
.sidebox h3 {{ margin: 0 0 10px; font-size: 14px; }}
.kv {{
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 6px 12px;
  font-size: 12px;
}}
.kv div:nth-child(odd) {{ color: var(--muted); }}
table {{ width: 100%; border-collapse: collapse; font-size: 11px; }}
th, td {{ border-bottom: 1px solid rgba(255,255,255,.08); padding: 5px 4px; text-align: left; }}
th {{ color: var(--muted); font-weight: 600; }}
.samples {{ max-height: 280px; overflow: auto; }}
.pill {{
  display: inline-block;
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 2px 7px;
  color: var(--muted);
  margin-left: 6px;
  font-size: 11px;
}}
.legend {{ display:flex; align-items:center; gap:8px; color:var(--muted); font-size:11px; }}
.bar {{ width: 130px; height: 9px; border-radius: 999px; background: linear-gradient(90deg, #325d88, #d4db6a, #ff6b66); }}
.warn {{ color: #f7c56b; }}
@media (max-width: 1100px) {{
  main, .views {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<header>
  <div>
    <h1>Phase D8.6 Cell Atlas / Basin Map <span class="pill" id="verdictPill"></span></h1>
    <div class="subtitle">Constellation graph is the closest behavior-space map. PCA is a projection. Command grid is a decision dashboard, not geometry.</div>
  </div>
  <div class="controls">
    <label>H <select id="hSelect"></select></label>
    <label>Color <select id="colorSelect"></select></label>
    <label>Size <select id="sizeSelect"></select></label>
    <label>Grid <select id="gridSort"></select></label>
    <label>Min conf <input id="confFilter" type="number" min="0" max="1" step="0.05" value="0"></label>
    <label><input id="labelsToggle" type="checkbox"> labels</label>
    <button id="exportBtn">Export selected</button>
  </div>
</header>
<main>
  <section class="views">
    <div class="card">
      <h2>Constellation Graph <span class="hint">kNN edges from high-D cosine neighbors</span></h2>
      <svg id="graphView" viewBox="0 0 720 360" role="img"></svg>
    </div>
    <div class="card">
      <h2>PCA/SVD Atlas <span class="hint">2D projection, not exact map</span></h2>
      <svg id="pcaView" viewBox="0 0 560 360" role="img"></svg>
    </div>
    <div class="card">
      <h2>Command Grid <span class="hint">priority dashboard, not spatial map</span></h2>
      <div id="gridView" class="grid"></div>
    </div>
  </section>
  <aside>
    <div class="sidebox">
      <h3>Selected Cell</h3>
      <div id="cellDetails" class="kv"></div>
    </div>
    <div class="sidebox">
      <h3>Color Scale</h3>
      <div class="legend"><span id="scaleMin"></span><span class="bar"></span><span id="scaleMax"></span></div>
    </div>
    <div class="sidebox samples">
      <h3>Sample States</h3>
      <table>
        <thead><tr><th>state</th><th>t</th><th>peak</th><th>gain</th><th>Ψ</th></tr></thead>
        <tbody id="sampleRows"></tbody>
      </table>
    </div>
  </aside>
</main>
<script>
const DATA = {data_json};
const METRICS = [
  "mean_future_gain","mean_psi","confidence","std_future_gain",
  "basin_precision","scan_priority","split_score","branch_trial_score",
  "n_samples","mean_current_peak"
];
const GRID_SORTS = ["scan_priority","branch_trial_score","split_score","mean_psi","mean_future_gain","confidence","n_samples"];
let selectedKey = null;

function byH() {{
  const h = document.getElementById("hSelect").value;
  const minConf = parseFloat(document.getElementById("confFilter").value || "0");
  return DATA.cells.filter(c => String(c.H) === String(h) && Number(c.confidence || 0) >= minConf);
}}
function cellKey(c) {{ return `H${{c.H}}_C${{c.cell_id}}`; }}
function fmt(v, digits=4) {{
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "NA";
  const n = Number(v);
  if (Math.abs(n) >= 100) return n.toFixed(0);
  return n.toFixed(digits);
}}
function scale(metric, cells) {{
  const vals = cells.map(c => Number(c[metric])).filter(Number.isFinite);
  if (!vals.length) return [0,1];
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (Math.abs(hi-lo) < 1e-12) hi = lo + 1;
  return [lo, hi];
}}
function colorFor(v, lo, hi) {{
  if (!Number.isFinite(v)) return "#67757b";
  const t = Math.max(0, Math.min(1, (v-lo)/(hi-lo)));
  const r = Math.round(50 + 220*t);
  const g = Math.round(93 + 120*Math.sin(t*Math.PI));
  const b = Math.round(136 - 80*t);
  return `rgb(${{r}},${{g}},${{b}})`;
}}
function sizeFor(c, metric) {{
  const v = Number(c[metric] || 0);
  if (metric === "n_samples") return 5 + Math.min(18, Math.sqrt(Math.max(0, v)) * 1.6);
  if (metric === "confidence") return 6 + 16 * Math.max(0, Math.min(1, v));
  return 6 + 13 * Math.max(0, Math.min(1, Number(c.confidence || 0)));
}}
function mapPoint(c, w, h) {{
  return [40 + Number(c.pca_x || 0.5) * (w - 80), 28 + Number(c.pca_y || 0.5) * (h - 56)];
}}
function renderSvg(id, withEdges) {{
  const svg = document.getElementById(id);
  const cells = byH();
  const hVal = document.getElementById("hSelect").value;
  const colorMetric = document.getElementById("colorSelect").value;
  const sizeMetric = document.getElementById("sizeSelect").value;
  const labels = document.getElementById("labelsToggle").checked;
  const [lo, hi] = scale(colorMetric, cells);
  document.getElementById("scaleMin").textContent = fmt(lo);
  document.getElementById("scaleMax").textContent = fmt(hi);
  const w = Number(svg.viewBox.baseVal.width || 720), h = Number(svg.viewBox.baseVal.height || 360);
  const byCell = new Map(cells.map(c => [Number(c.cell_id), c]));
  let out = "";
  if (withEdges) {{
    const seen = new Set();
    for (const e of DATA.edges.filter(e => String(e.H) === String(hVal))) {{
      const a = byCell.get(Number(e.source_cell_id)), b = byCell.get(Number(e.target_cell_id));
      if (!a || !b) continue;
      const key = [a.cell_id,b.cell_id].sort((x,y)=>x-y).join("-");
      if (seen.has(key)) continue;
      seen.add(key);
      const [x1,y1] = mapPoint(a,w,h), [x2,y2] = mapPoint(b,w,h);
      out += `<line class="edge" x1="${{x1.toFixed(2)}}" y1="${{y1.toFixed(2)}}" x2="${{x2.toFixed(2)}}" y2="${{y2.toFixed(2)}}"></line>`;
    }}
  }}
  for (const c of cells) {{
    const [x,y] = mapPoint(c,w,h);
    const key = cellKey(c);
    const cls = key === selectedKey ? "node selected" : "node";
    const stroke = c.split_candidate ? "#ff5c7a" : (c.sample_more_candidate ? "#f6c85f" : "#11191d");
    out += `<circle class="${{cls}}" data-key="${{key}}" cx="${{x.toFixed(2)}}" cy="${{y.toFixed(2)}}" r="${{sizeFor(c,sizeMetric).toFixed(2)}}" fill="${{colorFor(Number(c[colorMetric]),lo,hi)}}" fill-opacity="${{Math.max(.25, Number(c.confidence || 0)).toFixed(2)}}" stroke="${{stroke}}"></circle>`;
    if (labels) out += `<text class="label" x="${{(x+8).toFixed(2)}}" y="${{(y-7).toFixed(2)}}">${{c.cell_id}}</text>`;
  }}
  svg.innerHTML = out;
  svg.querySelectorAll(".node").forEach(n => {{
    n.addEventListener("mouseenter", () => selectCell(n.dataset.key, false));
    n.addEventListener("click", () => selectCell(n.dataset.key, true));
  }});
}}
function renderGrid() {{
  const grid = document.getElementById("gridView");
  const sort = document.getElementById("gridSort").value;
  const colorMetric = document.getElementById("colorSelect").value;
  const cells = byH().slice().sort((a,b) => Number(b[sort] || 0) - Number(a[sort] || 0) || Number(a.cell_id)-Number(b.cell_id));
  const [lo, hi] = scale(colorMetric, cells);
  grid.innerHTML = cells.map(c => {{
    const key = cellKey(c);
    const bg = colorFor(Number(c[colorMetric]), lo, hi);
    const border = c.split_candidate ? "#ff5c7a" : (c.sample_more_candidate ? "#f6c85f" : "#2f444b");
    return `<div class="tile ${{key === selectedKey ? "selected" : ""}}" data-key="${{key}}" style="background:linear-gradient(135deg, ${{bg}}, #142027 80%); border-color:${{border}}; opacity:${{Math.max(.42, Number(c.confidence || 0)).toFixed(2)}}"><div class="id">H${{c.H}} / C${{c.cell_id}}</div><div class="metric">n ${{c.n_samples}}/${{c.knee_H}} conf ${{fmt(c.confidence,2)}}</div><div class="metric">gain ${{fmt(c.mean_future_gain)}} std ${{fmt(c.std_future_gain)}}</div><div class="metric">Ψ ${{fmt(c.mean_psi)}} basin ${{fmt(c.basin_precision,2)}}</div></div>`;
  }}).join("");
  grid.querySelectorAll(".tile").forEach(t => {{
    t.addEventListener("mouseenter", () => selectCell(t.dataset.key, false));
    t.addEventListener("click", () => selectCell(t.dataset.key, true));
  }});
}}
function renderDetails() {{
  const cells = DATA.cells;
  const c = cells.find(x => cellKey(x) === selectedKey) || byH()[0];
  if (!c) return;
  selectedKey = cellKey(c);
  const detailKeys = [
    "H","cell_id","n_samples","knee_H","confidence","mean_future_gain",
    "median_future_gain","std_future_gain","iqr_future_gain","mean_psi",
    "median_psi","basin_precision","mean_current_peak","mean_time_pct",
    "source_diversity","run_diversity","scan_priority","split_score",
    "branch_trial_score","retire_score","sample_more_candidate",
    "split_candidate","retire_candidate"
  ];
  document.getElementById("cellDetails").innerHTML = detailKeys.map(k => `<div>${{k}}</div><div>${{typeof c[k] === "number" ? fmt(c[k]) : c[k]}}</div>`).join("");
  const samples = DATA.samples.filter(s => String(s.H) === String(c.H) && String(s.cell_id) === String(c.cell_id)).slice(0, 20);
  document.getElementById("sampleRows").innerHTML = samples.map(s => `<tr><td title="${{s.state_id || ""}}">${{String(s.state_id || "").slice(-18)}}</td><td>${{fmt(s.time_pct,2)}}</td><td>${{fmt(s.current_peak)}}</td><td>${{fmt(s.future_gain_final)}}</td><td>${{fmt(s.psi_pred)}}</td></tr>`).join("");
}}
function selectCell(key, sticky) {{
  selectedKey = key;
  renderDetails();
  renderSvg("graphView", true);
  renderSvg("pcaView", false);
  renderGrid();
}}
function renderAll() {{
  if (!selectedKey || !byH().some(c => cellKey(c) === selectedKey)) {{
    const first = byH().sort((a,b)=>Number(b.scan_priority||0)-Number(a.scan_priority||0))[0];
    selectedKey = first ? cellKey(first) : null;
  }}
  renderDetails();
  renderSvg("graphView", true);
  renderSvg("pcaView", false);
  renderGrid();
}}
function init() {{
  document.getElementById("verdictPill").textContent = DATA.summary.verdict;
  const hs = [...new Set(DATA.cells.map(c => c.H))].sort((a,b)=>a-b);
  document.getElementById("hSelect").innerHTML = hs.map(h => `<option value="${{h}}">${{h}}</option>`).join("");
  document.getElementById("colorSelect").innerHTML = METRICS.map(m => `<option value="${{m}}">${{m}}</option>`).join("");
  document.getElementById("sizeSelect").innerHTML = ["n_samples","confidence"].map(m => `<option value="${{m}}">${{m}}</option>`).join("");
  document.getElementById("gridSort").innerHTML = GRID_SORTS.map(m => `<option value="${{m}}">${{m}}</option>`).join("");
  for (const id of ["hSelect","colorSelect","sizeSelect","gridSort","confFilter","labelsToggle"]) {{
    document.getElementById(id).addEventListener("change", renderAll);
    document.getElementById(id).addEventListener("input", renderAll);
  }}
  document.getElementById("exportBtn").addEventListener("click", () => {{
    const payload = JSON.stringify({{selected_cell: selectedKey}}, null, 2);
    const blob = new Blob([payload], {{type:"application/json"}});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "selected_cell.json";
    a.click();
    URL.revokeObjectURL(a.href);
  }});
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

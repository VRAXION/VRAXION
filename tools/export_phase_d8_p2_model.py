"""Export a compact D8 P2_PSI_CONF model for live archive-parent tests.

The export intentionally mirrors the existing D8 offline analyzers:
- per-H ridge model for Psi = future_gain_final potential
- per-H robust behavior-sphere scaler
- deterministic spherical anchors
- scan-depth knee thresholds

This file does not run live search.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("output/phase_d8_archive_psi_replay_20260427/analysis/panel_state_dataset.csv")
DEFAULT_KNEE_SUMMARY = Path("output/phase_d8_scan_depth_knee_20260427/analysis/summary.json")
PSI_FEATURES = [
    "edges",
    "unique_predictions",
    "collision_rate",
    "f_active",
    "stable_rank",
    "kernel_rank",
    "separation_sp",
    "accept_rate_window",
    "main_peak_acc",
    "panel_probe_acc",
    "H",
    "jackpot",
    "time_pct",
]
SPHERE_FEATURES = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--knee-summary", type=Path, default=DEFAULT_KNEE_SUMMARY)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--anchor-count", type=int, default=64)
    parser.add_argument("--anchor-seed", type=int, default=23)
    return parser.parse_args()


def fit_ridge(df: pd.DataFrame, features: list[str], target: str, alpha: float) -> dict:
    x = df[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    med = np.nanmedian(x, axis=0)
    inds = np.where(np.isnan(x))
    x[inds] = np.take(med, inds[1])
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0.0] = 1.0
    z = (x - mean) / std
    y = pd.to_numeric(df[target], errors="coerce").fillna(0.0).to_numpy(float)
    x_aug = np.c_[np.ones(len(z)), z]
    reg = np.eye(x_aug.shape[1]) * alpha
    reg[0, 0] = 0.0
    beta = np.linalg.pinv(x_aug.T @ x_aug + reg) @ x_aug.T @ y
    return {
        "intercept": float(beta[0]),
        "beta": [float(v) for v in beta[1:]],
        "median": [float(v) for v in med],
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
    }


def fit_sphere(df: pd.DataFrame, features: list[str], anchor_count: int, anchor_seed: int) -> dict:
    x = df[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    med = np.nanmedian(x, axis=0)
    q25 = np.nanpercentile(x, 25, axis=0)
    q75 = np.nanpercentile(x, 75, axis=0)
    iqr = q75 - q25
    iqr[iqr == 0.0] = 1.0
    rng = np.random.default_rng(anchor_seed)
    anchors = rng.normal(size=(anchor_count, len(features)))
    norm = np.linalg.norm(anchors, axis=1)
    norm[norm == 0.0] = 1.0
    anchors = anchors / norm[:, None]
    return {
        "median": [float(v) for v in med],
        "iqr": [float(v) for v in iqr],
        "anchors": [[float(x) for x in row] for row in anchors],
    }


def load_knees(path: Path) -> dict[int, int]:
    out = dict(DEFAULT_KNEES)
    if not path.exists():
        return out
    data = json.loads(path.read_text(encoding="utf-8"))
    for key, row in data.get("decision", {}).get("by_h", {}).items():
        try:
            h = int(key)
            knee = row.get("median_knee_sample_n")
            if knee is not None:
                out[h] = max(1, int(np.ceil(float(knee))))
        except (TypeError, ValueError):
            continue
    return out


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.input)
    df = df[df["H"].isin([128, 256, 384])].copy()
    df = df.dropna(subset=["future_gain_final"])
    knees = load_knees(args.knee_summary)
    per_h = {}
    for h, sub in df.groupby("H", dropna=False):
        h_int = int(h)
        if len(sub) < 16:
            continue
        for col in PSI_FEATURES + SPHERE_FEATURES:
            if col not in sub:
                sub[col] = np.nan
        per_h[str(h_int)] = {
            "knee_n": int(knees.get(h_int, DEFAULT_KNEES.get(h_int, 8))),
            "psi": fit_ridge(sub, PSI_FEATURES, "future_gain_final", args.ridge_alpha),
            "sphere": fit_sphere(sub, SPHERE_FEATURES, args.anchor_count, args.anchor_seed),
            "training_rows": int(len(sub)),
        }
    model = {
        "schema_version": "d8_p2_model_v1",
        "source_dataset": str(args.input),
        "knee_summary": str(args.knee_summary),
        "ridge_alpha": args.ridge_alpha,
        "anchor_count": args.anchor_count,
        "anchor_seed": args.anchor_seed,
        "psi_features": PSI_FEATURES,
        "sphere_features": SPHERE_FEATURES,
        "per_h": per_h,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(model, indent=2), encoding="utf-8")
    print(f"Wrote: {args.out}")
    print(f"H models: {sorted(per_h)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

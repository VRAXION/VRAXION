"""Phase D8.7 cell-scan delta analyzer.

Compares a D8.7 observer-only spin against the current D8 Cell Atlas.
The spin does not change SAF or restore archive parents; it only logs
P2 cell assignments and Psi predictions for live panel states.

Outputs a compact "what opened / reinforced / cooled" report plus an
augmented panel-state dataset that can be fed back into
build_phase_d8_cell_atlas.py for an after-spin dashboard.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("output/phase_d8_cell_scan_20260428")
DEFAULT_BASE_PANEL = Path("output/phase_d8_archive_psi_replay_20260427/analysis/panel_state_dataset.csv")
DEFAULT_BASE_ATLAS = Path("output/phase_d8_cell_atlas_20260427/analysis/cell_table.csv")
DEFAULT_REPORT = Path("docs/research/PHASE_D8_CELL_SCAN_DELTA.md")
DEFAULT_KNEES = {128: 8, 256: 13, 384: 5}
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--base-panel", type=Path, default=DEFAULT_BASE_PANEL)
    parser.add_argument("--base-atlas", type=Path, default=DEFAULT_BASE_ATLAS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
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
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_state_logs(root: Path, basin_delta: float) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("accepted_state_log.csv")):
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        run_dir = path.parent
        meta = read_json(run_dir / "run_meta.json")
        df["source"] = root.name
        df["run_dir"] = str(run_dir)
        df["fixture"] = meta.get("fixture", "mutual_inhibition")
        df["jackpot"] = meta.get("jackpot", np.nan)
        df["run_meta_arm"] = meta.get("arm", "")
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    numeric = [
        "H",
        "seed",
        "panel_index",
        "step",
        "time_pct",
        "current_peak",
        "panel_probe_acc",
        "accept_rate_window",
        "accepted_window",
        "rejected_window",
        "edges",
        "unique_predictions",
        "collision_rate",
        "f_active",
        "stable_rank",
        "kernel_rank",
        "separation_sp",
        "archive_cell_id",
        "psi_pred",
        "cell_confidence",
        "jackpot",
    ]
    for col in numeric:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out[out["H"].notna()].copy()
    out["H"] = out["H"].astype(int)
    out["archive_cell_id"] = out["archive_cell_id"].astype("Int64")
    sort_cols = [c for c in ["H", "run_id", "panel_index", "state_id"] if c in out]
    out = out.sort_values(sort_cols).reset_index(drop=True)
    # Live future-gain labels are within-run hindsight labels, matching D8.0's
    # offline target semantics but on the short observer spin.
    out["future_peak_final"] = out.groupby("run_id")["current_peak"].transform(lambda s: s.iloc[::-1].cummax().iloc[::-1])
    out["future_gain_final"] = (out["future_peak_final"] - out["current_peak"]).clip(lower=0.0)
    out["basin_hit"] = (out["future_gain_final"] >= basin_delta).astype(int)
    out["psi_pred_seed_cv"] = out["psi_pred"]
    out["psi_atlas"] = out["psi_pred"]
    out["main_peak_acc"] = out["current_peak"]
    out["global_run_id"] = out["source"].astype(str) + "::" + out["run_id"].astype(str)
    return out


def load_base_atlas(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "cell_id" not in df and "archive_cell_id" in df:
        df["cell_id"] = df["archive_cell_id"]
    for col in ["H", "cell_id", "n_samples", "confidence", "mean_psi", "mean_future_gain", "std_future_gain", "basin_precision"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_delta(live: pd.DataFrame, base_atlas: pd.DataFrame) -> pd.DataFrame:
    if live.empty:
        return pd.DataFrame()
    usable = live[live["archive_cell_id"].notna()].copy()
    if usable.empty:
        return pd.DataFrame()
    grouped = (
        usable.groupby(["H", "archive_cell_id"], dropna=False)
        .agg(
            new_samples=("state_id", "size"),
            new_runs=("run_id", "nunique"),
            new_mean_psi=("psi_pred", "mean"),
            new_mean_future_gain=("future_gain_final", "mean"),
            new_max_current_peak=("current_peak", "max"),
            new_mean_current_peak=("current_peak", "mean"),
            new_basin_precision=("basin_hit", "mean"),
            new_mean_confidence=("cell_confidence", "mean"),
            first_panel=("panel_index", "min"),
            last_panel=("panel_index", "max"),
        )
        .reset_index()
        .rename(columns={"archive_cell_id": "cell_id"})
    )
    if base_atlas.empty:
        grouped["baseline_samples"] = 0
        grouped["baseline_confidence"] = 0.0
        grouped["baseline_mean_psi"] = np.nan
        grouped["baseline_mean_future_gain"] = np.nan
        grouped["opened_new_cell"] = True
    else:
        base_cols = [
            "H",
            "cell_id",
            "n_samples",
            "confidence",
            "mean_psi",
            "mean_future_gain",
            "std_future_gain",
            "basin_precision",
            "scan_priority",
            "split_score",
            "branch_trial_score",
        ]
        base = base_atlas[[c for c in base_cols if c in base_atlas]].copy()
        rename = {
            "n_samples": "baseline_samples",
            "confidence": "baseline_confidence",
            "mean_psi": "baseline_mean_psi",
            "mean_future_gain": "baseline_mean_future_gain",
            "std_future_gain": "baseline_std_future_gain",
            "basin_precision": "baseline_basin_precision",
            "scan_priority": "baseline_scan_priority",
            "split_score": "baseline_split_score",
            "branch_trial_score": "baseline_branch_trial_score",
        }
        base = base.rename(columns=rename)
        grouped = grouped.merge(base, on=["H", "cell_id"], how="left")
        grouped["baseline_samples"] = grouped["baseline_samples"].fillna(0).astype(int)
        grouped["baseline_confidence"] = grouped["baseline_confidence"].fillna(0.0)
        grouped["opened_new_cell"] = grouped["baseline_samples"] == 0
    grouped["knee_H"] = grouped["H"].map(DEFAULT_KNEES).fillna(8).astype(int)
    grouped["after_samples"] = grouped["baseline_samples"] + grouped["new_samples"]
    grouped["after_confidence"] = np.minimum(1.0, grouped["after_samples"] / grouped["knee_H"].clip(lower=1))
    grouped["confidence_delta"] = grouped["after_confidence"] - grouped["baseline_confidence"]
    grouped["psi_delta_vs_baseline"] = grouped["new_mean_psi"] - grouped["baseline_mean_psi"]
    grouped["gain_delta_vs_baseline"] = grouped["new_mean_future_gain"] - grouped["baseline_mean_future_gain"]
    grouped["reinforced"] = (
        (~grouped["opened_new_cell"])
        & (grouped["new_mean_future_gain"] >= 0.005)
        & (grouped["new_mean_psi"] >= grouped["baseline_mean_psi"].fillna(-np.inf))
    )
    grouped["cooled"] = (
        (~grouped["opened_new_cell"])
        & (grouped["baseline_mean_psi"].fillna(0.0) > 0.01)
        & (grouped["new_mean_future_gain"] < 0.002)
    )
    return grouped.sort_values(["H", "new_mean_future_gain", "new_mean_psi"], ascending=[True, False, False]).reset_index(drop=True)


def write_augmented_dataset(base_panel: Path, live: pd.DataFrame, out_path: Path) -> None:
    if not base_panel.exists() or live.empty:
        return
    base = pd.read_csv(base_panel)
    live_rows = pd.DataFrame()
    # Keep the D8.0 schema where possible so the existing atlas builder can
    # consume the output as a normal panel-state dataset.
    for col in base.columns:
        if col in live:
            live_rows[col] = live[col]
        elif col == "source":
            live_rows[col] = live["source"]
        elif col == "phase":
            live_rows[col] = live.get("phase", "D8C")
        elif col == "arm":
            live_rows[col] = live.get("arm", live.get("run_meta_arm", "D8C_CELL_SCAN_OBSERVER"))
        elif col == "fixture":
            live_rows[col] = live.get("fixture", "mutual_inhibition")
        elif col == "future_gain_local":
            live_rows[col] = live["future_gain_final"]
        elif col == "future_gain_mid":
            live_rows[col] = live["future_gain_final"]
        elif col == "psi_pred_source":
            live_rows[col] = "d8_cell_scan_live"
        elif col == "psi_pred_seed_cv":
            live_rows[col] = live["psi_pred"]
        elif col == "psi_pred":
            live_rows[col] = live["psi_pred"]
        else:
            live_rows[col] = np.nan
    augmented = pd.concat([base, live_rows], ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(out_path, index=False)


def write_report(path: Path, summary: dict, delta: pd.DataFrame) -> None:
    lines = [
        "# Phase D8.7 Cell Scan Delta",
        "",
        f"Verdict: `{summary['verdict']}`",
        "",
        "D8.7 is an observer-only spin: SAF v1, K(H), strict acceptance, and operator schedule stay unchanged. The P2 model is used only to log archive cell IDs and Psi values.",
        "",
        "## Coverage",
        "",
        f"- Runs: `{summary['coverage']['runs']}`",
        f"- Live panel states: `{summary['coverage']['live_states']}`",
        f"- Visited cells: `{summary['coverage']['visited_cells']}`",
        f"- Opened new cells: `{summary['decision']['opened_new_cells']}`",
        f"- Reinforced cells: `{summary['decision']['reinforced_cells']}`",
        f"- Cooled cells: `{summary['decision']['cooled_cells']}`",
        "",
        "## Top Reinforced / Opened Cells",
        "",
    ]
    if delta.empty:
        lines.append("No cell-level delta rows were available.")
    else:
        top = delta.sort_values(["reinforced", "opened_new_cell", "new_mean_future_gain", "new_mean_psi"], ascending=[False, False, False, False]).head(12)
        lines.extend([
            "| H | cell | new_n | base_n | conf_after | new_psi | new_gain | opened | reinforced | cooled |",
            "|---|---:|---:|---:|---:|---:|---:|---|---|---|",
        ])
        for _, row in top.iterrows():
            lines.append(
                f"| {int(row['H'])} | {int(row['cell_id'])} | {int(row['new_samples'])} | "
                f"{int(row['baseline_samples'])} | {row['after_confidence']:.3f} | "
                f"{row['new_mean_psi']:.4f} | {row['new_mean_future_gain']:.4f} | "
                f"{bool(row['opened_new_cell'])} | {bool(row['reinforced'])} | {bool(row['cooled'])} |"
            )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Opened cells are cells visited by the live spin that were absent from the baseline atlas. Reinforced cells are existing atlas cells where the new live panels still show positive future-gain and Psi support. Cooled cells are previously high-Psi cells where this spin produced little follow-on gain.",
        "",
        "This does not prove live branch improvement. It tells us how the live trajectory populated the atlas before we choose split/sample/branch actions.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    analysis_dir = args.root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    live = load_state_logs(args.root, args.basin_delta)
    if live.empty:
        summary = {"verdict": "D8_CELL_SCAN_DATA_GAP", "coverage": {"runs": 0, "live_states": 0, "visited_cells": 0}, "decision": {"opened_new_cells": 0, "reinforced_cells": 0, "cooled_cells": 0}}
        (analysis_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_report(args.report, summary, pd.DataFrame())
        print(json.dumps(summary, indent=2))
        return 2
    base_atlas = load_base_atlas(args.base_atlas)
    delta = build_delta(live, base_atlas)
    opened = delta[delta["opened_new_cell"].astype(bool)].copy() if not delta.empty else pd.DataFrame()
    reinforced = delta[delta["reinforced"].astype(bool)].copy() if not delta.empty else pd.DataFrame()
    cooled = delta[delta["cooled"].astype(bool)].copy() if not delta.empty else pd.DataFrame()
    live.to_csv(analysis_dir / "live_state_log_panel_dataset.csv", index=False)
    delta.to_csv(analysis_dir / "cell_scan_delta.csv", index=False)
    opened.to_csv(analysis_dir / "opened_cells.csv", index=False)
    reinforced.to_csv(analysis_dir / "reinforced_cells.csv", index=False)
    cooled.to_csv(analysis_dir / "cooled_cells.csv", index=False)
    write_augmented_dataset(args.base_panel, live, analysis_dir / "augmented_panel_state_dataset.csv")
    verdict = "D8_CELL_SCAN_DELTA_READY"
    if delta.empty:
        verdict = "D8_CELL_SCAN_DATA_GAP"
    summary = {
        "verdict": verdict,
        "coverage": {
            "root": str(args.root),
            "runs": int(live["run_id"].nunique()),
            "live_states": int(len(live)),
            "visited_cells": int(delta[["H", "cell_id"]].drop_duplicates().shape[0]) if not delta.empty else 0,
            "H_values": [int(v) for v in sorted(live["H"].unique().tolist())],
        },
        "decision": {
            "opened_new_cells": int(len(opened)),
            "reinforced_cells": int(len(reinforced)),
            "cooled_cells": int(len(cooled)),
            "by_H": {
                str(h): {
                    "live_states": int((live["H"] == h).sum()),
                    "visited_cells": int((delta["H"] == h).sum()) if not delta.empty else 0,
                    "opened": int((opened["H"] == h).sum()) if not opened.empty else 0,
                    "reinforced": int((reinforced["H"] == h).sum()) if not reinforced.empty else 0,
                    "cooled": int((cooled["H"] == h).sum()) if not cooled.empty else 0,
                }
                for h in sorted(live["H"].unique())
            },
        },
    }
    (analysis_dir / "summary.json").write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")
    write_report(args.report, summary, delta)
    print(json.dumps(json_ready(summary), indent=2))
    return 0 if verdict == "D8_CELL_SCAN_DELTA_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())

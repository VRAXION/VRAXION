"""Build D9.0d progressive planet renderer state.

Converts D9.0b direct-genome landscape samples into a JSONP `state.js` file:

    window.ATLAS_STATE = {...};

The schema is intentionally stable (`d9.0d-1`) so the renderer can be built in
parallel without coupling to the sampler implementation.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMA_VERSION = "d9.0d-1"
DEFAULT_SAMPLES = Path("output/phase_d9_direct_genome_landscape_20260428/samples.csv")
DEFAULT_SUMMARY = Path("output/phase_d9_direct_genome_landscape_20260428/analysis/summary.json")
DEFAULT_OUT = Path("output/phase_d9_0d_progressive_planet_20260428")
MUTATION_TYPES = ("edge", "threshold", "channel", "polarity")
TILE_STATES = {
    "UNKNOWN",
    "SCOUT",
    "DESERT",
    "CLIFFY",
    "NOISY",
    "PROMISING",
    "CONFIRMED_GOOD",
    "RETIRED",
    "SPLIT_CANDIDATE",
}
CLIFF_DELTA = -0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--lat-bins", type=int, default=32)
    parser.add_argument("--lon-bins", type=int, default=64)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--phase-status", choices=("running", "finished", "stopped"), default="finished")
    parser.add_argument("--scout-eval-len", type=int, default=None)
    parser.add_argument("--confirmed-eval-len", type=int, default=1000)
    parser.add_argument("--scout-target-per-tile", type=int, default=2)
    parser.add_argument("--confirmed-target-per-tile", type=int, default=10)
    parser.add_argument("--cliff-delta", type=float, default=CLIFF_DELTA)
    return parser.parse_args()


def json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_ready(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_ready(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def zscore(values: list[float | None]) -> list[float]:
    arr = np.array([np.nan if v is None else float(v) for v in values], dtype=float)
    valid = np.isfinite(arr)
    if not valid.any():
        return [0.0 for _ in values]
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr))
    if std <= 1e-12 or not math.isfinite(std):
        out = np.zeros_like(arr)
    else:
        out = (arr - mean) / std
    out[~valid] = 0.0
    return [float(v) for v in out]


def read_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def make_sphere_projection(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "direct_genome_distance",
        "edge_edits",
        "threshold_edits",
        "channel_edits",
        "polarity_edits",
        "behavior_distance",
        "edges",
        "panel_probe_acc",
        "unique_predictions",
        "collision_rate",
        "f_active",
        "stable_rank",
        "kernel_rank",
        "separation_sp",
    ]
    existing = [c for c in cols if c in df.columns]
    if not existing:
        raise ValueError("No projection feature columns found in samples.csv")

    x = df[existing].astype(float).replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True)).to_numpy(dtype=float)
    x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-12)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    coords = x @ vt[:3].T
    if coords.shape[1] < 3:
        coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])), constant_values=0.0)
    norm = np.linalg.norm(coords, axis=1, keepdims=True)
    norm[norm <= 1e-12] = 1.0

    out = df.copy()
    out["sx"] = coords[:, 0] / norm[:, 0]
    out["sy"] = coords[:, 1] / norm[:, 0]
    out["sz"] = coords[:, 2] / norm[:, 0]
    out["lat"] = np.arcsin(np.clip(out["sz"], -1.0, 1.0))
    out["lon"] = np.arctan2(out["sy"], out["sx"])
    return out


def assign_tiles(df: pd.DataFrame, lat_bins: int, lon_bins: int) -> pd.DataFrame:
    work = make_sphere_projection(df)
    work["lat_bin"] = np.clip(((work["lat"] + np.pi / 2) / np.pi * lat_bins).astype(int), 0, lat_bins - 1)
    work["lon_bin"] = np.clip(((work["lon"] + np.pi) / (2 * np.pi) * lon_bins).astype(int), 0, lon_bins - 1)
    work["tile_id"] = work["lat_bin"].astype(str) + "_" + work["lon_bin"].astype(str)
    return work


def metric_stats(group: pd.DataFrame, cliff_delta: float) -> dict[str, float | int | None]:
    if group.empty:
        return {
            "n": 0,
            "mean_delta": None,
            "best_delta": None,
            "median_delta": None,
            "std_delta": None,
            "cliff_rate": None,
            "positive_rate": None,
            "mean_behavior_distance": None,
        }
    deltas = group["delta_score"].astype(float)
    return {
        "n": int(len(group)),
        "mean_delta": float(deltas.mean()),
        "best_delta": float(deltas.max()),
        "median_delta": float(deltas.median()),
        "std_delta": float(deltas.std(ddof=0)),
        "cliff_rate": float((deltas <= cliff_delta).mean()),
        "positive_rate": float((deltas > 0.0).mean()),
        "mean_behavior_distance": float(group["behavior_distance"].astype(float).mean())
        if "behavior_distance" in group.columns
        else None,
    }


def per_type_stats(group: pd.DataFrame, cliff_delta: float) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    for mutation_type in MUTATION_TYPES:
        sub = group[group["mutation_type"].astype(str) == mutation_type]
        stats = metric_stats(sub, cliff_delta)
        result[mutation_type] = {
            "n": stats["n"],
            "mean_delta": stats["mean_delta"],
            "best_delta": stats["best_delta"],
            "std_delta": stats["std_delta"],
            "cliff_rate": stats["cliff_rate"],
            "positive_rate": stats["positive_rate"],
        }
    return result


def classify_tile(stats: dict[str, Any], confidence: float, target_n: int) -> str:
    n = int(stats["n"])
    mean_delta = stats["mean_delta"]
    std_delta = stats["std_delta"]
    cliff_rate = stats["cliff_rate"]
    positive_rate = stats["positive_rate"]

    if n <= 0:
        return "UNKNOWN"
    if n >= 5 and std_delta is not None and std_delta > 0.02:
        return "SPLIT_CANDIDATE"
    if n >= 3 and cliff_rate is not None and cliff_rate > 0.65:
        return "CLIFFY"
    if n >= 3 and mean_delta is not None and std_delta is not None and mean_delta <= -0.005 and std_delta < 0.005:
        return "DESERT"
    if n >= 5 and std_delta is not None and std_delta > 0.015:
        return "NOISY"
    if n >= 3 and mean_delta is not None and positive_rate is not None and mean_delta >= 0.0 and positive_rate > 0.10:
        return "PROMISING"
    if n >= 5 and confidence >= 1.0 and mean_delta is not None and mean_delta < 0.0 and positive_rate == 0.0:
        return "RETIRED"
    return "SCOUT"


def action_for_state(state: str) -> str:
    return {
        "UNKNOWN": "scout",
        "SCOUT": "sample_more",
        "DESERT": "retire",
        "CLIFFY": "retire",
        "NOISY": "split",
        "PROMISING": "confirm_expensive",
        "CONFIRMED_GOOD": "branch_candidate",
        "RETIRED": "retired",
        "SPLIT_CANDIDATE": "split",
    }[state]


def dominant_value(group: pd.DataFrame, column: str) -> str | int | None:
    if group.empty or column not in group.columns:
        return None
    mode = group[column].mode()
    if mode.empty:
        return None
    value = mode.iloc[0]
    if isinstance(value, (np.integer,)):
        return int(value)
    return str(value)


def build_tiles(
    samples: pd.DataFrame,
    lat_bins: int,
    lon_bins: int,
    scout_target_n: int,
    confirmed_target_n: int,
    cliff_delta: float,
) -> list[dict[str, Any]]:
    grouped = {str(k): v.copy() for k, v in samples.groupby("tile_id", dropna=False)}
    tiles: list[dict[str, Any]] = []

    for lat_bin in range(lat_bins):
        lat_center = -np.pi / 2 + (lat_bin + 0.5) * np.pi / lat_bins
        for lon_bin in range(lon_bins):
            lon_center = -np.pi + (lon_bin + 0.5) * 2 * np.pi / lon_bins
            tile_id = f"{lat_bin}_{lon_bin}"
            group = grouped.get(tile_id, samples.iloc[0:0])
            stats = metric_stats(group, cliff_delta)
            if "scout_layer" in group.columns:
                n_scout = int((group["scout_layer"].astype(str) == "scout").sum())
                n_confirmed = int((group["scout_layer"].astype(str) == "confirmed").sum())
            else:
                n_scout = int(stats["n"])
                n_confirmed = 0
            target_n = confirmed_target_n if n_confirmed > 0 else scout_target_n
            n_total = n_scout + n_confirmed
            confidence = min(1.0, n_total / max(1, target_n))
            state = classify_tile(stats, confidence, target_n)
            if n_confirmed >= confirmed_target_n and stats["mean_delta"] is not None and stats["mean_delta"] >= 0.0:
                state = "CONFIRMED_GOOD"
            assert state in TILE_STATES

            tile = {
                "tile_id": tile_id,
                "lat_bin": lat_bin,
                "lon_bin": lon_bin,
                "lat_center": float(lat_center),
                "lon_center": float(lon_center),
                "x": float(np.cos(lat_center) * np.cos(lon_center)),
                "y": float(np.cos(lat_center) * np.sin(lon_center)),
                "z": float(np.sin(lat_center)),
                "n_scout": n_scout,
                "n_confirmed": n_confirmed,
                "target_n": target_n,
                "mean_delta": stats["mean_delta"],
                "best_delta": stats["best_delta"],
                "median_delta": stats["median_delta"],
                "std_delta": stats["std_delta"],
                "cliff_rate": stats["cliff_rate"],
                "positive_rate": stats["positive_rate"],
                "mean_behavior_distance": stats["mean_behavior_distance"],
                "confidence": float(confidence),
                "state": state,
                "recommended_action": action_for_state(state),
                "dominant_mutation_type": dominant_value(group, "mutation_type"),
                "dominant_radius": dominant_value(group, "requested_radius"),
                "per_type": per_type_stats(group, cliff_delta),
                "scan_priority": 0.0,
                "split_priority": 0.0,
            }
            tiles.append(tile)

    best_z = zscore([t["best_delta"] for t in tiles])
    std_z = zscore([t["std_delta"] for t in tiles])
    pos_z = zscore([t["positive_rate"] for t in tiles])
    cliff_z = zscore([t["cliff_rate"] for t in tiles])
    behavior_z = zscore([t["mean_behavior_distance"] for t in tiles])

    for i, tile in enumerate(tiles):
        confidence = float(tile["confidence"])
        if tile["state"] == "UNKNOWN":
            tile["scan_priority"] = 0.25
            tile["split_priority"] = 0.0
            continue
        tile["scan_priority"] = float(
            0.35 * best_z[i]
            + 0.25 * (1.0 - confidence)
            + 0.20 * std_z[i]
            + 0.10 * pos_z[i]
            - 0.20 * cliff_z[i]
            - 0.10 * confidence
        )
        tile["split_priority"] = float(std_z[i] + behavior_z[i] + 0.5 * cliff_z[i])

    return tiles


def build_queue(tiles: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    candidates = [
        tile
        for tile in tiles
        if tile["state"] not in {"DESERT", "CLIFFY", "RETIRED", "CONFIRMED_GOOD"}
    ]
    candidates.sort(key=lambda t: (-float(t["scan_priority"]), str(t["tile_id"])))
    queue = []
    for tile in candidates[:limit]:
        mutation_type = tile["dominant_mutation_type"] or "edge"
        if mutation_type not in MUTATION_TYPES:
            mutation_type = "edge"
        queue.append(
            {
                "tile_id": tile["tile_id"],
                "type": mutation_type,
                "priority": float(tile["scan_priority"]),
                "next_action": tile["recommended_action"],
            }
        )
    return queue


def build_state(args: argparse.Namespace) -> dict[str, Any]:
    if not args.samples.exists():
        raise FileNotFoundError(args.samples)
    summary = read_summary(args.summary)
    run_meta = summary.get("run_meta", {})
    df = pd.read_csv(args.samples)
    if "mutation_type" not in df.columns or "delta_score" not in df.columns:
        raise ValueError("samples.csv must include mutation_type and delta_score")

    has_direct_tiles = {"lat_bin", "lon_bin"}.issubset(df.columns) and df["lat_bin"].notna().any()
    if has_direct_tiles:
        projected = df.copy()
        projected["lat_bin"] = projected["lat_bin"].astype(int)
        projected["lon_bin"] = projected["lon_bin"].astype(int)
        projected["tile_id"] = projected["lat_bin"].astype(str) + "_" + projected["lon_bin"].astype(str)
    else:
        projected = assign_tiles(df, args.lat_bins, args.lon_bins)
    tiles = build_tiles(
        projected,
        args.lat_bins,
        args.lon_bins,
        args.scout_target_per_tile,
        args.confirmed_target_per_tile,
        args.cliff_delta,
    )
    queue = build_queue(tiles)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    checkpoint = ""
    checkpoints = run_meta.get("checkpoints")
    if isinstance(checkpoints, list) and checkpoints:
        checkpoint = str(checkpoints[0])
    elif "base_checkpoint" in df.columns and len(df):
        checkpoint = str(df["base_checkpoint"].iloc[0])

    total_tiles = args.lat_bins * args.lon_bins
    tiles_with_data = sum(1 for tile in tiles if tile["n_scout"] > 0 or tile["n_confirmed"] > 0)
    confident = sum(1 for tile in tiles if float(tile["confidence"]) >= 1.0)
    promising = sum(1 for tile in tiles if tile["state"] in {"PROMISING", "CONFIRMED_GOOD"})
    retired = sum(1 for tile in tiles if tile["state"] in {"DESERT", "CLIFFY", "RETIRED"})
    split = sum(1 for tile in tiles if tile["state"] == "SPLIT_CANDIDATE")
    scout_eval_len = args.scout_eval_len if args.scout_eval_len is not None else int(run_meta.get("eval_len", 100))
    run_id = args.run_id or f"d9_0d_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    state = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "source_run_id": str(summary.get("input", args.samples.parent)),
        "generated_at_utc": now,
        "last_updated": now,
        "phase_status": args.phase_status,
        "stop_clock_active": False,
        "source_samples_csv": str(args.samples).replace("\\", "/"),
        "source_summary_json": str(args.summary).replace("\\", "/") if args.summary.exists() else None,
        "config": {
            "H": int(run_meta.get("h", df["H"].iloc[0] if "H" in df.columns and len(df) else 256)),
            "resolution": f"{args.lat_bins}x{args.lon_bins}",
            "lat_bins": args.lat_bins,
            "lon_bins": args.lon_bins,
            "scout_eval_len": scout_eval_len,
            "confirmed_eval_len": args.confirmed_eval_len,
            "scout_target_per_tile": args.scout_target_per_tile,
            "confirmed_target_per_tile": args.confirmed_target_per_tile,
            "checkpoint": checkpoint,
            "score_mode": str(run_meta.get("score_mode", df["score_mode"].iloc[0] if "score_mode" in df.columns and len(df) else "unknown")),
            "tile_assignment": "direct_csv" if has_direct_tiles else "pca_shadow_projection",
        },
        "tiles": tiles,
        "queue": queue,
        "progress": {
            "total_tiles_possible": total_tiles,
            "tiles_with_data": tiles_with_data,
            "coverage": tiles_with_data / total_tiles if total_tiles else 0.0,
            "confident_fraction": confident / total_tiles if total_tiles else 0.0,
            "promising_count": promising,
            "retired_count": retired,
            "split_candidate_count": split,
            "samples_total": int(len(df)),
            "samples_scout": int(len(df)),
            "samples_confirmed": 0,
        },
        "acquisition_weights": {
            "best": 0.35,
            "uncertainty": 0.25,
            "std": 0.20,
            "positive_rate": 0.10,
            "cliff_rate": -0.20,
            "confidence": -0.10,
        },
    }
    return json_ready(state)


def validate_state(state: dict[str, Any]) -> None:
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("schema_version mismatch")
    required_tile_fields = {
        "tile_id",
        "lat_bin",
        "lon_bin",
        "n_scout",
        "n_confirmed",
        "target_n",
        "confidence",
        "state",
        "recommended_action",
        "per_type",
        "scan_priority",
        "split_priority",
    }
    for tile in state["tiles"]:
        missing = required_tile_fields - set(tile)
        if missing:
            raise ValueError(f"tile {tile.get('tile_id')} missing fields: {sorted(missing)}")
        if tile["state"] not in TILE_STATES:
            raise ValueError(f"bad tile state: {tile['state']}")
        if set(tile["per_type"].keys()) != set(MUTATION_TYPES):
            raise ValueError(f"bad per_type keys on tile {tile['tile_id']}")
    for item in state["queue"]:
        if item["type"] not in MUTATION_TYPES:
            raise ValueError(f"bad queue mutation type: {item['type']}")


def write_outputs(state: dict[str, Any], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    state_json = json.dumps(state, indent=2, sort_keys=False)
    (out / "progressive_atlas_state.json").write_text(state_json + "\n", encoding="utf-8")
    (out / "state.js").write_text("window.ATLAS_STATE = " + state_json + ";\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    state = build_state(args)
    validate_state(state)
    write_outputs(state, args.out)
    print(
        json.dumps(
            {
                "schema_version": state["schema_version"],
                "out": str(args.out),
                "tiles": len(state["tiles"]),
                "coverage": state["progress"]["coverage"],
                "queue": len(state["queue"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

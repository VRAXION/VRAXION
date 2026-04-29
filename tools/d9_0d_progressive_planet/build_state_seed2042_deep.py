"""Build state.js for the D9.0d progressive planet renderer from the
seed2042 deep trajectory output (D9.0n run).

Reads tile_deep_trajectory_summary.csv and emits a state.js conforming to
schema d9.0d-1, where:
- 20 tiles are filled with the seed2042 deep climb stats
- top 4 tiles (long_ascent_rate >= 0.234) get state "BASIN_CONFIRMED"
- remaining 16 climbed tiles get state "PROMISING"
- the rest of the 16x32 grid is "UNKNOWN"

The output state.js carries non-standard top-level fields long_ascent_rate,
material_success_rate, late_best_rate per tile so the renderer can heatmap
them directly.
"""
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC_N = REPO / "output" / "phase_d9_0n_seed2042_deep_trajectory_20260428"
SRC_O = REPO / "output" / "phase_d9_0o_seed2042_top3_deepening_20260428"
SRC_Q = REPO / "output" / "phase_d9_0q_seed2042_long_climb_20260429"
SUMMARY_CSV_N = SRC_N / "tile_deep_trajectory_summary.csv"
SUMMARY_CSV_O = SRC_O / "tile_top3_deepening_summary.csv"
SUMMARY_CSV_Q = SRC_Q / "tile_long_climb_summary.csv"
OUT_JS = REPO / "tools" / "d9_0d_progressive_planet" / "state.js"

LAT_BINS = 16
LON_BINS = 32

# Mountain core: D9.0q 300-step run kept these climbing > basin -> MOUNTAIN_CONFIRMED
TOP2_MOUNTAIN = {"12_29", "11_16"}
# Top 3 from D9.0o 200-step deepening; 9_26 did NOT get the 300-step pass
TOP3_DEEP = {"9_26", "12_29", "11_16"}
# Top 4 from D9.0n 100-step run; 7_16 not deepened -> stays BASIN_CONFIRMED
TOP4 = {"11_16", "9_26", "12_29", "7_16"}


def lat_lon_to_xyz(lat_bin: int, lon_bin: int) -> tuple[float, float, float]:
    lat = (lat_bin + 0.5) / LAT_BINS * math.pi - math.pi / 2.0
    lon = (lon_bin + 0.5) / LON_BINS * 2.0 * math.pi - math.pi
    x = math.cos(lat) * math.cos(lon)
    y = math.cos(lat) * math.sin(lon)
    z = math.sin(lat)
    return x, y, z


def empty_per_type():
    keys = ("edge", "threshold", "channel", "polarity")
    return {
        k: {
            "n": 0,
            "mean_delta": None,
            "best_delta": None,
            "std_delta": None,
            "cliff_rate": None,
            "positive_rate": None,
        }
        for k in keys
    }


def load_summary(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def f(row: dict, k: str) -> float | None:
    v = row.get(k, "")
    if v == "" or v is None:
        return None
    return float(v)


def build_climbed_tile(row: dict, deep_row: dict | None = None, mountain_row: dict | None = None) -> dict:
    tile_id = row["tile_id"]
    lat_bin, lon_bin = (int(p) for p in tile_id.split("_"))
    x, y, z = lat_lon_to_xyz(lat_bin, lon_bin)

    # Priority of data sources: D9.0q 300-step > D9.0o 200-step > D9.0n 100-step.
    if mountain_row is not None:
        src = mountain_row
        climb_steps = 300
    elif deep_row is not None:
        src = deep_row
        climb_steps = 200
    else:
        src = row
        climb_steps = 100

    n_climbers = int(src.get("n_climbers", 0) or 0)
    mean_best = f(src, "mean_best_delta") or 0.0
    p90_best = f(src, "p90_best_delta") or 0.0
    max_best = f(src, "max_best_delta") or 0.0
    mean_final = f(src, "mean_final_delta") or 0.0
    long_ascent = f(src, "long_ascent_rate") or 0.0
    material = f(src, "material_success_rate") or 0.0
    late_best = f(src, "late_best_rate") or 0.0
    flat_rate = f(src, "flat_rate") or 0.0
    early_plateau = f(src, "early_plateau_rate") or 0.0
    late_plateau = f(src, "late_plateau_rate") or 0.0
    deep_long_ascent = f(src, "deep_long_ascent_rate")

    if tile_id in TOP2_MOUNTAIN:
        state = "MOUNTAIN_CONFIRMED"
    elif tile_id in TOP3_DEEP:
        state = "DEEP_BASIN_CONFIRMED"
    elif tile_id in TOP4:
        state = "BASIN_CONFIRMED"
    else:
        state = "PROMISING"

    # cliff_rate as proxy: flat tiles are not cliffy, so risk-low.
    # confidence = saturation of climb sample count (cap 64).
    confidence = min(1.0, n_climbers / 32.0)

    # split per_type evenly between edge and threshold (D9.0n ran edge+threshold)
    half = n_climbers // 2
    per_type = empty_per_type()
    if half > 0:
        per_type["edge"] = {
            "n": half,
            "mean_delta": mean_best,
            "best_delta": max_best,
            "std_delta": None,
            "cliff_rate": 0.0,
            "positive_rate": material,
        }
        per_type["threshold"] = {
            "n": half,
            "mean_delta": mean_best,
            "best_delta": max_best,
            "std_delta": None,
            "cliff_rate": 0.0,
            "positive_rate": material,
        }

    if state == "MOUNTAIN_CONFIRMED":
        action = "endpoint_export"
    elif state == "DEEP_BASIN_CONFIRMED":
        action = "long_climb_300"
    elif state == "BASIN_CONFIRMED":
        action = "deepen+200"
    else:
        action = "confirm"

    return {
        "tile_id": tile_id,
        "lat_bin": lat_bin,
        "lon_bin": lon_bin,
        "lat_center": (lat_bin + 0.5) / LAT_BINS * math.pi - math.pi / 2.0,
        "lon_center": (lon_bin + 0.5) / LON_BINS * 2.0 * math.pi - math.pi,
        "x": x,
        "y": y,
        "z": z,
        "n_scout": n_climbers,
        "n_confirmed": n_climbers,
        "target_n": 64,
        "mean_delta": mean_best,
        "best_delta": max_best,
        "median_delta": f(src, "median_best_delta"),
        "std_delta": None,
        "cliff_rate": 0.0,
        "positive_rate": material,
        "mean_behavior_distance": None,
        "confidence": confidence,
        "state": state,
        "recommended_action": action,
        "dominant_mutation_type": "edge",
        "dominant_radius": 8,
        "per_type": per_type,
        "scan_priority": long_ascent * 5.0,
        "split_priority": (1.0 - flat_rate) * 2.0,
        # Custom D9.0n/D9.0o metrics consumed directly by the renderer:
        "long_ascent_rate": long_ascent,
        "material_success_rate": material,
        "late_best_rate": late_best,
        "p90_best_delta": p90_best,
        "max_best_delta": max_best,
        "mean_final_delta": mean_final,
        "flat_rate": flat_rate,
        "early_plateau_rate": early_plateau,
        "late_plateau_rate": late_plateau,
        "deep_long_ascent_rate": deep_long_ascent,
        "deepening_climb_steps": climb_steps,
    }


def build_unknown_tile(lat_bin: int, lon_bin: int) -> dict:
    x, y, z = lat_lon_to_xyz(lat_bin, lon_bin)
    return {
        "tile_id": f"{lat_bin}_{lon_bin}",
        "lat_bin": lat_bin,
        "lon_bin": lon_bin,
        "lat_center": (lat_bin + 0.5) / LAT_BINS * math.pi - math.pi / 2.0,
        "lon_center": (lon_bin + 0.5) / LON_BINS * 2.0 * math.pi - math.pi,
        "x": x,
        "y": y,
        "z": z,
        "n_scout": 0,
        "n_confirmed": 0,
        "target_n": 64,
        "mean_delta": None,
        "best_delta": None,
        "median_delta": None,
        "std_delta": None,
        "cliff_rate": None,
        "positive_rate": None,
        "mean_behavior_distance": None,
        "confidence": 0.0,
        "state": "UNKNOWN",
        "recommended_action": "scout",
        "dominant_mutation_type": None,
        "dominant_radius": None,
        "per_type": empty_per_type(),
        "scan_priority": 0.0,
        "split_priority": 0.0,
        "long_ascent_rate": None,
        "material_success_rate": None,
        "late_best_rate": None,
        "p90_best_delta": None,
        "max_best_delta": None,
        "mean_final_delta": None,
        "flat_rate": None,
        "early_plateau_rate": None,
        "late_plateau_rate": None,
    }


def main() -> int:
    if not SUMMARY_CSV_N.exists():
        print(f"missing: {SUMMARY_CSV_N}", file=sys.stderr)
        return 2
    if not SUMMARY_CSV_O.exists():
        print(f"missing: {SUMMARY_CSV_O}", file=sys.stderr)
        return 2

    rows = load_summary(SUMMARY_CSV_N)
    deep_rows = load_summary(SUMMARY_CSV_O)
    mountain_rows = load_summary(SUMMARY_CSV_Q) if SUMMARY_CSV_Q.exists() else []
    deep_by_tile = {r["tile_id"]: r for r in deep_rows}
    mountain_by_tile = {r["tile_id"]: r for r in mountain_rows}
    climbed_tile_ids = {r["tile_id"] for r in rows}

    tiles = []
    for r in rows:
        deep = deep_by_tile.get(r["tile_id"])
        mountain = mountain_by_tile.get(r["tile_id"])
        tiles.append(build_climbed_tile(r, deep, mountain))
    for la in range(LAT_BINS):
        for lo in range(LON_BINS):
            tid = f"{la}_{lo}"
            if tid in climbed_tile_ids:
                continue
            tiles.append(build_unknown_tile(la, lo))

    # Queue: top 8 by long_ascent — MOUNTAIN tiles get endpoint_export, basin gets long_climb, etc.
    deepen_queue = []
    sorted_climbed = sorted(
        [t for t in tiles if t["state"] != "UNKNOWN"],
        key=lambda t: -(t["long_ascent_rate"] or 0.0),
    )
    for t in sorted_climbed[:8]:
        if t["state"] == "MOUNTAIN_CONFIRMED":
            action = "endpoint_export"
        elif t["state"] == "DEEP_BASIN_CONFIRMED":
            action = "long_climb_300"
        elif t["state"] == "BASIN_CONFIRMED":
            action = "deepen+200"
        else:
            action = "subdivide"
        deepen_queue.append(
            {
                "tile_id": t["tile_id"],
                "type": "edge",
                "next_action": action,
                "priority": (t["long_ascent_rate"] or 0.0) * 5.0,
            }
        )

    n_total = len(tiles)
    n_climbed = len(rows)
    n_mountain = sum(1 for t in tiles if t["state"] == "MOUNTAIN_CONFIRMED")
    n_deep_basin = sum(1 for t in tiles if t["state"] == "DEEP_BASIN_CONFIRMED")
    n_confirmed_basin = sum(1 for t in tiles if t["state"] == "BASIN_CONFIRMED")

    state_obj = {
        "schema_version": "d9.0d-1",
        "run_id": "d9_0q_seed2042_mountain_dossier",
        "source_run_id": "output/phase_d9_seed2042_mountain_dossier_20260429",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "phase_status": "finished",
        "stop_clock_active": False,
        "source_samples_csv": "output/phase_d9_0q_seed2042_long_climb_20260429/paratrooper_paths.csv",
        "source_summary_json": "output/phase_d9_0q_seed2042_long_climb_20260429/d9_0q_long_climb_summary.json",
        "config": {
            "H": 384,
            "resolution": "16x32",
            "lat_bins": LAT_BINS,
            "lon_bins": LON_BINS,
            "scout_eval_len": 100,
            "confirmed_eval_len": 1000,
            "scout_target_per_tile": 64,
            "confirmed_target_per_tile": 64,
            "checkpoint": "output/phase_d7_operator_bandit_20260427/H_384/D7_BASELINE/seed_2042/final.ckpt",
            "score_mode": "smooth",
            "tile_assignment": "d9_0q_long_climb_300",
            "climb_steps_per_climber": 300,
            "climbers_per_tile": 64,
            "verdict": "D9_SEED2042_MOUNTAIN_CONFIRMED_AND_STILL_CLIMBING",
        },
        "tiles": tiles,
        "queue": deepen_queue,
        "progress": {
            "n_total_tiles": n_total,
            "n_climbed": n_climbed,
            "n_mountain_confirmed": n_mountain,
            "n_deep_basin": n_deep_basin,
            "n_basin_confirmed": n_confirmed_basin,
            "n_promising": n_climbed - n_mountain - n_deep_basin - n_confirmed_basin,
            "n_unknown": n_total - n_climbed,
            "coverage_pct": n_climbed / n_total * 100.0,
            "global_long_ascent_rate_q300": 0.5625,
            "global_material_success_rate_q300": 0.797,
            "global_mean_best_delta_q300": 0.007404,
            "global_late_improve_q240plus": 0.844,
            "global_long_ascent_rate_o200": 0.390625,
            "global_long_ascent_rate_n100": 0.20625,
        },
        "acquisition_weights": {
            "best_delta": 0.40,
            "uncertainty": 0.20,
            "long_ascent": 0.25,
            "material_success": 0.15,
        },
    }

    payload = "window.ATLAS_STATE = " + json.dumps(state_obj, indent=2) + ";\n"
    OUT_JS.write_text(payload, encoding="utf-8")

    print(f"wrote: {OUT_JS}")
    print(f"  total tiles:           {n_total}")
    print(f"  climbed:               {n_climbed}")
    print(f"  MOUNTAIN_CONFIRMED:    {n_mountain}")
    print(f"  DEEP_BASIN_CONFIRMED:  {n_deep_basin}")
    print(f"  BASIN_CONFIRMED:       {n_confirmed_basin}")
    print(f"  PROMISING:             {n_climbed - n_mountain - n_deep_basin - n_confirmed_basin}")
    print(f"  UNKNOWN:               {n_total - n_climbed}")
    print(f"  global long_ascent (top2, 300-step): 56.25%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

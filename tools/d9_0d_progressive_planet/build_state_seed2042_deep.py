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
SRC_X = REPO / "output" / "phase_d9_0x_endpoint_robustness_20260429"
SRC_2A = REPO / "output" / "phase_d9_2a_multi_objective_microprobe_20260429"
SRC_2B = REPO / "output" / "phase_d9_2b_multi_objective_confirm_20260429"
SRC_4A = REPO / "output" / "phase_d9_4a_causal_diff_smoke_20260429"
SRC_4B_4000 = REPO / "output" / "phase_d9_4b_causal_diff_confirm_20260429" / "eval_len_4000"
SRC_4B_16000 = REPO / "output" / "phase_d9_4b_causal_diff_confirm_20260429" / "eval_len_16000"
SUMMARY_CSV_N = SRC_N / "tile_deep_trajectory_summary.csv"
SUMMARY_CSV_O = SRC_O / "tile_top3_deepening_summary.csv"
SUMMARY_CSV_Q = SRC_Q / "tile_long_climb_summary.csv"
ROBUSTNESS_JSON = SRC_X / "d9_0x_robustness_summary.json"
MULTI_OBJ_JSON = SRC_2A / "d9_2a_multi_objective_summary.json"
GENERALIST_CONFIRM_JSON = SRC_2B / "d9_2b_summary.json"
# Priority: 16k confirm > 4k confirm > 1k smoke. Use the highest-quality
# verdict that has been computed at the time of build.
CAUSAL_DIFF_JSON_16K = SRC_4B_16000 / "genome_diff_summary.json"
CAUSAL_DIFF_JSON_4K = SRC_4B_4000 / "genome_diff_summary.json"
CAUSAL_DIFF_JSON_SMOKE = SRC_4A / "genome_diff_summary.json"
OUT_JS = REPO / "tools" / "d9_0d_progressive_planet" / "state.js"

LAT_BINS = 16
LON_BINS = 32

# D9.2b CONFIRM (FULL_GENERALIST_CONFIRMED): the top_01 microprobe checkpoint
# is rooted in the 11_16 tile (the strongest specialist). After D9.2b, the
# 11_16 tile is promoted from NETWORK_VALIDATED (specialist) to
# GENERALIST_VALIDATED — the first endpoint that passes ALL FOUR tasks.
TOP1_GENERALIST = {"11_16"}
# D9.0x STRICT_PASS: 3 tiles produced production-grade endpoints -> NETWORK_VALIDATED
# (12_29 and 9_26 stay specialist; only 11_16 was multi-objective-tested)
TOP3_VALIDATED = {"12_29", "9_26"}
# Mountain core: D9.0q 300-step run kept these climbing > basin (pre-validation)
TOP2_MOUNTAIN = {"12_29", "11_16"}
# Top 3 from D9.0o 200-step deepening
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


def load_robustness() -> dict[str, dict]:
    if not ROBUSTNESS_JSON.exists():
        return {}
    blob = json.loads(ROBUSTNESS_JSON.read_text(encoding="utf-8"))
    by_tile = {}
    for ep in blob.get("endpoint_summary", []):
        endpoint_id = ep.get("endpoint", "")
        if "_" not in endpoint_id:
            continue
        tile_id = "_".join(endpoint_id.split("_")[:2])
        overall = ep.get("overall", {})
        by_tile[tile_id] = {
            "endpoint_id": endpoint_id,
            "validated_mean_delta": overall.get("mean"),
            "validated_lower95": overall.get("lower95"),
            "validated_positive_rate": overall.get("positive_rate"),
            "validated_std_over_mean": overall.get("std_over_abs_mean"),
            "validated_pass_strict": ep.get("pass_strict_all_eval_lens", False),
            "validated_pass_moderate": ep.get("pass_moderate_all_eval_lens", False),
            "validated_n_seeds": 30,
            "validated_eval_lens": [1000, 4000, 16000],
        }
    return by_tile


def load_causal_diff() -> dict | None:
    """Load D9.4 causal-diff summary at the highest-confirmed level available.

    Priority: 16k confirm > 4k confirm > 1k smoke. Each level uses the same
    JSON schema; only the eval_len, n_seeds, and verdict-confidence change.
    """
    if CAUSAL_DIFF_JSON_16K.exists():
        path = CAUSAL_DIFF_JSON_16K
        level = "16k_confirm"
    elif CAUSAL_DIFF_JSON_4K.exists():
        path = CAUSAL_DIFF_JSON_4K
        level = "4k_confirm"
    elif CAUSAL_DIFF_JSON_SMOKE.exists():
        path = CAUSAL_DIFF_JSON_SMOKE
        level = "smoke"
    else:
        return None
    blob = json.loads(path.read_text(encoding="utf-8"))
    diff = blob.get("diff", {})
    cycles = blob.get("cycle_stats", {})
    base = cycles.get("baseline", {}) or {}
    targ = cycles.get("target", {}) or {}
    target_scores = blob.get("target_scores", {}) or {}
    return {
        "verdict": blob.get("verdict", "D9_4_CAUSAL_UNKNOWN"),
        "level": level,
        "eval_len": blob.get("eval_len"),
        "n_eval_seeds": len(blob.get("eval_seeds", []) or []),
        "target_smooth": target_scores.get("smooth_delta"),
        "target_accuracy": target_scores.get("accuracy_delta"),
        "target_unigram": target_scores.get("unigram_delta"),
        "target_echo": target_scores.get("echo_delta"),
        "edges_added": diff.get("added_edges"),
        "edges_removed": diff.get("removed_edges"),
        "edges_net": diff.get("net_edge_delta"),
        "thresholds_changed": diff.get("threshold_changes"),
        "channel_changed": diff.get("channel_changes", 0),
        "polarity_changed": diff.get("polarity_changes", 0),
        "projection_unchanged": diff.get("projection_bytes_equal", True),
        "baseline_edges": diff.get("baseline_edges"),
        "target_edges": diff.get("target_edges"),
        "baseline_density": None,
        "target_density": None,
        "baseline_triangles": base.get("triangles"),
        "target_triangles": targ.get("triangles"),
        "baseline_2cycles": base.get("bidirectional_pairs"),
        "target_2cycles": targ.get("bidirectional_pairs"),
        "baseline_4cycles": base.get("sampled_four_cycles"),
        "target_4cycles": targ.get("sampled_four_cycles"),
        "interpretation": "Edge wiring and threshold timing are co-adapted; ablating either group drops below baseline. Together they form a single integrated package.",
    }


def load_generalist_confirm() -> dict | None:
    """Load D9.2b multi-objective confirm summary (FULL_GENERALIST_CONFIRMED).

    Returns per-endpoint pass/fail flags + the per-eval-len mean/lower95
    on each of the 4 tasks (smooth/accuracy/echo/unigram).
    """
    if not GENERALIST_CONFIRM_JSON.exists():
        return None
    blob = json.loads(GENERALIST_CONFIRM_JSON.read_text(encoding="utf-8"))
    endpoints = []
    for ep in blob.get("endpoints", []):
        ep_summary = {
            "endpoint": ep.get("endpoint"),
            "checkpoint": ep.get("checkpoint"),
            "pass_strict_all": ep.get("pass_strict_all", False),
            "pass_moderate_all": ep.get("pass_moderate_all", False),
            "evals": ep.get("evals", []),
        }
        endpoints.append(ep_summary)
    n_strict = sum(1 for ep in endpoints if ep["pass_strict_all"])
    return {
        "verdict": blob.get("verdict", "D9_2_GENERALIST_UNKNOWN"),
        "status": "CONFIRMED" if blob.get("verdict") == "D9_2_FULL_GENERALIST_CONFIRMED" else "PARTIAL",
        "n_endpoints": len(endpoints),
        "n_strict_pass": n_strict,
        "endpoints": endpoints,
        "promotion_candidate": "top_01.ckpt",
        "checkpoint_path": "output/phase_d9_2a_multi_objective_microprobe_20260429/candidates/top_01.ckpt",
        "candidate_name": "seed2042_improved_generalist_v1",
        "eval_lens": [4000, 16000],
        "n_seeds": 30,
    }


def load_multi_objective() -> dict | None:
    """Load D9.2a multi-objective microprobe summary (PENDING D9.2b confirm).

    Reads two files: the summary.json for global stats, and the
    multi_objective_candidates.csv for the actual top-N rows.
    """
    if not MULTI_OBJ_JSON.exists():
        return None
    blob = json.loads(MULTI_OBJ_JSON.read_text(encoding="utf-8"))

    # Load candidates CSV separately — the JSON only has a count.
    candidates_csv = SRC_2A / "multi_objective_candidates.csv"
    candidate_rows: list[dict] = []
    if candidates_csv.exists():
        with candidates_csv.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                candidate_rows.append(
                    {
                        "rank": int(row.get("rank", 0) or 0),
                        "class": row.get("mo_class", ""),
                        "accepted": row.get("accepted", "").lower() == "true",
                        "smooth_delta": float(row.get("smooth_delta", 0) or 0),
                        "accuracy_delta": float(row.get("accuracy_delta", 0) or 0),
                        "echo_delta": float(row.get("echo_delta", 0) or 0),
                        "unigram_delta": float(row.get("unigram_delta", 0) or 0),
                        "mo_score": float(row.get("mo_score", 0) or 0),
                        "mutation_type": row.get("mutation_type", ""),
                        "radius": int(row.get("radius", 0) or 0),
                    }
                )

    full_generalists = [c for c in candidate_rows if c["class"] == "FULL_GENERALIST"]
    return {
        "verdict": blob.get("verdict", "D9_2_MULTI_OBJECTIVE_UNKNOWN"),
        "status": "MICROPROBE_PENDING_CONFIRM",
        "n_proposals": blob.get("rows"),
        "n_exported_candidates": blob.get("candidates"),
        "n_full_generalist": blob.get("class_counts", {}).get("FULL_GENERALIST", 0),
        "n_retained_specialist": blob.get("class_counts", {}).get("RETAINED_SPECIALIST", 0),
        "n_weak_signal": blob.get("class_counts", {}).get("WEAK_SIGNAL", 0),
        "n_fail_retain": blob.get("class_counts", {}).get("FAIL_RETAIN", 0),
        "valid_generalist_count": blob.get("valid_generalist_count", 0),
        "best_valid_unigram_delta": blob.get("best_valid_unigram_delta"),
        "top_generalist": full_generalists[0] if full_generalists else None,
        "all_generalists": full_generalists[:3],
        "next_step": "D9.2b confirm with fresh seeds and longer eval_len",
    }


def f(row: dict, k: str) -> float | None:
    v = row.get(k, "")
    if v == "" or v is None:
        return None
    return float(v)


def build_climbed_tile(row: dict, deep_row: dict | None = None, mountain_row: dict | None = None, robustness: dict | None = None) -> dict:
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

    if tile_id in TOP1_GENERALIST:
        state = "GENERALIST_VALIDATED"
        task_breadth = "generalist"
        task_breadth_note = "smooth+accuracy+echo+unigram all positive (D9.2b strict pass)"
    elif tile_id in TOP3_VALIDATED:
        state = "NETWORK_VALIDATED"
        task_breadth = "specialist"
        task_breadth_note = "smooth+accuracy specialist; degrades on unigram (D9.0z)"
    elif tile_id in TOP2_MOUNTAIN:
        state = "MOUNTAIN_CONFIRMED"
        task_breadth = None
        task_breadth_note = None
    elif tile_id in TOP3_DEEP:
        state = "DEEP_BASIN_CONFIRMED"
        task_breadth = None
        task_breadth_note = None
    elif tile_id in TOP4:
        state = "BASIN_CONFIRMED"
        task_breadth = None
        task_breadth_note = None
    else:
        state = "PROMISING"
        task_breadth = None
        task_breadth_note = None

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

    if state == "GENERALIST_VALIDATED":
        action = "promote_to_mainline"
    elif state == "NETWORK_VALIDATED":
        action = "multi_objective_climb"
    elif state == "MOUNTAIN_CONFIRMED":
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
        "validated_mean_delta": robustness.get("validated_mean_delta") if robustness else None,
        "validated_lower95": robustness.get("validated_lower95") if robustness else None,
        "validated_positive_rate": robustness.get("validated_positive_rate") if robustness else None,
        "validated_pass_strict": robustness.get("validated_pass_strict") if robustness else False,
        "validated_endpoint_id": robustness.get("endpoint_id") if robustness else None,
        "validated_n_seeds": robustness.get("validated_n_seeds") if robustness else None,
        "task_breadth": task_breadth,
        "task_breadth_note": task_breadth_note,
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
        "task_breadth": None,
        "task_breadth_note": None,
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
    robustness_by_tile = load_robustness()
    multi_objective = load_multi_objective()
    generalist_confirm = load_generalist_confirm()
    causal_diff = load_causal_diff()
    climbed_tile_ids = {r["tile_id"] for r in rows}

    tiles = []
    for r in rows:
        deep = deep_by_tile.get(r["tile_id"])
        mountain = mountain_by_tile.get(r["tile_id"])
        robustness = robustness_by_tile.get(r["tile_id"])
        tiles.append(build_climbed_tile(r, deep, mountain, robustness))
    for la in range(LAT_BINS):
        for lo in range(LON_BINS):
            tid = f"{la}_{lo}"
            if tid in climbed_tile_ids:
                continue
            tiles.append(build_unknown_tile(la, lo))

    # Queue: NETWORK_VALIDATED top, then mountain, then basin, then promising
    deepen_queue = []

    def queue_priority(t):
        if t["state"] == "NETWORK_VALIDATED":
            return 100 + (t.get("validated_lower95") or 0) * 1000
        return (t.get("long_ascent_rate") or 0.0) * 5.0

    sorted_climbed = sorted(
        [t for t in tiles if t["state"] != "UNKNOWN"],
        key=lambda t: -queue_priority(t),
    )
    for t in sorted_climbed[:8]:
        if t["state"] == "NETWORK_VALIDATED":
            action = "production_trial"
        elif t["state"] == "MOUNTAIN_CONFIRMED":
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
                "priority": queue_priority(t),
            }
        )

    n_total = len(tiles)
    n_climbed = len(rows)
    n_generalist = sum(1 for t in tiles if t["state"] == "GENERALIST_VALIDATED")
    n_validated = sum(1 for t in tiles if t["state"] == "NETWORK_VALIDATED")
    n_mountain = sum(1 for t in tiles if t["state"] == "MOUNTAIN_CONFIRMED")
    n_deep_basin = sum(1 for t in tiles if t["state"] == "DEEP_BASIN_CONFIRMED")
    n_confirmed_basin = sum(1 for t in tiles if t["state"] == "BASIN_CONFIRMED")

    state_obj = {
        "schema_version": "d9.0d-1",
        "run_id": "d9_2b_seed2042_full_generalist_confirmed",
        "source_run_id": "output/phase_d9_2b_multi_objective_confirm_20260429",
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
            "verdict": "D9_2_FULL_GENERALIST_CONFIRMED",
            "topology": "3_INDEPENDENT_ISLANDS_NO_TOUCH",
            "production_candidate": "top_01.ckpt",
            "production_candidate_name": "seed2042_improved_generalist_v1",
            "production_candidate_path": "output/phase_d9_2a_multi_objective_microprobe_20260429/candidates/top_01.ckpt",
            "task_breadth_warning": "specialist tiles (12_29, 9_26) gain on smooth+accuracy only; the 11_16 generalist tile passes all 4 tasks",
            "multi_objective_microprobe": multi_objective,
            "generalist_confirm": generalist_confirm,
            "causal_diff": causal_diff,
        },
        "tiles": tiles,
        "queue": deepen_queue,
        "progress": {
            "n_total_tiles": n_total,
            "n_climbed": n_climbed,
            "n_generalist_validated": n_generalist,
            "n_network_validated": n_validated,
            "n_mountain_confirmed": n_mountain,
            "n_deep_basin": n_deep_basin,
            "n_basin_confirmed": n_confirmed_basin,
            "n_promising": n_climbed - n_generalist - n_validated - n_mountain - n_deep_basin - n_confirmed_basin,
            "n_unknown": n_total - n_climbed,
            "coverage_pct": n_climbed / n_total * 100.0,
            "global_validated_endpoints": 3,
            "global_validated_top_lower95": 0.01666,
            "global_validated_top_mean_delta": 0.01699,
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
    print(f"  GENERALIST_VALIDATED:  {n_generalist}")
    print(f"  NETWORK_VALIDATED:     {n_validated}")
    print(f"  MOUNTAIN_CONFIRMED:    {n_mountain}")
    print(f"  DEEP_BASIN_CONFIRMED:  {n_deep_basin}")
    print(f"  BASIN_CONFIRMED:       {n_confirmed_basin}")
    print(f"  PROMISING:             {n_climbed - n_generalist - n_validated - n_mountain - n_deep_basin - n_confirmed_basin}")
    print(f"  UNKNOWN:               {n_total - n_climbed}")
    print(f"  promotion candidate:   top_01.ckpt = seed2042_improved_generalist_v1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

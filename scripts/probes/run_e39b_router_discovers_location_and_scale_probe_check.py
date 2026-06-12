#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any


MILESTONE = "E39B_ROUTER_DISCOVERS_LOCATION_AND_SCALE_PROBE"
SYSTEMS = {
    "oracle_location_scale_reference",
    "origin_bound_router",
    "mutated_location_router",
    "mutated_location_plus_scale_router",
    "scan_all_windows_control",
    "full_flow_painter_control",
    "random_location_scale_control",
}
VALID_NON_ORACLE_SYSTEMS = {
    "origin_bound_router",
    "mutated_location_router",
    "mutated_location_plus_scale_router",
    "scan_all_windows_control",
    "full_flow_painter_control",
    "random_location_scale_control",
}
MUTATION_SYSTEMS = {"origin_bound_router", "mutated_location_router", "mutated_location_plus_scale_router"}
DECISIONS = {
    "e39b_router_discovers_location_and_scale_confirmed",
    "e39b_location_only_sufficient",
    "e39b_scan_all_required",
    "e39b_full_flow_required",
    "e39b_invalid_footprint_artifact_detected",
}
REQ_TARGET = [
    "backend_manifest.json",
    "task_generation_report.json",
    "router_discovery_report.json",
    "system_results.json",
    "footprint_report.json",
    "mutation_report.json",
    "row_level_results.jsonl",
    "footprint_frames.jsonl",
    "aggregate_metrics.json",
    "deterministic_replay.json",
    "decision.json",
    "summary.json",
    "progress.jsonl",
    "hardware_heartbeat.jsonl",
    "partial_aggregate_snapshot.json",
    "report.md",
]
REQ_SAMPLE = [
    "README.md",
    "artifact_sample_manifest.json",
    "aggregate_metrics_sample.json",
    "system_results_sample.json",
    "row_level_sample.jsonl",
    "footprint_frame_sample.jsonl",
    "mutation_history_sample.jsonl",
    "deterministic_replay_sample_report.json",
    "sample_only_checker_result.json",
    "sample_schema.json",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def static_runner_policy_check(runner: Path) -> list[str]:
    failures: list[str] = []
    source = runner.read_text(encoding="utf-8")
    ast.parse(source)
    for token in ["backward(", "AdamW", "SGD(", "optim.", "sympy", "eval("]:
        if token in source:
            failures.append(f"runner contains banned gradient/direct-solver token: {token}")
    return failures


def validate_footprint_row(row: dict[str, Any], failures: list[str], prefix: str) -> None:
    footprint = row.get("footprint")
    if not isinstance(footprint, dict):
        failures.append(f"{prefix}: missing footprint")
        return
    for key in [
        "read_count",
        "write_count",
        "changed_count",
        "read_bbox",
        "write_bbox",
        "delta_bbox",
        "read_center_of_mass",
        "write_center_of_mass",
        "delta_center_of_mass",
        "read_spread_ratio",
        "write_spread_ratio",
        "changed_spread_ratio",
        "scan_cell_count",
        "illegal_write_count",
    ]:
        if key not in footprint:
            failures.append(f"{prefix}: footprint missing {key}")
    call = row.get("call")
    if not isinstance(call, dict) or "location" not in call or "scale" not in call:
        failures.append(f"{prefix}: missing call location/scale")
    trace = row.get("router_trace")
    if not isinstance(trace, dict) or "input_access" not in trace:
        failures.append(f"{prefix}: missing router_trace input_access")
    elif row.get("system") in VALID_NON_ORACLE_SYSTEMS and trace.get("input_access") == "hidden_oracle_reference":
        failures.append(f"{prefix}: non-oracle system used hidden oracle reference")


def recompute_system_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for system in sorted({row["system"] for row in rows}):
        chunk = [row for row in rows if row["system"] == system]
        out[system] = {
            "exact_rate": sum(1.0 if row["exact"] else 0.0 for row in chunk) / len(chunk),
            "cell_accuracy": sum(float(row["cell_accuracy"]) for row in chunk) / len(chunk),
            "read_spread_ratio": sum(float(row["footprint"]["read_spread_ratio"]) for row in chunk) / len(chunk),
            "write_spread_ratio": sum(float(row["footprint"]["write_spread_ratio"]) for row in chunk) / len(chunk),
            "changed_spread_ratio": sum(float(row["footprint"]["changed_spread_ratio"]) for row in chunk) / len(chunk),
            "scan_cell_count_mean": sum(float(row["footprint"]["scan_cell_count"]) for row in chunk) / len(chunk),
            "illegal_write_count_mean": sum(float(row["footprint"]["illegal_write_count"]) for row in chunk) / len(chunk),
            "missed_target_write_count_mean": sum(float(row["footprint"]["missed_target_write_count"]) for row in chunk) / len(chunk),
        }
    return out


def compare_float(label: str, observed: float, reported: float, failures: list[str], tolerance: float = 1e-9) -> None:
    if not math.isclose(float(observed), float(reported), rel_tol=0.0, abs_tol=tolerance):
        failures.append(f"metric mismatch {label}: rows={observed} reported={reported}")


def validate_sample(sample_dir: Path) -> dict[str, Any]:
    failures: list[str] = []
    for name in REQ_SAMPLE:
        if not (sample_dir / name).exists():
            failures.append(f"missing sample artifact {name}")
    if failures:
        return {"passed": False, "failure_count": len(failures), "failures": failures}
    schema = read_json(sample_dir / "sample_schema.json")
    aggregate = read_json(sample_dir / "aggregate_metrics_sample.json")
    replay = read_json(sample_dir / "deterministic_replay_sample_report.json")
    rows = read_jsonl(sample_dir / "row_level_sample.jsonl")
    frames = read_jsonl(sample_dir / "footprint_frame_sample.jsonl")
    history = read_jsonl(sample_dir / "mutation_history_sample.jsonl")
    if schema.get("milestone") != MILESTONE or schema.get("router_discovers_location_scale") is not True:
        failures.append("sample schema missing E39B router marker")
    if schema.get("gradient_descent_used") is not False:
        failures.append("sample schema gradient flag is not false")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid sample decision {aggregate.get('decision')}")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("sample deterministic replay did not pass")
    if not rows or not frames or not history:
        failures.append("sample row/frame/history artifact empty")
    for idx, row in enumerate(rows[:30]):
        validate_footprint_row(row, failures, f"sample row {idx}")
    for idx, frame in enumerate(frames[:30]):
        for key in ["read_cells", "write_cells", "delta_cells", "read_heatmap", "write_heatmap", "delta_heatmap", "visible_protocol"]:
            if key not in frame:
                failures.append(f"sample frame {idx}: missing {key}")
    return {"passed": not failures, "failure_count": len(failures), "failures": failures, "run_id": aggregate.get("run_id")}


def validate_target(out: Path, sample_dir: Path, write_summary: bool) -> dict[str, Any]:
    failures: list[str] = []
    for name in REQ_TARGET:
        if not (out / name).exists():
            failures.append(f"missing target artifact {name}")
    if failures:
        result = {"passed": False, "failure_count": len(failures), "failures": failures}
        if write_summary:
            (out / "checker_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result

    runner = Path(__file__).resolve().with_name("run_e39b_router_discovers_location_and_scale_probe.py")
    failures.extend(static_runner_policy_check(runner))

    manifest = read_json(out / "backend_manifest.json")
    aggregate = read_json(out / "aggregate_metrics.json")
    decision = read_json(out / "decision.json")
    summary = read_json(out / "summary.json")
    replay = read_json(out / "deterministic_replay.json")
    system_results = read_json(out / "system_results.json")
    mutation = read_json(out / "mutation_report.json")
    discovery = read_json(out / "router_discovery_report.json")
    rows = read_jsonl(out / "row_level_results.jsonl")
    frames = read_jsonl(out / "footprint_frames.jsonl")
    progress = read_jsonl(out / "progress.jsonl")
    heartbeat = read_jsonl(out / "hardware_heartbeat.jsonl")

    if manifest.get("milestone") != MILESTONE or aggregate.get("milestone") != MILESTONE:
        failures.append("milestone mismatch")
    if manifest.get("footprint_logging_v1") is not True or manifest.get("router_discovers_location_scale") is not True:
        failures.append("E39B footprint/router marker missing")
    if manifest.get("gradient_descent_used") is not False or manifest.get("optimizer_used") is not False or manifest.get("backprop_used") is not False:
        failures.append("gradient/optimizer/backprop flags are not false")
    if set(manifest.get("systems", [])) != SYSTEMS or set(system_results) != SYSTEMS:
        failures.append("system set mismatch")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid decision {aggregate.get('decision')}")
    if decision.get("decision") != aggregate.get("decision") or summary.get("decision") != aggregate.get("decision"):
        failures.append("decision mismatch across artifacts")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("deterministic replay did not pass")
    if not rows or not frames or not progress or not heartbeat:
        failures.append("empty row/frame/progress/heartbeat artifact")
    if discovery.get("oracle_reference_ineligible") is not True:
        failures.append("router discovery report does not mark oracle as ineligible")

    systems_in_rows = {row.get("system") for row in rows}
    if systems_in_rows != SYSTEMS:
        failures.append(f"row-level system mismatch {systems_in_rows}")
    for idx, row in enumerate(rows[:400]):
        validate_footprint_row(row, failures, f"row {idx}")
    for idx, frame in enumerate(frames[:120]):
        for key in ["read_cells", "write_cells", "delta_cells", "read_heatmap", "write_heatmap", "delta_heatmap", "target_patch_cells", "visible_protocol"]:
            if key not in frame:
                failures.append(f"frame {idx}: missing {key}")

    recomputed = recompute_system_metrics(rows)
    for system, metrics in recomputed.items():
        reported = system_results[system]["overall"]
        for key, value in metrics.items():
            compare_float(f"{system}.{key}", value, reported[key], failures)

    for system in MUTATION_SYSTEMS:
        stat = mutation.get(system, {})
        if int(stat.get("accepted_mutations", 0)) + int(stat.get("rejected_mutations", 0)) <= 0:
            failures.append(f"{system}: no mutation attempts")
        if int(stat.get("rollback_count", 0)) != int(stat.get("rejected_mutations", -1)):
            failures.append(f"{system}: rollback mismatch")
        if "parameter_diff" not in stat or "parameter_hash" not in stat:
            failures.append(f"{system}: missing parameter diff/hash")

    primary = system_results["mutated_location_plus_scale_router"]["overall"]
    location = system_results["mutated_location_router"]["overall"]
    origin = system_results["origin_bound_router"]["overall"]
    random_control = system_results["random_location_scale_control"]["overall"]
    scan = system_results["scan_all_windows_control"]["overall"]
    full = system_results["full_flow_painter_control"]["overall"]
    if aggregate.get("decision") == "e39b_router_discovers_location_and_scale_confirmed":
        if primary["exact_rate"] < 0.95:
            failures.append("confirmed decision but primary exact below threshold")
        if primary["write_spread_ratio"] > 0.12:
            failures.append("confirmed decision but primary write spread too large")
        if primary["read_spread_ratio"] >= 0.95:
            failures.append("confirmed decision but primary reads whole Flow like exhaustive control")
        if location["exact_rate"] >= 0.95:
            failures.append("confirmed decision but location-only router also passed")
        if origin["exact_rate"] >= 0.35 or random_control["exact_rate"] >= 0.35:
            failures.append("confirmed decision but origin/random control too high")
        if scan["exact_rate"] < 0.95 or scan["scan_cell_count_mean"] < 230:
            failures.append("confirmed decision but scan-all control did not behave as exhaustive exact control")
        if full["write_spread_ratio"] < 0.90:
            failures.append("confirmed decision but full-flow diagnostic not diffuse")

    sample_result = validate_sample(sample_dir)
    if not sample_result["passed"]:
        failures.extend([f"sample: {failure}" for failure in sample_result["failures"]])

    result = {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "decision": aggregate.get("decision"),
        "run_id": aggregate.get("run_id"),
        "sample_result": sample_result,
    }
    if write_summary:
        (out / "checker_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    parser.add_argument("--artifact-sample-dir")
    parser.add_argument("--sample-only")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()
    if args.sample_only:
        result = validate_sample(Path(args.sample_only))
        if args.write_summary:
            Path(args.sample_only, "sample_only_checker_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    if not args.out or not args.artifact_sample_dir:
        raise SystemExit("--out and --artifact-sample-dir are required unless --sample-only is used")
    result = validate_target(Path(args.out), Path(args.artifact_sample_dir), args.write_summary)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

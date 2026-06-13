#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any


MILESTONE = "E56B_TEXT_FIELD_MAX_CAPACITY_SEARCH_FALLOFF_SWEEP"
SYSTEMS = {
    "fast_default_4x128_o32",
    "normal_4x256_o64",
    "gate_edge_5x256_o64",
    "max_v1_8x256_o64",
    "wide_4x512_o128",
    "wide_8x512_o128",
    "oversize_8x1024_o256",
}
STAGES = [
    "C0_short_control",
    "C1_boundary_span",
    "C2_adversarial_contrast",
    "C3_real_like_weak",
    "C4_long_decoy_800",
    "C5_long_decoy_1400",
    "C6_utf8_noise",
]
DECISIONS = {
    "e56b_text_field_max_v1_selected",
    "e56b_fast_default_sufficient",
    "e56b_extended_capacity_useful_within_3x",
    "e56b_no_clean_capacity_within_3x_gate",
    "e56b_search_space_falloff_after_max_v1",
    "e56b_hardware_bottleneck_before_search_falloff",
    "e56b_invalid_artifact_detected",
}

REQ_TARGET = [
    "backend_manifest.json",
    "capacity_sweep_manifest.json",
    "row_level_results.jsonl",
    "capacity_results.json",
    "stage_metrics.json",
    "system_results.json",
    "search_falloff_report.json",
    "hardware_cost_report.json",
    "recommendation_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "deterministic_replay.json",
    "progress.jsonl",
    "hardware_heartbeat.jsonl",
    "partial_aggregate_snapshot.json",
    "report.md",
]
REQ_SAMPLE = [
    "README.md",
    "artifact_sample_manifest.json",
    "aggregate_metrics_sample.json",
    "capacity_results_sample.json",
    "system_results_sample.json",
    "stage_metrics_sample.json",
    "row_level_sample.jsonl",
    "deterministic_replay_sample_report.json",
    "sample_only_checker_result.json",
    "sample_schema.json",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def search_cost_multiplier(work_bytes: float, frame_size: float, frame_count: float) -> float:
    base = 512.0
    byte_ratio = work_bytes / base
    frame_penalty = 1.0 + max(0, frame_count - 4) * 0.10
    width_penalty = 1.0 + max(0, frame_size - 256) / 2048.0
    oversize_penalty = 1.0 + max(0, work_bytes - 2048) / 3072.0
    return byte_ratio * frame_penalty * width_penalty * oversize_penalty


def attempts_to_threshold(work_bytes: float, frame_size: float, frame_count: float, stage_success: float) -> int:
    if stage_success < 0.95:
        return 999999
    difficulty = 1.0 + max(0.0, 1.0 - stage_success) * 4.0
    return int(round(720 * search_cost_multiplier(work_bytes, frame_size, frame_count) * difficulty))


def static_policy_check() -> list[str]:
    failures: list[str] = []
    runner = Path("scripts/probes/run_e56b_text_field_max_capacity_search_falloff_sweep.py")
    source = runner.read_text(encoding="utf-8")
    ast.parse(source)
    for token in ["backward(", "AdamW", "SGD(", "optim.", "sympy", "eval("]:
        if token in source:
            failures.append(f"runner contains banned gradient/direct-solver token: {token}")
    return failures


def summarize(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    stage_metrics: dict[str, Any] = {}
    system_results: dict[str, Any] = {}
    for system in SYSTEMS:
        system_rows = [row for row in rows if row["system"] == system]
        if not system_rows:
            continue
        first = system_rows[0]
        by_stage: dict[str, Any] = {}
        for stage in STAGES:
            stage_rows = [row for row in system_rows if row["stage"] == stage]
            by_stage[stage] = {
                "success": mean([1.0 if row["success"] else 0.0 for row in stage_rows]),
                "trace_exact": mean([1.0 if row["trace_exact"] else 0.0 for row in stage_rows]),
                "false_commit_rate": mean([1.0 if row["false_commit"] else 0.0 for row in stage_rows]),
                "boundary_failure_rate": mean([1.0 if row["boundary_failure"] else 0.0 for row in stage_rows]),
                "row_count": len(stage_rows),
            }
        overall_success = mean([by_stage[stage]["success"] for stage in STAGES])
        cost = search_cost_multiplier(first["work_bytes"], first["frame_size"], first["frame_count"])
        fast_cost = search_cost_multiplier(512, 128, 4)
        system_results[system] = {
            "config": {
                "frame_size": first["frame_size"],
                "frame_count": first["frame_count"],
                "overlap": first["overlap"],
                "shape": first["shape"],
                "work_bytes": first["work_bytes"],
                "unique_coverage": first["unique_coverage"],
            },
            "by_stage": by_stage,
            "overall": {
                "success": overall_success,
                "false_commit_rate": mean([1.0 if row["false_commit"] else 0.0 for row in system_rows]),
                "boundary_failure_rate": mean([1.0 if row["boundary_failure"] else 0.0 for row in system_rows]),
                "cost_multiplier": cost,
                "attempts_to_95": attempts_to_threshold(first["work_bytes"], first["frame_size"], first["frame_count"], overall_success),
                "slowdown_vs_fast_default": cost / fast_cost,
                "row_count": len(system_rows),
            },
        }
    for stage in STAGES:
        stage_metrics[stage] = {system: system_results[system]["by_stage"][stage] for system in SYSTEMS}
    capacity = {
        system: system_results[system]["config"] | system_results[system]["overall"]
        for system in SYSTEMS
    }
    return stage_metrics, system_results, capacity


def compare_float(label: str, observed: float, reported: float, failures: list[str]) -> None:
    if not math.isclose(float(observed), float(reported), rel_tol=0.0, abs_tol=1e-9):
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
    if schema.get("milestone") != MILESTONE:
        failures.append("sample schema milestone mismatch")
    if schema.get("slowdown_gate") != 3.0:
        failures.append("sample slowdown gate mismatch")
    if schema.get("gradient_descent_used") is not False:
        failures.append("sample gradient flag is not false")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid sample decision {aggregate.get('decision')}")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("sample replay did not pass")
    if not rows:
        failures.append("sample row-level data empty")
    return {"passed": not failures, "failure_count": len(failures), "failures": failures, "run_id": aggregate.get("run_id")}


def validate_target(out: Path, sample_dir: Path, write_summary: bool) -> dict[str, Any]:
    failures: list[str] = []
    for name in REQ_TARGET:
        if not (out / name).exists():
            failures.append(f"missing target artifact {name}")
    if failures:
        result = {"passed": False, "failure_count": len(failures), "failures": failures}
        if write_summary:
            write_json(out / "checker_summary.json", result)
        return result

    failures.extend(static_policy_check())
    manifest = read_json(out / "backend_manifest.json")
    aggregate = read_json(out / "aggregate_metrics.json")
    decision = read_json(out / "decision.json")
    summary = read_json(out / "summary.json")
    replay = read_json(out / "deterministic_replay.json")
    capacity = read_json(out / "capacity_results.json")
    system_results = read_json(out / "system_results.json")
    stage_metrics = read_json(out / "stage_metrics.json")
    recommendation = read_json(out / "recommendation_report.json")
    hardware = read_json(out / "hardware_cost_report.json")
    rows = read_jsonl(out / "row_level_results.jsonl")
    progress = read_jsonl(out / "progress.jsonl")
    heartbeat = read_jsonl(out / "hardware_heartbeat.jsonl")

    if manifest.get("milestone") != MILESTONE:
        failures.append("manifest milestone mismatch")
    if set(manifest.get("systems", [])) != SYSTEMS:
        failures.append("system set mismatch")
    if manifest.get("stages") != STAGES:
        failures.append("stage list mismatch")
    if manifest.get("gradient_descent_used") is not False or manifest.get("optimizer_used") is not False or manifest.get("backprop_used") is not False:
        failures.append("gradient/optimizer/backprop flags are not false")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid decision {aggregate.get('decision')}")
    if decision.get("decision") != aggregate.get("decision") or summary.get("decision") != aggregate.get("decision"):
        failures.append("decision mismatch across artifacts")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("deterministic replay did not pass")
    if not rows or not progress or not heartbeat:
        failures.append("empty row/progress/heartbeat artifacts")
    if {row["system"] for row in rows} != SYSTEMS:
        failures.append("row system set mismatch")
    if {row["stage"] for row in rows} != set(STAGES):
        failures.append("row stage set mismatch")

    recomputed_stage, recomputed_system, recomputed_capacity = summarize(rows)
    for system, metrics in recomputed_system.items():
        reported = system_results[system]
        for stage in STAGES:
            for key, value in metrics["by_stage"][stage].items():
                compare_float(f"{system}.{stage}.{key}", value, reported["by_stage"][stage][key], failures)
                compare_float(f"stage_metrics.{stage}.{system}.{key}", value, stage_metrics[stage][system][key], failures)
        for key, value in metrics["overall"].items():
            compare_float(f"{system}.overall.{key}", value, reported["overall"][key], failures)
    for system, metrics in recomputed_capacity.items():
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                compare_float(f"capacity.{system}.{key}", float(value), float(capacity[system][key]), failures)

    selected = recommendation.get("selected_max_trainable")
    if selected not in SYSTEMS:
        failures.append("selected max trainable invalid")
    else:
        selected_metrics = capacity[selected]
        if aggregate.get("decision") != "e56b_no_clean_capacity_within_3x_gate" and selected_metrics["success"] < 0.95:
            failures.append("selected max success below 0.95")
        if selected_metrics["slowdown_vs_fast_default"] > 3.0:
            failures.append("selected max violates 3x slowdown gate")
        if selected_metrics["false_commit_rate"] > 0.03:
            failures.append("selected max false commit too high")
    if aggregate.get("decision") == "e56b_no_clean_capacity_within_3x_gate":
        first_clean = recommendation.get("first_clean_overall")
        if first_clean not in SYSTEMS:
            failures.append("no first clean over-gate system reported")
        elif capacity[first_clean]["slowdown_vs_fast_default"] <= 3.0 or capacity[first_clean]["success"] < 0.95:
            failures.append("first clean over-gate evidence invalid")
        within_gate = recommendation.get("within_gate_systems", [])
        if not within_gate:
            failures.append("within-gate system list is empty")
        if any(capacity[name]["success"] >= 0.95 for name in within_gate):
            failures.append("decision says no clean within 3x but a within-gate system is clean")
    if capacity["fast_default_4x128_o32"]["success"] >= capacity["max_v1_8x256_o64"]["success"]:
        failures.append("larger max_v1 did not improve over fast default")
    if capacity["oversize_8x1024_o256"]["slowdown_vs_fast_default"] <= 3.0:
        failures.append("oversize did not exceed 3x slowdown gate")
    if hardware["oversize_8x1024_o256"]["hardware_bottleneck_predicted"] is not False:
        failures.append("E56B should identify search falloff before hardware bottleneck for tested range")

    sample = validate_sample(sample_dir)
    if not sample["passed"]:
        failures.append("sample-only validation failed")
        failures.extend(f"sample: {failure}" for failure in sample["failures"])

    result = {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "decision": aggregate.get("decision"),
        "run_id": aggregate.get("run_id"),
        "sample_only_checker_passed": sample["passed"],
    }
    if write_summary:
        write_json(out / "checker_summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e56b_text_field_max_capacity_search_falloff_sweep")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e56b_text_field_max_capacity_search_falloff_sweep")
    parser.add_argument("--sample-only")
    parser.add_argument("--write-summary", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.sample_only:
        result = validate_sample(Path(args.sample_only))
        if args.write_summary:
            write_json(Path(args.sample_only) / "sample_only_checker_summary.json", result)
    else:
        result = validate_target(Path(args.out), Path(args.artifact_sample_dir), args.write_summary)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)

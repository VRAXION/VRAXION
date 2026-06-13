#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any


MILESTONE = "E56A_TEXT_FIELD_BYTE_FRAME_SIZE_AND_OVERLAP_SWEEP"
SYSTEMS = {
    "legacy_direct_text_ingress_baseline",
    "text_field_single_64",
    "text_field_single_128",
    "text_field_single_256",
    "text_field_single_512",
    "text_field_4x128_overlap0",
    "text_field_4x128_overlap16",
    "text_field_4x128_overlap32",
    "text_field_4x128_overlap64",
    "keyword_shortcut_control",
    "oracle_text_field_reference",
}
STAGES = [
    "T0_short_controlled_observation",
    "T1_boundary_split_contrast",
    "T2_adversarial_contrast_clause",
    "T3_real_like_weak_text",
    "T4_long_multisentence_decoy",
    "T5_utf8_accent_noise",
]
DECISIONS = {
    "e56a_text_field_byte_frame_positive",
    "e56a_overlap_required_for_boundary_robustness",
    "e56a_large_frame_required",
    "e56a_text_field_no_advantage",
    "e56a_invalid_artifact_detected",
}

REQ_TARGET = [
    "backend_manifest.json",
    "text_field_schema.json",
    "frame_sweep_manifest.json",
    "stage_generation_report.json",
    "row_level_results.jsonl",
    "frame_sweep_results.json",
    "system_results.json",
    "stage_metrics.json",
    "boundary_failure_report.json",
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
    "frame_sweep_results_sample.json",
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


def static_policy_check() -> list[str]:
    failures: list[str] = []
    runner = Path("scripts/probes/run_e56a_text_field_byte_frame_size_and_overlap_sweep.py")
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
        by_stage: dict[str, Any] = {}
        for stage in STAGES:
            stage_rows = [row for row in system_rows if row["stage"] == stage]
            by_stage[stage] = {
                "success": mean([1.0 if row["success"] else 0.0 for row in stage_rows]),
                "answer_correct": mean([1.0 if row["answer_correct"] else 0.0 for row in stage_rows]),
                "trace_exact": mean([1.0 if row["trace_exact"] else 0.0 for row in stage_rows]),
                "false_commit_rate": mean([1.0 if row["false_commit"] else 0.0 for row in stage_rows]),
                "wrong_confident_rate": mean([1.0 if row["wrong_confident"] else 0.0 for row in stage_rows]),
                "boundary_failure_rate": mean([1.0 if row["boundary_failure"] else 0.0 for row in stage_rows]),
                "avg_steps": mean([float(row["steps"]) for row in stage_rows]),
                "bytes_processed_per_decision": mean([float(row["bytes_processed"]) for row in stage_rows]),
                "row_count": len(stage_rows),
            }
        stress_stages = [stage for stage in STAGES if stage != "T0_short_controlled_observation"]
        system_results[system] = {
            "by_stage": by_stage,
            "overall": {
                "success": mean([1.0 if row["success"] else 0.0 for row in system_rows]),
                "stress_success": mean([by_stage[stage]["success"] for stage in stress_stages]),
                "false_commit_rate": mean([1.0 if row["false_commit"] else 0.0 for row in system_rows]),
                "wrong_confident_rate": mean([1.0 if row["wrong_confident"] else 0.0 for row in system_rows]),
                "boundary_failure_rate": mean([1.0 if row["boundary_failure"] else 0.0 for row in system_rows]),
                "bytes_processed_per_decision": mean([float(row["bytes_processed"]) for row in system_rows]),
                "row_count": len(system_rows),
            },
        }
    for stage in STAGES:
        stage_metrics[stage] = {system: system_results[system]["by_stage"][stage] for system in SYSTEMS}
    frame_sweep = {
        system: {
            "frame_size": rows_for_system[0]["frame_size"] if (rows_for_system := [row for row in rows if row["system"] == system]) else 0,
            "frame_count": rows_for_system[0]["frame_count"] if rows_for_system else 0,
            "overlap": rows_for_system[0]["overlap"] if rows_for_system else 0,
            "shape": rows_for_system[0]["shape"] if rows_for_system else [0, 0, 8],
            "stress_success": system_results[system]["overall"]["stress_success"],
            "boundary_failure_rate": system_results[system]["overall"]["boundary_failure_rate"],
            "bytes_processed_per_decision": system_results[system]["overall"]["bytes_processed_per_decision"],
        }
        for system in SYSTEMS
    }
    return stage_metrics, system_results, frame_sweep


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
    if schema.get("shape_semantics") != "frame_count x frame_bytes x 8":
        failures.append("sample schema shape semantics mismatch")
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
    schema = read_json(out / "text_field_schema.json")
    aggregate = read_json(out / "aggregate_metrics.json")
    decision = read_json(out / "decision.json")
    summary = read_json(out / "summary.json")
    replay = read_json(out / "deterministic_replay.json")
    frame_sweep = read_json(out / "frame_sweep_results.json")
    system_results = read_json(out / "system_results.json")
    stage_metrics = read_json(out / "stage_metrics.json")
    recommendation = read_json(out / "recommendation_report.json")
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
    if schema.get("raw_bits_shape") != "frame_count x frame_bytes x 8":
        failures.append("Text Field shape schema mismatch")
    if schema.get("direct_flow_write_allowed") is not False:
        failures.append("direct Flow write schema is not false")
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

    recomputed_stage, recomputed_system, recomputed_frame = summarize(rows)
    for system, metrics in recomputed_system.items():
        reported = system_results[system]
        for stage in STAGES:
            for key, value in metrics["by_stage"][stage].items():
                compare_float(f"{system}.{stage}.{key}", value, reported["by_stage"][stage][key], failures)
                compare_float(f"stage_metrics.{stage}.{system}.{key}", value, stage_metrics[stage][system][key], failures)
        for key, value in metrics["overall"].items():
            compare_float(f"{system}.overall.{key}", value, reported["overall"][key], failures)
    for system, metrics in recomputed_frame.items():
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                compare_float(f"frame_sweep.{system}.{key}", float(value), float(frame_sweep[system][key]), failures)

    legacy = system_results["legacy_direct_text_ingress_baseline"]["overall"]
    best_system = aggregate.get("best_system")
    if best_system not in SYSTEMS:
        failures.append("aggregate best_system invalid")
    else:
        best = system_results[best_system]["overall"]
        if best["stress_success"] - legacy["stress_success"] < 0.20:
            failures.append("Text Field best did not beat legacy by >=0.20 stress success")
        if best["false_commit_rate"] != 0.0 or best["wrong_confident_rate"] != 0.0:
            failures.append("best Text Field has false commit or wrong confident")
    if system_results["text_field_4x128_overlap32"]["by_stage"]["T1_boundary_split_contrast"]["success"] <= system_results["text_field_4x128_overlap0"]["by_stage"]["T1_boundary_split_contrast"]["success"]:
        failures.append("overlap32 did not improve boundary split over overlap0")
    if system_results["keyword_shortcut_control"]["overall"]["wrong_confident_rate"] <= 0.30:
        failures.append("keyword shortcut control did not fail visibly")
    if system_results["oracle_text_field_reference"]["overall"]["stress_success"] != 1.0:
        failures.append("oracle reference is not ceiling")
    if recommendation.get("recommended_default") not in {"text_field_4x128_overlap32", "text_field_single_256"}:
        failures.append("recommended default not allowed")

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
    parser.add_argument("--out", default="target/pilot_wave/e56a_text_field_byte_frame_size_and_overlap_sweep")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e56a_text_field_byte_frame_size_and_overlap_sweep")
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

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any


MILESTONE = "E44B_PARALLEL_ABSTRACT_WIRE_STREAM_FIELD_SMOKE"
REQUIRED_CAPACITY_BITS = 5
DECISIONS = {
    "e44b_parallel_serial_capacity_detected",
    "e44b_wire_shape_tradeoff_detected",
    "e44b_wire_stream_unreliable",
    "e44b_headerless_stream_unreliable",
    "e44b_invalid_artifact_detected",
}
REQ_TARGET = [
    "backend_manifest.json",
    "task_generation_report.json",
    "wire_stream_grid_results.json",
    "wire_stream_table.md",
    "system_results.json",
    "mutation_report.json",
    "row_level_results.jsonl",
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
    "wire_stream_grid_results_sample.json",
    "row_level_sample.jsonl",
    "mutation_history_sample.jsonl",
    "deterministic_replay_sample_report.json",
    "sample_only_checker_result.json",
    "sample_schema.json",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def static_policy_check(runner: Path) -> list[str]:
    failures: list[str] = []
    source = runner.read_text(encoding="utf-8")
    ast.parse(source)
    for token in ["backward(", "AdamW", "SGD(", "optim.", "sympy", "eval("]:
        if token in source:
            failures.append(f"runner contains banned gradient/direct-solver token: {token}")
    return failures


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 1.0


def validate_row(row: dict[str, Any], failures: list[str], prefix: str) -> None:
    for key in [
        "system",
        "row_id",
        "split",
        "family",
        "wire_count",
        "bits_per_wire",
        "capacity_bits",
        "capacity_collision_rate",
        "expected_action",
        "action",
        "action_correct",
        "agency_decision_success",
        "trace_exact",
        "stream_code",
        "wire_bits",
        "false_commit",
        "missed_commit",
        "used_fixed_header",
    ]:
        if key not in row:
            failures.append(f"{prefix}: missing {key}")


def recompute(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for system in sorted({row["system"] for row in rows}):
        chunk = [row for row in rows if row["system"] == system]
        commit_rows = [row for row in chunk if row["expected_action"] == "COMMIT"]
        actual_commit = [row for row in chunk if row["action"] == "COMMIT"]
        out[system] = {
            "row_count": float(len(chunk)),
            "wire_count": float(chunk[0]["wire_count"]),
            "bits_per_wire": float(chunk[0]["bits_per_wire"]),
            "capacity_bits": float(chunk[0]["capacity_bits"]),
            "capacity_collision_rate": float(chunk[0]["capacity_collision_rate"]),
            "agency_decision_success": mean([1.0 if row["agency_decision_success"] else 0.0 for row in chunk]),
            "action_accuracy": mean([1.0 if row["action_correct"] else 0.0 for row in chunk]),
            "trace_exact_rate": mean([1.0 if row["trace_exact"] else 0.0 for row in chunk]),
            "false_commit_rate": mean([1.0 if row["false_commit"] else 0.0 for row in chunk]),
            "missed_commit_rate": mean([1.0 if row["missed_commit"] else 0.0 for row in chunk]),
            "commit_target_value_accuracy": mean([1.0 if row["payload_decode_correct"] else 0.0 for row in actual_commit]),
            "expected_commit_recovery": mean([1.0 if row["agency_decision_success"] else 0.0 for row in commit_rows]),
            "uses_fixed_header_rate": mean([1.0 if row["used_fixed_header"] else 0.0 for row in chunk]),
        }
    return out


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
    grid = read_json(sample_dir / "wire_stream_grid_results_sample.json")
    rows = read_jsonl(sample_dir / "row_level_sample.jsonl")
    history = read_jsonl(sample_dir / "mutation_history_sample.jsonl")
    if schema.get("milestone") != MILESTONE or schema.get("wire_stream_field") is not True:
        failures.append("sample schema missing E44B marker")
    if schema.get("gradient_descent_used") is not False:
        failures.append("sample schema gradient flag is not false")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid sample decision {aggregate.get('decision')}")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("sample deterministic replay did not pass")
    if not grid:
        failures.append("sample grid empty")
    if not rows or not history:
        failures.append("sample row/history artifact empty")
    for idx, row in enumerate(rows[:120]):
        validate_row(row, failures, f"sample row {idx}")
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

    failures.extend(static_policy_check(Path("scripts/probes/run_e44b_parallel_abstract_wire_stream_field_smoke.py")))
    manifest = read_json(out / "backend_manifest.json")
    aggregate = read_json(out / "aggregate_metrics.json")
    decision = read_json(out / "decision.json")
    summary = read_json(out / "summary.json")
    replay = read_json(out / "deterministic_replay.json")
    system_results = read_json(out / "system_results.json")
    grid = read_json(out / "wire_stream_grid_results.json")
    mutation = read_json(out / "mutation_report.json")
    rows = read_jsonl(out / "row_level_results.jsonl")
    progress = read_jsonl(out / "progress.jsonl")
    heartbeat = read_jsonl(out / "hardware_heartbeat.jsonl")

    if manifest.get("milestone") != MILESTONE:
        failures.append("milestone mismatch")
    if manifest.get("gradient_descent_used") is not False or manifest.get("optimizer_used") is not False or manifest.get("backprop_used") is not False:
        failures.append("gradient/optimizer/backprop flags are not false")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid decision {aggregate.get('decision')}")
    if decision.get("decision") != aggregate.get("decision") or summary.get("decision") != aggregate.get("decision"):
        failures.append("decision mismatch across artifacts")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("deterministic replay did not pass")
    if not rows or not progress or not heartbeat:
        failures.append("empty row/progress/heartbeat artifact")
    for idx, row in enumerate(rows[:200]):
        validate_row(row, failures, f"row {idx}")

    recomputed = recompute(rows)
    for system, metrics in recomputed.items():
        reported = system_results[system]["overall"]
        for key, value in metrics.items():
            compare_float(f"{system}.{key}", value, reported[key], failures)

    for system, report in mutation.items():
        if system.startswith("wire_stream_") or system == "headerless_stream_5x1_control":
            if report.get("mutation_attempts", 0) <= 0:
                failures.append(f"{system}: no mutation attempts")
            if report.get("rollback_count") != report.get("rejected"):
                failures.append(f"{system}: rollback mismatch")
            if not report.get("parameter_diff_written") or not report.get("parameter_diff_hash"):
                failures.append(f"{system}: missing parameter diff/hash")

    if aggregate.get("decision") == "e44b_parallel_serial_capacity_detected":
        max_wire = int(aggregate["max_wire"])
        max_bits = int(aggregate["max_bits"])
        for wire in range(1, max_wire + 1):
            for bits in range(1, max_bits + 1):
                item = grid[f"wire_stream_{wire}x{bits}"]
                if wire * bits < REQUIRED_CAPACITY_BITS and item["passes"]:
                    failures.append(f"below-capacity shape unexpectedly passed: {wire}x{bits}")
                if wire * bits >= REQUIRED_CAPACITY_BITS and not item["passes"]:
                    failures.append(f"above-capacity shape unexpectedly failed: {wire}x{bits}")
        if system_results["headerless_stream_5x1_control"]["overall"]["agency_decision_success"] >= 0.95:
            failures.append("headerless stream control unexpectedly passed")
        if system_results["random_stream_decoder_5x1_control"]["overall"]["agency_decision_success"] >= 0.95:
            failures.append("random stream decoder unexpectedly passed")

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

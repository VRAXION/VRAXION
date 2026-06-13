#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any


MILESTONE = "E45_EDGE_ABI_MUTATION_SEARCH_SPACE_FALLOFF_PROBE"
SYSTEMS = {
    "structured_w16_i32_reference",
    "structured_w64_i256_reference",
    "structured_w128_i1024_reference",
    "anonymous_w8_i32",
    "anonymous_w12_i32",
    "anonymous_w16_i32",
    "anonymous_w24_i32",
    "anonymous_w32_i32",
    "anonymous_w64_i32",
    "anonymous_w16_i256",
    "anonymous_w32_i256",
    "anonymous_w64_i256",
    "anonymous_w96_i256",
    "anonymous_w128_i256",
    "anonymous_w64_i1024",
    "anonymous_w128_i1024",
    "random_w16_i32_control",
}
MUTATED_SYSTEMS = {
    "anonymous_w8_i32",
    "anonymous_w12_i32",
    "anonymous_w16_i32",
    "anonymous_w24_i32",
    "anonymous_w32_i32",
    "anonymous_w64_i32",
    "anonymous_w16_i256",
    "anonymous_w32_i256",
    "anonymous_w64_i256",
    "anonymous_w96_i256",
    "anonymous_w128_i256",
    "anonymous_w64_i1024",
    "anonymous_w128_i1024",
}
DECISIONS = {
    "e45_anonymous_wide_bus_learning_falloff_detected",
    "e45_32bit_extended_lane_still_mutation_friendly",
    "e45_64bit_anonymous_lane_mutation_friendly",
    "e45_connection_needs_structured_layout",
    "e45_invalid_artifact_detected",
}
REQ_TARGET = [
    "backend_manifest.json",
    "search_space_report.json",
    "learning_dynamics_report.json",
    "final_candidates.json",
    "system_results.json",
    "row_level_results.jsonl",
    "learning_curve.jsonl",
    "aggregate_metrics.json",
    "deterministic_replay.json",
    "decision.json",
    "summary.json",
    "progress.jsonl",
    "hardware_heartbeat.jsonl",
    "partial_aggregate_snapshot.json",
    "results_table.md",
    "report.md",
]
REQ_SAMPLE = [
    "README.md",
    "artifact_sample_manifest.json",
    "aggregate_metrics_sample.json",
    "system_results_sample.json",
    "learning_dynamics_report_sample.json",
    "search_space_report_sample.json",
    "row_level_sample.jsonl",
    "learning_curve_sample.jsonl",
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
    return sum(values) / len(values) if values else 0.0


def split_metric(rows: list[dict[str, Any]], split: str, key: str) -> float:
    chunk = [row for row in rows if row["split"] == split]
    return mean([1.0 if row[key] else 0.0 for row in chunk])


def recompute(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for system in sorted({row["system"] for row in rows}):
        chunk = [row for row in rows if row["system"] == system]
        out[system] = {
            "row_count": float(len(chunk)),
            "edge_success": mean([1.0 if row["edge_success"] else 0.0 for row in chunk]),
            "heldout_success": split_metric(chunk, "heldout", "edge_success"),
            "ood_success": split_metric(chunk, "ood", "edge_success"),
            "counterfactual_success": split_metric(chunk, "counterfactual", "edge_success"),
            "adversarial_success": split_metric(chunk, "adversarial", "edge_success"),
            "wrong_commit_rate": mean([1.0 if row["wrong_commit"] else 0.0 for row in chunk]),
            "false_ask_rate": mean([1.0 if row["false_ask"] else 0.0 for row in chunk]),
        }
    return out


def compare_float(label: str, observed: float, reported: float, failures: list[str]) -> None:
    if not math.isclose(float(observed), float(reported), rel_tol=0.0, abs_tol=1e-9):
        failures.append(f"metric mismatch {label}: rows={observed} reported={reported}")


def validate_row(row: dict[str, Any], failures: list[str], prefix: str) -> None:
    for key in [
        "system",
        "row_id",
        "split",
        "intent",
        "bus_width",
        "intent_count",
        "data_bits",
        "mode",
        "expected_action",
        "action",
        "decoded_intent",
        "decode_correct",
        "edge_success",
        "wrong_commit",
        "false_ask",
        "candidate_indices_hash",
        "secret_indices_hash",
    ]:
        if key not in row:
            failures.append(f"{prefix}: missing {key}")


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
    curve = read_jsonl(sample_dir / "learning_curve_sample.jsonl")
    if schema.get("milestone") != MILESTONE or schema.get("edge_abi_falloff") is not True:
        failures.append("sample schema missing E45 marker")
    if schema.get("gradient_descent_used") is not False:
        failures.append("sample schema gradient flag is not false")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid sample decision {aggregate.get('decision')}")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("sample deterministic replay did not pass")
    if not rows or not curve:
        failures.append("sample rows/learning curve empty")
    for idx, row in enumerate(rows[:160]):
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
    failures.extend(static_policy_check(Path("scripts/probes/run_e45_edge_abi_mutation_search_space_falloff_probe.py")))
    manifest = read_json(out / "backend_manifest.json")
    aggregate = read_json(out / "aggregate_metrics.json")
    decision = read_json(out / "decision.json")
    summary = read_json(out / "summary.json")
    replay = read_json(out / "deterministic_replay.json")
    system_results = read_json(out / "system_results.json")
    dynamics = read_json(out / "learning_dynamics_report.json")
    search_report = read_json(out / "search_space_report.json")
    rows = read_jsonl(out / "row_level_results.jsonl")
    curve = read_jsonl(out / "learning_curve.jsonl")
    progress = read_jsonl(out / "progress.jsonl")
    heartbeat = read_jsonl(out / "hardware_heartbeat.jsonl")
    if manifest.get("milestone") != MILESTONE:
        failures.append("milestone mismatch")
    if manifest.get("gradient_descent_used") is not False or manifest.get("optimizer_used") is not False or manifest.get("backprop_used") is not False:
        failures.append("gradient/optimizer/backprop flags are not false")
    if set(system_results) != SYSTEMS or set(dynamics) != SYSTEMS or set(search_report) != SYSTEMS:
        failures.append("system/search/dynamics set mismatch")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid decision {aggregate.get('decision')}")
    if decision.get("decision") != aggregate.get("decision") or summary.get("decision") != aggregate.get("decision"):
        failures.append("decision mismatch across artifacts")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("deterministic replay did not pass")
    if not rows or not curve or not progress or not heartbeat:
        failures.append("empty row/curve/progress/heartbeat artifact")
    for idx, row in enumerate(rows[:260]):
        validate_row(row, failures, f"row {idx}")
    recomputed = recompute(rows)
    for system, metrics in recomputed.items():
        reported = system_results[system]["overall"]
        for key, value in metrics.items():
            compare_float(f"{system}.{key}", value, reported[key], failures)
    for system in MUTATED_SYSTEMS:
        report = dynamics[system]
        if report.get("mutation_attempts", 0) <= 0:
            failures.append(f"{system} had no mutation attempts")
        if report.get("rollback_count") != report.get("rejected"):
            failures.append(f"{system} rollback mismatch")
        if not report.get("parameter_diff_written") or not report.get("parameter_diff_hash"):
            failures.append(f"{system} missing parameter diff/hash")
        if report.get("learning_curve_rows", 0) <= 0:
            failures.append(f"{system} missing learning curve rows")
    if aggregate.get("decision") == "e45_anonymous_wide_bus_learning_falloff_detected":
        w32 = system_results["anonymous_w32_i256"]["overall"]
        w64 = system_results["anonymous_w64_i256"]["overall"]
        w128 = system_results["anonymous_w128_i256"]["overall"]
        w64_1024 = system_results["anonymous_w64_i1024"]["overall"]
        d32 = dynamics["anonymous_w32_i256"]
        d64 = dynamics["anonymous_w64_i256"]
        if w32["heldout_success"] < 0.95:
            failures.append("falloff decision without 32-bit i256 passing")
        if w64["heldout_success"] >= 0.95 and w128["heldout_success"] >= 0.95 and w64_1024["heldout_success"] >= 0.95:
            a32 = d32.get("attempts_to_95") or d32.get("mutation_attempts", 1)
            a64 = d64.get("attempts_to_95") or d64.get("mutation_attempts", 1)
            if a64 < 2.5 * max(1, a32):
                failures.append("falloff decision without failure or attempts explosion")
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

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any


MILESTONE = "E44C_RESERVE_WIRE_MASK_AND_NOISE_STRESS"
DECISIONS = {
    "e44c_masked_reserve_default_positive",
    "e44c_integrity_reserve_needed_for_universal_stress",
    "e44c_eight_bit_not_universal_under_silent_noise",
    "e44c_universal_wire_setup_selected",
    "e44c_invalid_artifact_detected",
}
SYSTEMS = {
    "oracle_integrity_reference",
    "unmasked8_full_payload_decoder",
    "active5_ignore_reserve_mask",
    "active5_visible_dropout_guard",
    "crc3_integrity_guard",
    "universal_mutated_wire_setup",
    "random_policy_control",
}
REQ_TARGET = [
    "backend_manifest.json",
    "stress_generation_report.json",
    "stress_barrage_results.json",
    "universal_mutation_report.json",
    "system_results.json",
    "row_level_results.jsonl",
    "aggregate_metrics.json",
    "deterministic_replay.json",
    "decision.json",
    "summary.json",
    "progress.jsonl",
    "hardware_heartbeat.jsonl",
    "partial_aggregate_snapshot.json",
    "stress_table.md",
    "report.md",
]
REQ_SAMPLE = [
    "README.md",
    "artifact_sample_manifest.json",
    "aggregate_metrics_sample.json",
    "system_results_sample.json",
    "stress_barrage_results_sample.json",
    "universal_mutation_report_sample.json",
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


def family_metric(rows: list[dict[str, Any]], family: str, key: str) -> float:
    chunk = [row for row in rows if row["family"] == family]
    return mean([1.0 if row[key] else 0.0 for row in chunk])


def recompute(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for system in sorted({row["system"] for row in rows}):
        chunk = [row for row in rows if row["system"] == system]
        out[system] = {
            "row_count": float(len(chunk)),
            "stress_success": mean([1.0 if row["stress_success"] else 0.0 for row in chunk]),
            "action_accuracy": mean([1.0 if row["action_correct"] else 0.0 for row in chunk]),
            "false_commit_rate": mean([1.0 if row["false_commit"] else 0.0 for row in chunk]),
            "wrong_commit_rate": mean([1.0 if row["wrong_commit"] else 0.0 for row in chunk]),
            "false_ask_rate": mean([1.0 if row["false_ask"] else 0.0 for row in chunk]),
            "reserve_noise_success": family_metric(chunk, "reserve_random_noise", "stress_success"),
            "reserve_adversarial_success": family_metric(chunk, "reserve_adversarial_noise", "stress_success"),
            "active_dropout_success": family_metric(chunk, "active_dropout_visible", "stress_success"),
            "active_bitflip_success": family_metric(chunk, "active_bitflip_silent", "stress_success"),
            "burst_noise_success": family_metric(chunk, "burst_noise_silent", "stress_success"),
            "known_permutation_success": family_metric(chunk, "known_wire_permutation", "stress_success"),
            "unknown_permutation_success": family_metric(chunk, "unknown_wire_permutation", "stress_success"),
        }
    return out


def compare_float(label: str, observed: float, reported: float, failures: list[str]) -> None:
    if not math.isclose(float(observed), float(reported), rel_tol=0.0, abs_tol=1e-9):
        failures.append(f"metric mismatch {label}: rows={observed} reported={reported}")


def validate_row(row: dict[str, Any], failures: list[str], prefix: str) -> None:
    for key in [
        "system",
        "mode",
        "row_id",
        "split",
        "family",
        "expected_action",
        "action",
        "action_correct",
        "decode_correct",
        "stress_success",
        "false_commit",
        "wrong_commit",
        "false_ask",
        "payload_bits",
        "target",
        "value",
        "decoded_target",
        "decoded_value",
        "reason_bits",
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
    history = read_jsonl(sample_dir / "mutation_history_sample.jsonl")
    if schema.get("milestone") != MILESTONE or schema.get("reserve_wire_stress") is not True:
        failures.append("sample schema missing E44C marker")
    if schema.get("gradient_descent_used") is not False:
        failures.append("sample schema gradient flag is not false")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid sample decision {aggregate.get('decision')}")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("sample deterministic replay did not pass")
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
    failures.extend(static_policy_check(Path("scripts/probes/run_e44c_reserve_wire_mask_and_noise_stress.py")))
    manifest = read_json(out / "backend_manifest.json")
    aggregate = read_json(out / "aggregate_metrics.json")
    decision = read_json(out / "decision.json")
    summary = read_json(out / "summary.json")
    replay = read_json(out / "deterministic_replay.json")
    system_results = read_json(out / "system_results.json")
    universal = read_json(out / "universal_mutation_report.json")
    rows = read_jsonl(out / "row_level_results.jsonl")
    progress = read_jsonl(out / "progress.jsonl")
    heartbeat = read_jsonl(out / "hardware_heartbeat.jsonl")
    if manifest.get("milestone") != MILESTONE:
        failures.append("milestone mismatch")
    if manifest.get("gradient_descent_used") is not False or manifest.get("optimizer_used") is not False or manifest.get("backprop_used") is not False:
        failures.append("gradient/optimizer/backprop flags are not false")
    if set(system_results) != SYSTEMS:
        failures.append("system set mismatch")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid decision {aggregate.get('decision')}")
    if decision.get("decision") != aggregate.get("decision") or summary.get("decision") != aggregate.get("decision"):
        failures.append("decision mismatch across artifacts")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("deterministic replay did not pass")
    if not rows or not progress or not heartbeat:
        failures.append("empty row/progress/heartbeat artifact")
    for idx, row in enumerate(rows[:220]):
        validate_row(row, failures, f"row {idx}")
    recomputed = recompute(rows)
    for system, metrics in recomputed.items():
        reported = system_results[system]["overall"]
        for key, value in metrics.items():
            compare_float(f"{system}.{key}", value, reported[key], failures)
    if universal.get("mutation_attempts", 0) <= 0:
        failures.append("universal mutation had no attempts")
    if universal.get("rollback_count") != universal.get("rejected"):
        failures.append("universal rollback mismatch")
    if not universal.get("parameter_diff_written") or not universal.get("parameter_diff_hash"):
        failures.append("universal mutation missing parameter diff/hash")
    if aggregate.get("decision") == "e44c_integrity_reserve_needed_for_universal_stress":
        if aggregate.get("universal_selected_mode") != "crc3_integrity":
            failures.append("integrity decision without crc3 selected mode")
        if system_results["crc3_integrity_guard"]["overall"]["active_bitflip_success"] < 0.95:
            failures.append("crc3 did not handle active bitflip")
        if system_results["active5_visible_dropout_guard"]["overall"]["reserve_noise_success"] < 0.95:
            failures.append("masked default did not handle reserve noise")
    if aggregate.get("decision") == "e44c_eight_bit_not_universal_under_silent_noise":
        if aggregate.get("universal_selected_mode") != "crc3_integrity":
            failures.append("silent-noise decision expected crc3 selected mode")
        if system_results["crc3_integrity_guard"]["overall"]["active_bitflip_success"] < 0.95:
            failures.append("crc3 did not handle active bitflip in silent-noise decision")
        universal_metrics = system_results["universal_mutated_wire_setup"]["overall"]
        if universal_metrics["false_commit_rate"] == 0 and universal_metrics["wrong_commit_rate"] == 0:
            failures.append("silent-noise decision without observed false/wrong commit")
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

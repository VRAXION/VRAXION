#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any


MILESTONE = "E44D_WIRE_BUS_WIDTH_AND_INTEGRITY_BUDGET_SWEEP"
DECISIONS = {
    "e44d_bus10_sufficient",
    "e44d_bus12_sufficient",
    "e44d_bus16_required",
    "e44d_integrity_reserve_tradeoff_persists",
    "e44d_no_universal_wire_bus_found",
    "e44d_invalid_artifact_detected",
}
SYSTEMS = {
    "oracle_reference",
    "bus8_5data_3reserve_masked",
    "bus8_5data_3crc",
    "bus10_5data_3crc_2reserve",
    "bus12_5data_4crc_3reserve",
    "bus16_5data_5ecc_6reserve",
    "universal_mutated_bus_policy",
    "random_policy_control",
}
REQ_TARGET = [
    "backend_manifest.json",
    "stress_generation_report.json",
    "bus_width_sweep_report.json",
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
    "bus_width_sweep_report_sample.json",
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
        reserve_success = mean(
            [
                family_metric(chunk, "reserve_random_noise", "stress_success"),
                family_metric(chunk, "reserve_adversarial_noise", "stress_success"),
                family_metric(chunk, "reserve_dropout", "stress_success"),
            ]
        )
        integrity_success = mean(
            [
                family_metric(chunk, "active_bitflip_silent", "stress_success"),
                family_metric(chunk, "double_alias_silent", "stress_success"),
                family_metric(chunk, "burst_noise_silent", "stress_success"),
            ]
        )
        out[system] = {
            "row_count": float(len(chunk)),
            "stress_success": mean([1.0 if row["stress_success"] else 0.0 for row in chunk]),
            "action_accuracy": mean([1.0 if row["action_correct"] else 0.0 for row in chunk]),
            "false_commit_rate": mean([1.0 if row["false_commit"] else 0.0 for row in chunk]),
            "wrong_commit_rate": mean([1.0 if row["wrong_commit"] else 0.0 for row in chunk]),
            "false_ask_rate": mean([1.0 if row["false_ask"] else 0.0 for row in chunk]),
            "reserve_success": reserve_success,
            "integrity_success": integrity_success,
            "reserve_noise_success": family_metric(chunk, "reserve_random_noise", "stress_success"),
            "reserve_adversarial_success": family_metric(chunk, "reserve_adversarial_noise", "stress_success"),
            "reserve_dropout_success": family_metric(chunk, "reserve_dropout", "stress_success"),
            "active_dropout_success": family_metric(chunk, "active_dropout_visible", "stress_success"),
            "active_bitflip_success": family_metric(chunk, "active_bitflip_silent", "stress_success"),
            "double_alias_success": family_metric(chunk, "double_alias_silent", "stress_success"),
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
        "config",
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
        "payload_width",
        "integrity_bits",
        "reserve_bits",
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
    if schema.get("milestone") != MILESTONE or schema.get("wire_bus_width_sweep") is not True:
        failures.append("sample schema missing E44D marker")
    if schema.get("gradient_descent_used") is not False:
        failures.append("sample schema gradient flag is not false")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid sample decision {aggregate.get('decision')}")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("sample deterministic replay did not pass")
    if not rows or not history:
        failures.append("sample row/history artifact empty")
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
    failures.extend(static_policy_check(Path("scripts/probes/run_e44d_wire_bus_width_and_integrity_budget_sweep.py")))
    manifest = read_json(out / "backend_manifest.json")
    aggregate = read_json(out / "aggregate_metrics.json")
    decision = read_json(out / "decision.json")
    summary = read_json(out / "summary.json")
    replay = read_json(out / "deterministic_replay.json")
    system_results = read_json(out / "system_results.json")
    mutation = read_json(out / "universal_mutation_report.json")
    sweep = read_json(out / "bus_width_sweep_report.json")
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
    for idx, row in enumerate(rows[:260]):
        validate_row(row, failures, f"row {idx}")
    recomputed = recompute(rows)
    for system, metrics in recomputed.items():
        reported = system_results[system]["overall"]
        for key, value in metrics.items():
            compare_float(f"{system}.{key}", value, reported[key], failures)
    if mutation.get("mutation_attempts", 0) <= 0:
        failures.append("universal mutation had no attempts")
    if mutation.get("rollback_count") != mutation.get("rejected"):
        failures.append("universal rollback mismatch")
    if not mutation.get("parameter_diff_written") or not mutation.get("parameter_diff_hash"):
        failures.append("universal mutation missing parameter diff/hash")
    if aggregate.get("minimal_passing_config") != "bus12_5data_4crc_3reserve" and aggregate.get("decision") == "e44d_bus12_sufficient":
        failures.append("bus12 decision without bus12 minimal passing config")
    if aggregate.get("decision") == "e44d_bus12_sufficient":
        bus10 = system_results["bus10_5data_3crc_2reserve"]["overall"]
        bus12 = system_results["bus12_5data_4crc_3reserve"]["overall"]
        if bus10["double_alias_success"] >= 0.95:
            failures.append("bus10 did not expose the CRC3 double-alias falsification")
        if bus12["stress_success"] < 0.95 or bus12["false_commit_rate"] != 0 or bus12["wrong_commit_rate"] != 0:
            failures.append("bus12 did not satisfy pass gate")
        if sweep.get("pass_gate", {}).get("bus12_5data_4crc_3reserve") is not True:
            failures.append("sweep report did not mark bus12 pass gate")
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

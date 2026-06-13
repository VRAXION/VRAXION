#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any


MILESTONE = "E58_STANDARD_IO_REGRESSION_BINARY_TEXT_EGRESS_CONFIRM"
SYSTEMS = {
    "legacy_standard_before_io_locks",
    "current_standard_without_bitslip_reassembly",
    "current_standard_with_bitslip_reassembly_candidate",
    "loose_start_only_unsafe",
    "direct_pocket_output_unsafe",
    "oracle_reference",
    "random_control",
}
STAGES = {
    "B0_binary_packet_clean",
    "B1_binary_packet_noise_10",
    "B2_binary_continuous_decoy",
    "B3_binary_bit_insert_slip",
    "B4_binary_bit_drop_slip",
    "T0_noisy_text_answerable",
    "T1_text_unresolved_must_ask",
    "T2_text_boundary_multiframe",
    "T3_real_like_weak_contrast",
    "O0_multires_output_consistency",
    "O1_stale_proposal_output_attack",
}
BINARY_STAGES = {stage for stage in STAGES if stage.startswith("B")}
TEXT_STAGES = {stage for stage in STAGES if stage.startswith("T")}
EGRESS_STAGES = {stage for stage in STAGES if stage.startswith("O")}
BITSLIP_STAGES = {"B3_binary_bit_insert_slip", "B4_binary_bit_drop_slip"}
DECISIONS = {
    "e58_standard_path_passes_with_bitslip_reassembly_candidate",
    "e58_standard_path_still_bitslip_limited",
    "e58_text_or_egress_regression_detected",
    "e58_unsafe_shortcut_or_stale_output_detected",
    "e58_invalid_artifact_detected",
}
REQ_TARGET = [
    "backend_manifest.json",
    "standard_io_manifest.json",
    "row_level_results.jsonl",
    "system_results.json",
    "stage_metrics.json",
    "binary_bitslip_report.json",
    "text_regression_report.json",
    "egress_examples_report.json",
    "multi_resolution_examples.json",
    "failure_examples.json",
    "training_history.json",
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
    "system_results_sample.json",
    "stage_metrics_sample.json",
    "multi_resolution_examples_sample.json",
    "failure_examples_sample.json",
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


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def static_policy_check() -> list[str]:
    failures: list[str] = []
    runner = Path("scripts/probes/run_e58_standard_io_regression_binary_text_egress_confirm.py")
    source = runner.read_text(encoding="utf-8")
    ast.parse(source)
    for token in ["backward(", "AdamW", "SGD(", "optim.", "sympy", "eval("]:
        if token in source:
            failures.append(f"runner contains banned gradient/direct-solver token: {token}")
    return failures


def summarize(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    stage_metrics: dict[str, Any] = {}
    system_results: dict[str, Any] = {}
    for system in SYSTEMS:
        system_rows = [row for row in rows if row["system"] == system]
        if not system_rows:
            continue
        by_stage: dict[str, Any] = {}
        for stage in STAGES:
            stage_rows = [row for row in system_rows if row["stage"] == stage]
            by_stage[stage] = {
                "closed_loop_success": mean([1.0 if row["closed_loop_success"] else 0.0 for row in stage_rows]),
                "trace_exact": mean([1.0 if row["trace_exact"] else 0.0 for row in stage_rows]),
                "false_commit_rate": mean([1.0 if row["false_commit"] else 0.0 for row in stage_rows]),
                "wrong_confident_rate": mean([1.0 if row["wrong_confident"] else 0.0 for row in stage_rows]),
                "false_ask_rate": mean([1.0 if row["false_ask"] else 0.0 for row in stage_rows]),
                "stale_output_leak_rate": mean([1.0 if row["stale_output_leak"] else 0.0 for row in stage_rows]),
                "multi_resolution_consistency": mean([1.0 if row["multi_resolution_consistency"] else 0.0 for row in stage_rows]),
                "net_utility": mean([float(row["net_utility"]) for row in stage_rows]),
                "row_count": len(stage_rows),
            }
        binary_rows = [row for row in system_rows if row["stage"] in BINARY_STAGES]
        bitslip_rows = [row for row in system_rows if row["stage"] in BITSLIP_STAGES]
        text_rows = [row for row in system_rows if row["stage"] in TEXT_STAGES]
        egress_rows = [row for row in system_rows if row["stage"] in EGRESS_STAGES]
        system_results[system] = {
            "by_stage": by_stage,
            "overall": {
                "closed_loop_success": mean([1.0 if row["closed_loop_success"] else 0.0 for row in system_rows]),
                "binary_success": mean([1.0 if row["binary_success"] else 0.0 for row in binary_rows]),
                "bitslip_success": mean([1.0 if row["bitslip_success"] else 0.0 for row in bitslip_rows]),
                "text_success": mean([1.0 if row["text_success"] else 0.0 for row in text_rows]),
                "egress_success": mean([1.0 if row["egress_success"] else 0.0 for row in egress_rows]),
                "trace_exact": mean([1.0 if row["trace_exact"] else 0.0 for row in system_rows]),
                "false_commit_rate": mean([1.0 if row["false_commit"] else 0.0 for row in system_rows]),
                "wrong_confident_rate": mean([1.0 if row["wrong_confident"] else 0.0 for row in system_rows]),
                "false_ask_rate": mean([1.0 if row["false_ask"] else 0.0 for row in system_rows]),
                "stale_output_leak_rate": mean([1.0 if row["stale_output_leak"] else 0.0 for row in system_rows]),
                "multi_resolution_consistency": mean([1.0 if row["multi_resolution_consistency"] else 0.0 for row in system_rows if row["requires_multires_output"]]),
                "net_utility": mean([float(row["net_utility"]) for row in system_rows]),
                "row_count": len(system_rows),
            },
        }
    for stage in STAGES:
        stage_metrics[stage] = {system: system_results[system]["by_stage"][stage] for system in SYSTEMS if system in system_results}
    return stage_metrics, system_results


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
    examples = read_json(sample_dir / "multi_resolution_examples_sample.json")
    failures_sample = read_json(sample_dir / "failure_examples_sample.json")
    rows = read_jsonl(sample_dir / "row_level_sample.jsonl")
    if schema.get("milestone") != MILESTONE:
        failures.append("sample schema milestone mismatch")
    if set(schema.get("systems", [])) != SYSTEMS:
        failures.append("sample systems mismatch")
    if set(schema.get("stages", [])) != STAGES:
        failures.append("sample stages mismatch")
    if schema.get("gradient_descent_used") is not False:
        failures.append("sample gradient flag is not false")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid sample decision {aggregate.get('decision')}")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("sample replay did not pass")
    if not rows:
        failures.append("sample row-level data empty")
    if len(examples) < 3:
        failures.append("sample lacks concrete multi-resolution examples")
    if not failures_sample:
        failures.append("sample lacks failure examples")
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
    reported_system_results = read_json(out / "system_results.json")
    reported_stage_metrics = read_json(out / "stage_metrics.json")
    examples = read_json(out / "multi_resolution_examples.json")
    failure_examples = read_json(out / "failure_examples.json")
    training = read_json(out / "training_history.json")
    rows = read_jsonl(out / "row_level_results.jsonl")

    if manifest.get("milestone") != MILESTONE:
        failures.append("manifest milestone mismatch")
    if set(manifest.get("systems", [])) != SYSTEMS:
        failures.append("manifest systems mismatch")
    if set(manifest.get("stages", [])) != STAGES:
        failures.append("manifest stages mismatch")
    if manifest.get("gradient_descent_used") is not False or manifest.get("optimizer_used") is not False or manifest.get("backprop_used") is not False:
        failures.append("gradient/optimizer/backprop flag is not false")
    if aggregate.get("decision") != decision.get("decision") or decision.get("decision") not in DECISIONS:
        failures.append("decision artifact mismatch or invalid decision")
    if summary.get("decision") != aggregate.get("decision"):
        failures.append("summary decision mismatch")
    if not rows:
        failures.append("row-level results empty")
    if {row["system"] for row in rows} != SYSTEMS:
        failures.append("row system coverage mismatch")
    if {row["stage"] for row in rows} != STAGES:
        failures.append("row stage coverage mismatch")

    stage_metrics, system_results = summarize(rows)
    for system in SYSTEMS:
        for key, value in system_results[system]["overall"].items():
            if isinstance(value, (int, float)):
                compare_float(f"{system}.overall.{key}", value, reported_system_results[system]["overall"][key], failures)
    for stage in STAGES:
        for system in SYSTEMS:
            compare_float(f"{stage}.{system}.closed_loop_success", stage_metrics[stage][system]["closed_loop_success"], reported_stage_metrics[stage][system]["closed_loop_success"], failures)

    current = reported_system_results["current_standard_without_bitslip_reassembly"]["overall"]
    candidate = reported_system_results["current_standard_with_bitslip_reassembly_candidate"]["overall"]
    unsafe = reported_system_results["direct_pocket_output_unsafe"]["overall"]
    if aggregate.get("decision") == "e58_standard_path_passes_with_bitslip_reassembly_candidate":
        if candidate["closed_loop_success"] < 0.98:
            failures.append("confirmed decision but candidate closed-loop below 0.98")
        if candidate["bitslip_success"] < 0.98:
            failures.append("confirmed decision but candidate bitslip below 0.98")
        if candidate["text_success"] < 0.98 or candidate["egress_success"] < 0.98:
            failures.append("confirmed decision but candidate text/egress below 0.98")
        if candidate["false_commit_rate"] > 0.01 or candidate["stale_output_leak_rate"] > 0.01:
            failures.append("confirmed decision but candidate has unsafe commits/leaks")
        if current["bitslip_success"] >= 0.70:
            failures.append("confirmed decision but current path did not expose bitslip gap")
    if unsafe["stale_output_leak_rate"] <= 0.05:
        failures.append("unsafe direct output control did not expose stale leak")
    if len(examples) < 4:
        failures.append("not enough multi-resolution examples")
    for example in examples:
        for key in ["compact_output", "short_output", "long_output", "consistency_hash"]:
            if key not in example:
                failures.append(f"example missing {key}")
    if not failure_examples:
        failures.append("failure examples missing")
    if training.get("gradient_descent_used") is not False or training.get("accepted_mutations", 0) <= 0 or training.get("rollback_count", 0) <= 0:
        failures.append("training history missing real mutation-style accept/reject/rollback counters")

    for name, expected_hash in replay.get("artifact_hashes", {}).items():
        path = out / name
        if not path.exists():
            failures.append(f"replay hash references missing artifact {name}")
        elif file_sha256(path) != expected_hash:
            failures.append(f"deterministic replay hash mismatch for {name}")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("deterministic replay did not pass")
    sample_result = validate_sample(sample_dir)
    if not sample_result["passed"]:
        failures.extend([f"sample: {failure}" for failure in sample_result["failures"]])
    result = {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "decision": aggregate.get("decision"),
        "run_id": aggregate.get("run_id"),
        "sample_only_checker_passed": sample_result["passed"],
    }
    if write_summary:
        write_json(out / "checker_summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e58_standard_io_regression_binary_text_egress_confirm")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e58_standard_io_regression_binary_text_egress_confirm")
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

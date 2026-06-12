#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any


MILESTONE = "E34D_CONTINUOUS_BINARY_STREAM_FRAMING_AND_RESYNC_PROBE"
DECISIONS = {
    "e34d_framing_resync_guard_positive",
    "e34d_crc_guard_positive_but_resync_brittle",
    "e34d_requested_feature_guard_positive",
    "e34d_eof_length_crc_partial",
    "e34d_framing_still_bottleneck",
    "e34d_shortcut_or_task_artifact_detected",
    "e34d_artifact_invalid",
}
SYSTEMS = {
    "start_only_baseline",
    "start_end_marker",
    "start_length_end",
    "start_length_crc_end",
    "crc_end_requested_feature_guard",
    "multi_frame_resync_guard",
    "first_sync_shortcut_control",
    "oracle_framing_reference",
}
SPLITS = {
    "packet_clean",
    "packet_noise_10",
    "continuous_stream",
    "continuous_bit_insert",
    "continuous_bit_drop",
    "adversarial_sync_decoy",
}
REQ_TARGET = [
    "backend_manifest.json",
    "task_generation_report.json",
    "framing_protocol_report.json",
    "policy_initial_state.json",
    "policy_final_state.json",
    "parameter_diff.json",
    "mutation_history.jsonl",
    "row_level_results.jsonl",
    "system_results.json",
    "aggregate_metrics.json",
    "deterministic_replay.json",
    "resource_usage_report.json",
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
    "system_metrics_sample.json",
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


def metric(rows: list[dict[str, Any]], key: str) -> float:
    return sum(1.0 if row.get(key) else 0.0 for row in rows) / len(rows) if rows else 0.0


def mean_value(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row.get(key, 0.0)) for row in rows) / len(rows) if rows else 0.0


def static_runner_policy_check(runner: Path) -> list[str]:
    failures: list[str] = []
    source = runner.read_text(encoding="utf-8")
    ast.parse(source)
    for token in ["torch", "tensorflow", "jax", "backward(", "AdamW", "SGD(", "optim.", "sympy", "eval("]:
        if token in source:
            failures.append(f"runner contains banned gradient/oracle token: {token}")
    return failures


def recompute_system_metrics(rows: list[dict[str, Any]], system: str) -> dict[str, Any]:
    sys_rows = [row for row in rows if row.get("system") == system]
    split_rows = {split: [row for row in sys_rows if row.get("split") == split] for split in SPLITS}
    return {
        "row_count": len(sys_rows),
        "closed_loop_success": metric(sys_rows, "closed_loop_success"),
        "answer_correct": metric(sys_rows, "answer_correct"),
        "trace_exact": metric(sys_rows, "trace_exact"),
        "wrong_confident_answer": metric(sys_rows, "wrong_confident_answer"),
        "false_ask": metric(sys_rows, "false_ask"),
        "redundant_actions": metric(sys_rows, "redundant_action"),
        "avg_steps": mean_value(sys_rows, "step_count"),
        "avg_inspects": mean_value(sys_rows, "inspect_count"),
        "binary_ingress_accuracy": mean_value(sys_rows, "binary_ingress_accuracy"),
        "accepted_flow_write_accuracy": mean_value(sys_rows, "accepted_flow_write_accuracy"),
        "frame_sync_accuracy": mean_value(sys_rows, "frame_sync_accuracy"),
        "false_frame_commit_rate": mean_value(sys_rows, "false_frame_commit_rate"),
        "wrong_feature_write_rate": mean_value(sys_rows, "wrong_feature_write_rate"),
        "avg_rejected_packets": mean_value(sys_rows, "rejected_packet_count"),
        "first_useful_evidence_action": metric(sys_rows, "first_useful_evidence_action"),
        "split_closed_loop_success": {split: metric(split_rows[split], "closed_loop_success") for split in SPLITS},
        "split_false_frame_commit_rate": {split: mean_value(split_rows[split], "false_frame_commit_rate") for split in SPLITS},
        "split_wrong_feature_write_rate": {split: mean_value(split_rows[split], "wrong_feature_write_rate") for split in SPLITS},
        "split_accepted_flow_write_accuracy": {split: mean_value(split_rows[split], "accepted_flow_write_accuracy") for split in SPLITS},
        "split_avg_steps": {split: mean_value(split_rows[split], "step_count") for split in SPLITS},
    }


def compare_float(label: str, observed: float, reported: float, failures: list[str], tolerance: float = 1e-12) -> None:
    if not math.isclose(float(observed), float(reported), rel_tol=0.0, abs_tol=tolerance):
        failures.append(f"metric mismatch {label}: rows={observed} reported={reported}")


def validate_sample(sample_dir: Path) -> dict[str, Any]:
    failures: list[str] = []
    for name in REQ_SAMPLE:
        if not (sample_dir / name).exists():
            failures.append(f"missing sample artifact {name}")
    if failures:
        return {"passed": False, "failure_count": len(failures), "failures": failures}
    aggregate = read_json(sample_dir / "aggregate_metrics_sample.json")
    system_metrics = read_json(sample_dir / "system_metrics_sample.json")
    schema = read_json(sample_dir / "sample_schema.json")
    replay = read_json(sample_dir / "deterministic_replay_sample_report.json")
    rows = read_jsonl(sample_dir / "row_level_sample.jsonl")
    history = read_jsonl(sample_dir / "mutation_history_sample.jsonl")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid sample decision {aggregate.get('decision')}")
    if schema.get("milestone") != MILESTONE:
        failures.append("sample schema milestone mismatch")
    if schema.get("binary_stream_resync") is not True:
        failures.append("sample schema missing binary_stream_resync=true")
    if schema.get("gradient_descent_used") is not False:
        failures.append("sample schema gradient flag is not false")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("sample deterministic replay did not pass")
    if set(system_metrics) != SYSTEMS:
        failures.append("sample system metrics missing required systems")
    if not rows or any("ingress_events" not in row or "initial_bits" not in row for row in rows):
        failures.append("sample row-level binary trace missing")
    if not history:
        failures.append("sample mutation history empty")
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

    runner = Path(__file__).resolve().with_name("run_e34d_continuous_binary_stream_framing_and_resync_probe.py")
    failures.extend(static_runner_policy_check(runner))
    manifest = read_json(out / "backend_manifest.json")
    task = read_json(out / "task_generation_report.json")
    protocol = read_json(out / "framing_protocol_report.json")
    aggregate = read_json(out / "aggregate_metrics.json")
    decision = read_json(out / "decision.json")
    summary = read_json(out / "summary.json")
    parameter_diff = read_json(out / "parameter_diff.json")
    replay = read_json(out / "deterministic_replay.json")
    system_results = read_json(out / "system_results.json")
    rows = read_jsonl(out / "row_level_results.jsonl")
    history = read_jsonl(out / "mutation_history.jsonl")
    progress = read_jsonl(out / "progress.jsonl")
    heartbeat = read_jsonl(out / "hardware_heartbeat.jsonl")

    if manifest.get("milestone") != MILESTONE or aggregate.get("milestone") != MILESTONE:
        failures.append("milestone mismatch")
    if manifest.get("gradient_descent_used") is not False or manifest.get("optimizer_used") is not False or manifest.get("backprop_used") is not False:
        failures.append("gradient/optimizer/backprop flags are not false")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid decision {aggregate.get('decision')}")
    if decision.get("decision") != aggregate.get("decision") or summary.get("decision") != aggregate.get("decision"):
        failures.append("decision mismatch across artifacts")
    if set(system_results) != SYSTEMS or set(manifest.get("systems", [])) != SYSTEMS:
        failures.append("missing required systems")
    if set(task.get("splits", [])) != SPLITS:
        failures.append("task split mismatch")
    if protocol.get("requested_feature_guard_tested") is not True or protocol.get("multi_frame_hypothesis_tested") is not True:
        failures.append("protocol report missing guard/multi-frame flags")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("deterministic replay did not pass")
    if parameter_diff.get("changed") is not True or parameter_diff.get("initial_hash") == parameter_diff.get("final_hash"):
        failures.append("missing parameter diff")
    if int(parameter_diff.get("accepted_mutations", 0)) <= 0:
        failures.append("no accepted mutations")
    if int(parameter_diff.get("rejected_mutations", 0)) <= 0 or int(parameter_diff.get("rollback_count", 0)) <= 0:
        failures.append("missing rejected/rollback mutations")
    if not rows or not history or not progress or not heartbeat:
        failures.append("empty required row/history/progress/heartbeat artifact")
    if any(row.get("system") not in SYSTEMS for row in rows):
        failures.append("unknown system in row-level results")
    if any(row.get("split") not in SPLITS for row in rows):
        failures.append("unknown split in row-level results")
    if any("ingress_events" not in row or "initial_bits" not in row or not row.get("actions") for row in rows):
        failures.append("row-level trace fields missing")

    if set(system_results) == SYSTEMS:
        for system in SYSTEMS:
            recomputed = recompute_system_metrics(rows, system)
            reported = system_results[system]
            if recomputed["row_count"] != reported.get("row_count"):
                failures.append(f"row_count mismatch {system}")
            for key in [
                "closed_loop_success",
                "answer_correct",
                "trace_exact",
                "wrong_confident_answer",
                "false_ask",
                "redundant_actions",
                "avg_steps",
                "avg_inspects",
                "binary_ingress_accuracy",
                "accepted_flow_write_accuracy",
                "frame_sync_accuracy",
                "false_frame_commit_rate",
                "wrong_feature_write_rate",
                "avg_rejected_packets",
                "first_useful_evidence_action",
            ]:
                compare_float(f"{system}.{key}", recomputed[key], reported.get(key), failures)
            for split in SPLITS:
                compare_float(f"{system}.split_success.{split}", recomputed["split_closed_loop_success"][split], reported.get("split_closed_loop_success", {}).get(split), failures)
                compare_float(f"{system}.split_false_frame.{split}", recomputed["split_false_frame_commit_rate"][split], reported.get("split_false_frame_commit_rate", {}).get(split), failures)

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
        parser.error("--out and --artifact-sample-dir are required unless --sample-only is used")
    result = validate_target(Path(args.out), Path(args.artifact_sample_dir), args.write_summary)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

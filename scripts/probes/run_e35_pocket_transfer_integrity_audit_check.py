#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any


MILESTONE = "E35_POCKET_TRANSFER_INTEGRITY_AUDIT"
DECISIONS = {
    "e35_pocket_transfer_integrity_confirmed",
    "e35_transfer_partial",
    "e35_no_transfer_detected",
    "e35_negative_transfer_detected",
    "e35_invalid_artifact_or_oracle_detected",
}
SYSTEMS = {
    "scratch_no_pocket",
    "frozen_import_pocket",
    "imported_plus_small_adapter",
    "full_retrain_from_import",
    "wrong_pocket_negative_control",
    "protocol_ablation_no_import",
    "oracle_invalid_control",
}
SPLITS = {
    "same_packet_clean",
    "same_continuous_stream",
    "target_packet_clean",
    "target_continuous_stream",
    "target_adversarial_decoy",
    "target_bit_insert",
    "target_bit_drop",
}
REQ_TARGET = [
    "backend_manifest.json",
    "source_task_report.json",
    "pocket_archive_report.json",
    "pocket_manifest.json",
    "transfer_task_report.json",
    "transfer_tests.json",
    "adapter_report.json",
    "source_policy_final_state.json",
    "repair_policy_final_state.json",
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
    "report.md",
]
REQ_SAMPLE = [
    "README.md",
    "artifact_sample_manifest.json",
    "aggregate_metrics_sample.json",
    "system_metrics_sample.json",
    "row_level_sample.jsonl",
    "mutation_history_sample.jsonl",
    "pocket_manifest_sample.json",
    "transfer_tests_sample.json",
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


def compare_float(label: str, observed: float, reported: float, failures: list[str], tolerance: float = 1e-12) -> None:
    if not math.isclose(float(observed), float(reported), rel_tol=0.0, abs_tol=tolerance):
        failures.append(f"metric mismatch {label}: rows={observed} reported={reported}")


def static_runner_policy_check(runner: Path) -> list[str]:
    failures: list[str] = []
    source = runner.read_text(encoding="utf-8")
    ast.parse(source)
    for token in ["torch", "tensorflow", "jax", "backward(", "AdamW", "SGD(", "optim.", "sympy", "eval(", "hash("]:
        if token in source:
            failures.append(f"runner contains banned nondeterministic/gradient/oracle token: {token}")
    return failures


def recompute_system_metrics(rows: list[dict[str, Any]], system: str) -> dict[str, Any]:
    sys_rows = [row for row in rows if row.get("system") == system]
    split_rows = {split: [row for row in sys_rows if row.get("split") == split] for split in SPLITS}
    stable_rows = [row for row in sys_rows if row.get("split") in {"target_packet_clean", "target_continuous_stream", "target_adversarial_decoy"}]
    bitslip_rows = [row for row in sys_rows if row.get("split") in {"target_bit_insert", "target_bit_drop"}]
    same_rows = [row for row in sys_rows if str(row.get("split", "")).startswith("same_")]
    target_rows = [row for row in sys_rows if str(row.get("split", "")).startswith("target_")]
    return {
        "row_count": len(sys_rows),
        "closed_loop_success": metric(sys_rows, "closed_loop_success"),
        "same_world_success": metric(same_rows, "closed_loop_success"),
        "target_world_success": metric(target_rows, "closed_loop_success"),
        "stable_target_success": metric(stable_rows, "closed_loop_success"),
        "bitslip_target_success": metric(bitslip_rows, "closed_loop_success"),
        "target_wrong_feature_write_rate": mean_value(target_rows, "wrong_feature_write_rate"),
        "target_false_frame_commit_rate": mean_value(target_rows, "false_frame_commit_rate"),
        "stable_target_wrong_feature_write_rate": mean_value(stable_rows, "wrong_feature_write_rate"),
        "stable_target_false_frame_commit_rate": mean_value(stable_rows, "false_frame_commit_rate"),
        "bitslip_target_wrong_feature_write_rate": mean_value(bitslip_rows, "wrong_feature_write_rate"),
        "bitslip_target_false_frame_commit_rate": mean_value(bitslip_rows, "false_frame_commit_rate"),
        "answer_correct": metric(sys_rows, "answer_correct"),
        "trace_exact": metric(sys_rows, "trace_exact"),
        "wrong_confident_answer": metric(sys_rows, "wrong_confident_answer"),
        "false_ask": metric(sys_rows, "false_ask"),
        "redundant_actions": metric(sys_rows, "redundant_action"),
        "avg_steps": mean_value(sys_rows, "step_count"),
        "binary_ingress_accuracy": mean_value(sys_rows, "binary_ingress_accuracy"),
        "accepted_flow_write_accuracy": mean_value(sys_rows, "accepted_flow_write_accuracy"),
        "frame_sync_accuracy": mean_value(sys_rows, "frame_sync_accuracy"),
        "false_frame_commit_rate": mean_value(sys_rows, "false_frame_commit_rate"),
        "wrong_feature_write_rate": mean_value(sys_rows, "wrong_feature_write_rate"),
        "split_closed_loop_success": {split: metric(split_rows[split], "closed_loop_success") for split in SPLITS},
        "split_wrong_feature_write_rate": {split: mean_value(split_rows[split], "wrong_feature_write_rate") for split in SPLITS},
        "split_false_frame_commit_rate": {split: mean_value(split_rows[split], "false_frame_commit_rate") for split in SPLITS},
    }


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
    pocket = read_json(sample_dir / "pocket_manifest_sample.json")
    transfer_tests = read_json(sample_dir / "transfer_tests_sample.json")
    rows = read_jsonl(sample_dir / "row_level_sample.jsonl")
    history = read_jsonl(sample_dir / "mutation_history_sample.jsonl")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid sample decision {aggregate.get('decision')}")
    if schema.get("milestone") != MILESTONE or schema.get("pocket_transfer") is not True:
        failures.append("sample schema missing E35 pocket transfer marker")
    if schema.get("gradient_descent_used") is not False:
        failures.append("sample schema gradient flag is not false")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("sample deterministic replay did not pass")
    if set(system_metrics) != SYSTEMS:
        failures.append("sample system metrics missing required systems")
    if pocket.get("pocket_id") != "ProtocolFramingIngressPocket":
        failures.append("sample pocket manifest mismatch")
    for key in ["same_codebook_zero_shot", "shifted_codebook_adapter", "wrong_pocket_target"]:
        if key not in transfer_tests:
            failures.append(f"sample transfer tests missing {key}")
    if not rows or any("ingress_events" not in row or "adapter_hash" not in row for row in rows):
        failures.append("sample row-level transfer trace missing")
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

    runner = Path(__file__).resolve().with_name("run_e35_pocket_transfer_integrity_audit.py")
    failures.extend(static_runner_policy_check(runner))
    manifest = read_json(out / "backend_manifest.json")
    archive = read_json(out / "pocket_archive_report.json")
    pocket = read_json(out / "pocket_manifest.json")
    transfer_task = read_json(out / "transfer_task_report.json")
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
    if set(transfer_task.get("splits", [])) != SPLITS:
        failures.append("transfer split mismatch")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("deterministic replay did not pass")
    if pocket.get("pocket_id") != "ProtocolFramingIngressPocket" or "frozen_params_sha256" not in pocket:
        failures.append("pocket manifest missing frozen ProtocolFramingIngressPocket")
    pocket_dir = Path(str(archive.get("pocket_dir", "")))
    for name in ["pocket_manifest.json", "pocket_contract.md", "frozen_params.json", "lineage.json", "source_metrics.json", "transfer_tests.json", "safety_report.json"]:
        if not (pocket_dir / name).exists():
            failures.append(f"missing pocket archive file {name}")
    for section in ["source", "adapter", "scratch_adapter", "repair"]:
        if section not in parameter_diff:
            failures.append(f"parameter_diff missing {section}")
    if int(parameter_diff.get("source", {}).get("accepted_mutations", 0)) <= 0:
        failures.append("source pocket training had no accepted mutations")
    if int(parameter_diff.get("adapter", {}).get("accepted_mutations", 0)) <= 0:
        failures.append("target adapter had no accepted mutations")
    if int(parameter_diff.get("adapter", {}).get("rejected_mutations", 0)) <= 0:
        failures.append("target adapter had no rejected/rollback mutations")
    if not rows or not history or not progress or not heartbeat:
        failures.append("empty required row/history/progress/heartbeat artifact")
    if any(row.get("system") not in SYSTEMS for row in rows):
        failures.append("unknown system in row-level results")
    if any(row.get("split") not in SPLITS for row in rows):
        failures.append("unknown split in row-level results")
    if any("ingress_events" not in row or "adapter_hash" not in row or "initial_bits" not in row for row in rows):
        failures.append("row-level transfer fields missing")

    if set(system_results) == SYSTEMS:
        for system in SYSTEMS:
            recomputed = recompute_system_metrics(rows, system)
            reported = system_results[system]
            if recomputed["row_count"] != reported.get("row_count"):
                failures.append(f"row_count mismatch {system}")
            for key in [
                "closed_loop_success",
                "same_world_success",
                "target_world_success",
                "stable_target_success",
                "bitslip_target_success",
                "target_wrong_feature_write_rate",
                "target_false_frame_commit_rate",
                "stable_target_wrong_feature_write_rate",
                "stable_target_false_frame_commit_rate",
                "bitslip_target_wrong_feature_write_rate",
                "bitslip_target_false_frame_commit_rate",
                "answer_correct",
                "trace_exact",
                "wrong_confident_answer",
                "false_ask",
                "redundant_actions",
                "avg_steps",
                "binary_ingress_accuracy",
                "accepted_flow_write_accuracy",
                "frame_sync_accuracy",
                "false_frame_commit_rate",
                "wrong_feature_write_rate",
            ]:
                compare_float(f"{system}.{key}", recomputed[key], reported.get(key), failures)
            for split in SPLITS:
                compare_float(f"{system}.split_success.{split}", recomputed["split_closed_loop_success"][split], reported.get("split_closed_loop_success", {}).get(split), failures)
                compare_float(f"{system}.split_wrong_feature.{split}", recomputed["split_wrong_feature_write_rate"][split], reported.get("split_wrong_feature_write_rate", {}).get(split), failures)

    tests = aggregate.get("transfer_tests", {})
    if tests.get("wrong_pocket_target", 1.0) >= tests.get("shifted_codebook_adapter", 0.0) - 0.02:
        failures.append("wrong pocket control matches imported adapter")
    if tests.get("same_codebook_zero_shot", 0.0) < 0.80:
        failures.append("frozen same-codebook import too weak to support transfer audit")

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

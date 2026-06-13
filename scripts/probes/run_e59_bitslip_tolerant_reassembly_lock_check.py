#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any


MILESTONE = "E59_BITSLIP_TOLERANT_REASSEMBLY_LOCK"

SYSTEMS = {
    "strict_single_offset_full_guard",
    "end_marker_only_decoder",
    "loose_start_only_decoder",
    "multi_offset_crc_no_feature_guard",
    "multi_offset_crc_requested_no_ambiguity_guard",
    "bitslip_tolerant_reassembly_lock",
    "oracle_frame_reference",
    "random_control",
}

STAGES = {
    "P0_clean_packet",
    "P1_noise_with_crc",
    "P2_continuous_decoy_false_start",
    "P3_single_bit_insert_before_frame",
    "P4_single_bit_drop_before_frame",
    "P5_payload_slip_with_repeated_frame",
    "P6_adversarial_sync_decoy_before_valid",
    "P7_wrong_feature_valid_crc_only",
    "P8_truncated_packet_must_defer",
    "P9_conflicting_duplicate_frames_must_defer",
}

BITSLIP_STAGES = {
    "P3_single_bit_insert_before_frame",
    "P4_single_bit_drop_before_frame",
    "P5_payload_slip_with_repeated_frame",
}

DEFER_STAGES = {
    "P7_wrong_feature_valid_crc_only",
    "P8_truncated_packet_must_defer",
    "P9_conflicting_duplicate_frames_must_defer",
}

DECISIONS = {
    "e59_bitslip_tolerant_reassembly_locked",
    "e59_reassembly_still_false_frame_limited",
    "e59_requested_feature_guard_required",
    "e59_ambiguity_guard_required",
    "e59_invalid_artifact_detected",
}

REQ_TARGET = [
    "backend_manifest.json",
    "ingress_protocol_manifest.json",
    "row_level_results.jsonl",
    "system_results.json",
    "stage_metrics.json",
    "reassembly_report.json",
    "false_frame_report.json",
    "requested_feature_guard_report.json",
    "ambiguity_guard_report.json",
    "reassembly_examples.json",
    "failure_examples.json",
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
    "reassembly_examples_sample.json",
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
    runner = Path("scripts/probes/run_e59_bitslip_tolerant_reassembly_lock.py")
    source = runner.read_text(encoding="utf-8")
    ast.parse(source)
    banned = ["backward(", "AdamW", "SGD(", "optim.", "sympy", "eval("]
    for token in banned:
        if token in source:
            failures.append(f"runner contains banned gradient/direct-solver token: {token}")
    for required in ["decoded_feature == requested_feature", "conflicting_requested_frames_defer", "START_SYNC", "END_SYNC", "checksum"]:
        if required not in source:
            failures.append(f"runner missing required reassembly guard marker: {required}")
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
                "bitslip_recovery": mean([1.0 if row["bitslip_recovery"] else 0.0 for row in stage_rows]),
                "trace_exact": mean([1.0 if row["trace_exact"] else 0.0 for row in stage_rows]),
                "false_commit_rate": mean([1.0 if row["false_commit"] else 0.0 for row in stage_rows]),
                "false_frame_commit_rate": mean([1.0 if row["false_frame_commit"] else 0.0 for row in stage_rows]),
                "wrong_feature_write_rate": mean([1.0 if row["wrong_feature_write"] else 0.0 for row in stage_rows]),
                "false_defer_rate": mean([1.0 if row["false_defer"] else 0.0 for row in stage_rows]),
                "avg_candidate_count": mean([float(row["candidate_count"]) for row in stage_rows]),
                "avg_crc_pass_count": mean([float(row["crc_pass_count"]) for row in stage_rows]),
                "net_utility": mean([float(row["net_utility"]) for row in stage_rows]),
                "row_count": len(stage_rows),
            }
        bitslip_rows = [row for row in system_rows if row["stage"] in BITSLIP_STAGES]
        defer_rows = [row for row in system_rows if row["stage"] in DEFER_STAGES]
        system_results[system] = {
            "by_stage": by_stage,
            "overall": {
                "closed_loop_success": mean([1.0 if row["closed_loop_success"] else 0.0 for row in system_rows]),
                "bitslip_recovery": mean([1.0 if row["bitslip_recovery"] else 0.0 for row in bitslip_rows]),
                "defer_success": mean([1.0 if row["closed_loop_success"] else 0.0 for row in defer_rows]),
                "trace_exact": mean([1.0 if row["trace_exact"] else 0.0 for row in system_rows]),
                "false_commit_rate": mean([1.0 if row["false_commit"] else 0.0 for row in system_rows]),
                "false_frame_commit_rate": mean([1.0 if row["false_frame_commit"] else 0.0 for row in system_rows]),
                "wrong_feature_write_rate": mean([1.0 if row["wrong_feature_write"] else 0.0 for row in system_rows]),
                "false_defer_rate": mean([1.0 if row["false_defer"] else 0.0 for row in system_rows]),
                "ambiguity_reject_rate": mean([1.0 if row["ambiguity_rejected"] else 0.0 for row in system_rows if row["stage"] == "P9_conflicting_duplicate_frames_must_defer"]),
                "avg_candidate_count": mean([float(row["candidate_count"]) for row in system_rows]),
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
    rows = read_jsonl(sample_dir / "row_level_sample.jsonl")
    examples = read_json(sample_dir / "reassembly_examples_sample.json")
    failure_examples = read_json(sample_dir / "failure_examples_sample.json")

    if schema.get("milestone") != MILESTONE:
        failures.append("sample schema milestone mismatch")
    if set(schema.get("systems", [])) != SYSTEMS:
        failures.append("sample systems mismatch")
    if set(schema.get("stages", [])) != STAGES:
        failures.append("sample stages mismatch")
    if schema.get("gradient_descent_used") is not False or schema.get("optimizer_used") is not False or schema.get("backprop_used") is not False:
        failures.append("sample gradient flags are not false")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid sample decision {aggregate.get('decision')}")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("sample replay did not pass")
    if {row["system"] for row in rows} != SYSTEMS:
        failures.append("sample rows do not cover all systems")
    if {row["stage"] for row in rows} != STAGES:
        failures.append("sample rows do not cover all stages")
    if len(examples) < 4:
        failures.append("sample lacks concrete reassembly examples")
    if len(failure_examples) < 3:
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
    protocol = read_json(out / "ingress_protocol_manifest.json")
    aggregate = read_json(out / "aggregate_metrics.json")
    decision = read_json(out / "decision.json")
    summary = read_json(out / "summary.json")
    replay = read_json(out / "deterministic_replay.json")
    reported_system_results = read_json(out / "system_results.json")
    reported_stage_metrics = read_json(out / "stage_metrics.json")
    reassembly_report = read_json(out / "reassembly_report.json")
    false_frame_report = read_json(out / "false_frame_report.json")
    requested_report = read_json(out / "requested_feature_guard_report.json")
    ambiguity_report = read_json(out / "ambiguity_guard_report.json")
    examples = read_json(out / "reassembly_examples.json")
    failure_examples = read_json(out / "failure_examples.json")
    rows = read_jsonl(out / "row_level_results.jsonl")

    if manifest.get("milestone") != MILESTONE:
        failures.append("manifest milestone mismatch")
    if set(manifest.get("systems", [])) != SYSTEMS:
        failures.append("manifest systems mismatch")
    if set(manifest.get("stages", [])) != STAGES:
        failures.append("manifest stages mismatch")
    if manifest.get("gradient_descent_used") is not False or manifest.get("optimizer_used") is not False or manifest.get("backprop_used") is not False:
        failures.append("gradient/optimizer/backprop flags are not false")
    if "decoded_feature == requested_feature" not in protocol.get("commit_guards", []):
        failures.append("protocol missing requested-feature commit guard")
    if "no conflicting requested-feature values" not in protocol.get("commit_guards", []):
        failures.append("protocol missing ambiguity guard")
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
    for required in ["candidate_count", "valid_candidate_count", "crc_pass_count", "requested_match_count", "selected_offset", "decision_reason"]:
        if any(required not in row for row in rows):
            failures.append(f"row-level missing required field {required}")
            break

    stage_metrics, system_results = summarize(rows)
    for system in SYSTEMS:
        for key, value in system_results[system]["overall"].items():
            if isinstance(value, (int, float)):
                compare_float(f"{system}.overall.{key}", value, reported_system_results[system]["overall"][key], failures)
    for stage in STAGES:
        for system in SYSTEMS:
            compare_float(
                f"{stage}.{system}.closed_loop_success",
                stage_metrics[stage][system]["closed_loop_success"],
                reported_stage_metrics[stage][system]["closed_loop_success"],
                failures,
            )

    locked = reported_system_results["bitslip_tolerant_reassembly_lock"]["overall"]
    strict = reported_system_results["strict_single_offset_full_guard"]["overall"]
    no_feature = reported_system_results["multi_offset_crc_no_feature_guard"]["overall"]
    no_ambiguity = reported_system_results["multi_offset_crc_requested_no_ambiguity_guard"]["overall"]
    loose = reported_system_results["loose_start_only_decoder"]["overall"]
    end_only = reported_system_results["end_marker_only_decoder"]["overall"]

    if aggregate.get("decision") == "e59_bitslip_tolerant_reassembly_locked":
        if locked["closed_loop_success"] < 0.995:
            failures.append("locked decision but locked closed-loop below 0.995")
        if locked["bitslip_recovery"] < 0.995:
            failures.append("locked decision but bitslip recovery below 0.995")
        if locked["false_frame_commit_rate"] > 0.001 or locked["wrong_feature_write_rate"] > 0.001:
            failures.append("locked decision but locked path has false-frame/wrong-feature commits")
        if locked["false_commit_rate"] > 0.001:
            failures.append("locked decision but locked path has false commits")
        if locked["ambiguity_reject_rate"] < 0.995:
            failures.append("locked decision but ambiguity reject below 0.995")
        if strict["bitslip_recovery"] > 0.05:
            failures.append("strict control did not expose bit-slip failure")
        if no_feature["wrong_feature_write_rate"] < 0.10:
            failures.append("no-feature control did not expose wrong-feature writes")
        if no_ambiguity["false_commit_rate"] < 0.05:
            failures.append("no-ambiguity control did not expose false commits")
        if loose["false_frame_commit_rate"] < 0.05:
            failures.append("loose control did not expose false-frame commits")
    if end_only["false_frame_commit_rate"] < 0.05:
        failures.append("end-marker-only control did not expose EOF insufficiency")

    if not math.isclose(reassembly_report["locked_bitslip_recovery"], locked["bitslip_recovery"], abs_tol=1e-9):
        failures.append("reassembly report mismatch")
    if not math.isclose(false_frame_report["locked_false_frame_commit_rate"], locked["false_frame_commit_rate"], abs_tol=1e-9):
        failures.append("false-frame report mismatch")
    if requested_report["locked_wrong_feature_write_rate"] > 0.001:
        failures.append("requested-feature report shows locked wrong-feature writes")
    if ambiguity_report["locked_ambiguity_reject_rate"] < 0.995:
        failures.append("ambiguity report shows weak locked ambiguity rejection")
    if len(examples) < 4:
        failures.append("not enough reassembly examples")
    if len(failure_examples) < 3:
        failures.append("not enough failure examples")

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
    parser.add_argument("--out", default="target/pilot_wave/e59_bitslip_tolerant_reassembly_lock")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e59_bitslip_tolerant_reassembly_lock")
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

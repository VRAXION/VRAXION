#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any


MILESTONE = "E43_PROPOSAL_FIELD_SHARED_THOUGHT_MATRIX_ADVERSARIAL_PROBE"
SYSTEMS = {
    "direct_flow_write_baseline",
    "explicit_single_proposal_packet",
    "shared_proposal_field",
    "per_pocket_proposal_planes",
    "shared_proposal_field_plus_agency",
    "per_pocket_planes_plus_agency",
    "oracle_commit_reference",
    "toxic_pocket_control",
    "proposal_flood_control",
    "stale_proposal_control",
}
DECISIONS = {
    "e43_shared_proposal_field_adversarial_confirmed",
    "e43_per_pocket_proposal_planes_required",
    "e43_proposal_field_partial_collision_bottleneck",
    "e43_direct_write_baseline_failed_as_expected",
    "e43_invalid_oracle_or_artifact_detected",
}
REQ_TARGET = [
    "backend_manifest.json",
    "task_generation_report.json",
    "proposal_field_frames.jsonl",
    "agency_decision_trace.jsonl",
    "collision_map.json",
    "toxic_pocket_report.json",
    "stale_proposal_report.json",
    "shared_vs_per_pocket_plane_summary.json",
    "system_results.json",
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
    "agency_decision_trace_sample.jsonl",
    "proposal_field_frames_sample.jsonl",
    "deterministic_replay_sample_report.json",
    "shared_vs_per_pocket_plane_summary_sample.json",
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


def validate_decision_row(row: dict[str, Any], failures: list[str], prefix: str) -> None:
    for key in [
        "system",
        "row_id",
        "split",
        "family",
        "expected_action",
        "action",
        "action_correct",
        "agency_decision_success",
        "trace_exact",
        "false_commit",
        "missed_commit",
        "stale_commit",
        "toxic_commit",
        "illegal_direct_flow_write",
        "proposal_field_cleared",
        "required_reason_bits",
        "reason_bits",
    ]:
        if key not in row:
            failures.append(f"{prefix}: missing {key}")


def validate_frame(row: dict[str, Any], failures: list[str], prefix: str) -> None:
    for key in [
        "system",
        "row_id",
        "family",
        "grid_side",
        "flow_before",
        "proposal_field_cells",
        "selected_proposal_id",
        "action",
        "expected_action",
        "direct_flow_write",
        "proposal_field_cleared",
    ]:
        if key not in row:
            failures.append(f"{prefix}: missing {key}")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 1.0


def metric_for_family(rows: list[dict[str, Any]], family: str, key: str) -> float:
    chunk = [row for row in rows if row["family"] == family]
    return mean([1.0 if row[key] else 0.0 for row in chunk])


def recompute(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for system in sorted({row["system"] for row in rows}):
        chunk = [row for row in rows if row["system"] == system]
        commit_expected = [row for row in chunk if row["expected_action"] == "COMMIT"]
        out[system] = {
            "row_count": float(len(chunk)),
            "action_accuracy": mean([1.0 if row["action_correct"] else 0.0 for row in chunk]),
            "agency_decision_success": mean([1.0 if row["agency_decision_success"] else 0.0 for row in chunk]),
            "trace_exact_rate": mean([1.0 if row["trace_exact"] else 0.0 for row in chunk]),
            "false_commit_rate": mean([1.0 if row["false_commit"] else 0.0 for row in chunk]),
            "missed_commit_rate": mean([1.0 if row["missed_commit"] else 0.0 for row in chunk]),
            "stale_commit_rate": mean([1.0 if row["stale_commit"] else 0.0 for row in chunk]),
            "toxic_commit_rate": mean([1.0 if row["toxic_commit"] else 0.0 for row in chunk]),
            "collision_resolution_accuracy": metric_for_family(chunk, "collision_same_target", "agency_decision_success"),
            "ground_conflict_rejection": metric_for_family(chunk, "ground_conflict", "action_correct"),
            "trace_mismatch_rejection": metric_for_family(chunk, "trace_mismatch", "action_correct"),
            "proposal_flood_resistance": metric_for_family(chunk, "proposal_flood", "agency_decision_success"),
            "no_valid_proposal_defer_accuracy": metric_for_family(chunk, "no_valid_proposal", "action_correct"),
            "write_spread": mean([float(row["write_spread"]) for row in chunk]),
            "illegal_direct_flow_write_rate": mean([1.0 if row["illegal_direct_flow_write"] else 0.0 for row in chunk]),
            "proposal_field_clear_rate": mean([1.0 if row["proposal_field_cleared"] else 0.0 for row in chunk]),
            "commit_target_value_accuracy": mean([1.0 if row["write_value_correct"] else 0.0 for row in commit_expected]),
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
    rows = read_jsonl(sample_dir / "agency_decision_trace_sample.jsonl")
    frames = read_jsonl(sample_dir / "proposal_field_frames_sample.jsonl")
    if schema.get("milestone") != MILESTONE or schema.get("proposal_field") is not True:
        failures.append("sample schema missing E43 marker")
    if schema.get("gradient_descent_used") is not False:
        failures.append("sample schema gradient flag is not false")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid sample decision {aggregate.get('decision')}")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("sample deterministic replay did not pass")
    if not rows or not frames:
        failures.append("sample row/frame artifact empty")
    for idx, row in enumerate(rows[:120]):
        validate_decision_row(row, failures, f"sample decision row {idx}")
    for idx, frame in enumerate(frames[:80]):
        validate_frame(frame, failures, f"sample frame {idx}")
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

    runner = Path(__file__).resolve().with_name("run_e43_proposal_field_shared_thought_matrix_adversarial_probe.py")
    failures.extend(static_policy_check(runner))
    manifest = read_json(out / "backend_manifest.json")
    aggregate = read_json(out / "aggregate_metrics.json")
    decision = read_json(out / "decision.json")
    summary = read_json(out / "summary.json")
    replay = read_json(out / "deterministic_replay.json")
    system_results = read_json(out / "system_results.json")
    shared_summary = read_json(out / "shared_vs_per_pocket_plane_summary.json")
    collision = read_json(out / "collision_map.json")
    toxic = read_json(out / "toxic_pocket_report.json")
    stale = read_json(out / "stale_proposal_report.json")
    rows = read_jsonl(out / "agency_decision_trace.jsonl")
    frames = read_jsonl(out / "proposal_field_frames.jsonl")
    progress = read_jsonl(out / "progress.jsonl")
    heartbeat = read_jsonl(out / "hardware_heartbeat.jsonl")

    if manifest.get("milestone") != MILESTONE or aggregate.get("milestone") != MILESTONE:
        failures.append("milestone mismatch")
    if manifest.get("proposal_field") is not True:
        failures.append("proposal field marker missing")
    if manifest.get("gradient_descent_used") is not False or manifest.get("optimizer_used") is not False or manifest.get("backprop_used") is not False:
        failures.append("gradient/optimizer/backprop flags are not false")
    if set(manifest.get("systems", [])) != SYSTEMS or set(system_results) != SYSTEMS:
        failures.append("system set mismatch")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid decision {aggregate.get('decision')}")
    if decision.get("decision") != aggregate.get("decision") or summary.get("decision") != aggregate.get("decision"):
        failures.append("decision mismatch across artifacts")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("deterministic replay did not pass")
    if not rows or not frames or not progress or not heartbeat:
        failures.append("empty row/frame/progress/heartbeat artifact")
    if {row.get("system") for row in rows} != SYSTEMS:
        failures.append("decision trace system mismatch")
    if {frame.get("system") for frame in frames} != SYSTEMS:
        failures.append("proposal frame system mismatch")
    for idx, row in enumerate(rows[:1200]):
        validate_decision_row(row, failures, f"decision row {idx}")
    for idx, frame in enumerate(frames[:500]):
        validate_frame(frame, failures, f"frame {idx}")
    if "shared_agency_success" not in shared_summary or "per_pocket_plane_gain" not in shared_summary:
        failures.append("shared/per-pocket summary missing fields")
    if "by_system" not in collision or "toxic_commit_rate_by_system" not in toxic or "stale_commit_rate_by_system" not in stale:
        failures.append("adversarial reports missing fields")

    recomputed = recompute(rows)
    for system, metrics in recomputed.items():
        reported = system_results[system]["overall"]
        for key, value in metrics.items():
            if key in reported:
                compare_float(f"{system}.{key}", value, reported[key], failures)

    shared = system_results["shared_proposal_field_plus_agency"]["overall"]
    planes = system_results["per_pocket_planes_plus_agency"]["overall"]
    direct = system_results["direct_flow_write_baseline"]["overall"]
    if aggregate.get("decision") == "e43_shared_proposal_field_adversarial_confirmed":
        if shared["action_accuracy"] < 0.95 or shared["agency_decision_success"] < 0.95:
            failures.append("shared positive decision but shared accuracy/success below threshold")
        if shared["false_commit_rate"] > 0.01 or shared["toxic_commit_rate"] > 0.01 or shared["stale_commit_rate"] > 0.01:
            failures.append("shared positive decision but false/toxic/stale commit too high")
        if shared["collision_resolution_accuracy"] < 0.95 or shared["no_valid_proposal_defer_accuracy"] < 0.95:
            failures.append("shared positive decision but collision/no-valid gate failed")
        if shared["illegal_direct_flow_write_rate"] != 0.0:
            failures.append("shared positive decision but illegal direct Flow write occurred")
        if direct["illegal_direct_flow_write_rate"] <= 0.0:
            failures.append("direct baseline did not perform illegal direct Flow writes")
        if planes["agency_decision_success"] > shared["agency_decision_success"] + 0.02:
            failures.append("per-pocket planes beat shared field despite shared positive decision")
    if aggregate.get("decision") == "e43_per_pocket_proposal_planes_required":
        if planes["agency_decision_success"] <= shared["agency_decision_success"] + 0.02:
            failures.append("per-pocket decision without per-pocket gain")

    sample_result = validate_sample(sample_dir)
    if not sample_result["passed"]:
        failures.extend([f"sample: {failure}" for failure in sample_result["failures"]])
    result = {"passed": not failures, "failure_count": len(failures), "failures": failures, "decision": aggregate.get("decision"), "run_id": aggregate.get("run_id"), "sample_result": sample_result}
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

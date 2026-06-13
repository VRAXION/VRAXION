#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any


MILESTONE = "E42_AGENCY_FIELD_COMMIT_AND_ACTION_MATRIX_PROBE"
SYSTEMS = {
    "oracle_agency_reference_only",
    "direct_pocket_action_baseline",
    "simple_priority_arbiter",
    "agency_field_without_ground",
    "agency_field_full_views_grow_shrink",
    "fixed_direct_decision_lanes_reference",
    "full_monolith_oracle_control",
    "random_action_control",
}
DECISIONS = {
    "e42_agency_field_positive",
    "e42_simple_arbiter_sufficient",
    "e42_ground_trace_not_needed",
    "e42_agency_field_growth_failed",
    "e42_monolith_control_required",
    "e42_invalid_artifact_detected",
}
REQ_TARGET = [
    "backend_manifest.json",
    "task_generation_report.json",
    "agency_field_report.json",
    "decision_lane_report.json",
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


def validate_row(row: dict[str, Any], failures: list[str], prefix: str) -> None:
    for key in [
        "system",
        "row_id",
        "split",
        "category",
        "views",
        "expected_action",
        "action",
        "action_correct",
        "wrong_commit",
        "missed_commit",
        "correct_defer",
        "correct_ask",
        "conflict_resolved",
        "unnecessary_call",
        "trace_exact",
        "required_reason_bits",
        "policy",
        "scan_sources",
    ]:
        if key not in row:
            failures.append(f"{prefix}: missing {key}")


def recompute(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for system in sorted({row["system"] for row in rows}):
        chunk = [row for row in rows if row["system"] == system]
        expected_defer = [row for row in chunk if row["expected_action"] == "DEFER"]
        expected_ask = [row for row in chunk if row["expected_action"] == "ASK"]
        expected_reject = [row for row in chunk if row["expected_action"] == "REJECT"]
        out[system] = {
            "action_accuracy": sum(1.0 if row["action_correct"] else 0.0 for row in chunk) / len(chunk),
            "wrong_commit_rate": sum(1.0 if row["wrong_commit"] else 0.0 for row in chunk) / len(chunk),
            "missed_commit_rate": sum(1.0 if row["missed_commit"] else 0.0 for row in chunk) / len(chunk),
            "correct_defer_rate": sum(1.0 if row["correct_defer"] else 0.0 for row in expected_defer) / len(expected_defer) if expected_defer else 1.0,
            "correct_ask_rate": sum(1.0 if row["correct_ask"] else 0.0 for row in expected_ask) / len(expected_ask) if expected_ask else 1.0,
            "conflict_resolution_rate": sum(1.0 if row["conflict_resolved"] else 0.0 for row in expected_reject) / len(expected_reject) if expected_reject else 1.0,
            "trace_exact_rate": sum(1.0 if row["trace_exact"] else 0.0 for row in chunk) / len(chunk),
            "unnecessary_call_rate": sum(1.0 if row["unnecessary_call"] else 0.0 for row in chunk) / len(chunk),
            "row_count": len(chunk),
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
    rows = read_jsonl(sample_dir / "row_level_sample.jsonl")
    history = read_jsonl(sample_dir / "mutation_history_sample.jsonl")
    if schema.get("milestone") != MILESTONE or schema.get("agency_field") is not True:
        failures.append("sample schema missing E42 marker")
    if schema.get("gradient_descent_used") is not False:
        failures.append("sample schema gradient flag is not false")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid sample decision {aggregate.get('decision')}")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("sample deterministic replay did not pass")
    if not rows or not history:
        failures.append("sample row/history artifact empty")
    for idx, row in enumerate(rows[:80]):
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

    runner = Path(__file__).resolve().with_name("run_e42_agency_field_commit_and_action_matrix_probe.py")
    failures.extend(static_policy_check(runner))
    manifest = read_json(out / "backend_manifest.json")
    aggregate = read_json(out / "aggregate_metrics.json")
    decision = read_json(out / "decision.json")
    summary = read_json(out / "summary.json")
    replay = read_json(out / "deterministic_replay.json")
    system_results = read_json(out / "system_results.json")
    mutation = read_json(out / "mutation_report.json")
    agency = read_json(out / "agency_field_report.json")
    lanes = read_json(out / "decision_lane_report.json")
    rows = read_jsonl(out / "row_level_results.jsonl")
    progress = read_jsonl(out / "progress.jsonl")
    heartbeat = read_jsonl(out / "hardware_heartbeat.jsonl")

    if manifest.get("milestone") != MILESTONE or aggregate.get("milestone") != MILESTONE:
        failures.append("milestone mismatch")
    if manifest.get("agency_field") is not True:
        failures.append("agency field marker missing")
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
    if not rows or not progress or not heartbeat:
        failures.append("empty row/progress/heartbeat artifact")
    if agency.get("primary_candidate") is None or lanes.get("no_semantic_lanes") is not True:
        failures.append("agency/lane report missing required fields")
    if {row.get("system") for row in rows} != SYSTEMS:
        failures.append("row-level system mismatch")
    for idx, row in enumerate(rows[:700]):
        validate_row(row, failures, f"row {idx}")

    recomputed = recompute(rows)
    for system, metrics in recomputed.items():
        reported = system_results[system]["overall"]
        for key, value in metrics.items():
            if key in reported:
                compare_float(f"{system}.{key}", value, reported[key], failures)

    for system in ["agency_field_without_ground", "agency_field_full_views_grow_shrink"]:
        stat = mutation.get(system, {})
        if int(stat.get("accepted_mutations", 0)) + int(stat.get("rejected_mutations", 0)) <= 0:
            failures.append(f"{system}: no mutation attempts")
        if int(stat.get("rollback_count", 0)) != int(stat.get("rejected_mutations", -1)):
            failures.append(f"{system}: rollback mismatch")
        if "parameter_diff" not in stat or "parameter_hash" not in stat:
            failures.append(f"{system}: missing parameter diff/hash")

    full = system_results["agency_field_full_views_grow_shrink"]["overall"]
    without_ground = system_results["agency_field_without_ground"]["overall"]
    simple = system_results["simple_priority_arbiter"]["overall"]
    direct = system_results["direct_pocket_action_baseline"]["overall"]
    if aggregate.get("decision") == "e42_agency_field_positive":
        if full["action_accuracy"] < 0.95 or full["trace_exact_rate"] < 0.90:
            failures.append("positive decision but full Agency accuracy/trace below threshold")
        if full["wrong_commit_rate"] > 0.03 or full["missed_commit_rate"] > 0.03:
            failures.append("positive decision but full Agency commit errors too high")
        if full["correct_defer_rate"] < 0.95 or full["correct_ask_rate"] < 0.95 or full["conflict_resolution_rate"] < 0.95:
            failures.append("positive decision but defer/ask/conflict gate failed")
        if simple["action_accuracy"] >= 0.85 or direct["action_accuracy"] >= 0.85:
            failures.append("positive decision but simple/direct control too strong")
        if without_ground["action_accuracy"] >= 0.95:
            failures.append("positive decision but no-ground ablation also passed")

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

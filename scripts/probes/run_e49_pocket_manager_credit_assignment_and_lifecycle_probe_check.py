#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any


MILESTONE = "E49_POCKET_MANAGER_CREDIT_ASSIGNMENT_AND_LIFECYCLE_PROBE"
SYSTEMS = {
    "no_manager_random_reuse",
    "final_answer_only_score",
    "immediate_score_only",
    "call_count_popularity_score",
    "full_event_credit_manager",
    "oracle_lifecycle_reference",
}
DECISIONS = {
    "e49_pocket_manager_credit_lifecycle_positive",
    "e49_final_answer_only_sufficient",
    "e49_immediate_score_sufficient",
    "e49_call_count_popularity_sufficient",
    "e49_counterfactual_credit_required",
    "e49_invalid_artifact_detected",
}

REQ_TARGET = [
    "backend_manifest.json",
    "pocket_event_schema.json",
    "pocket_evaluation_events.jsonl",
    "pocket_feature_report.json",
    "credit_assignment_report.json",
    "lifecycle_decision_report.json",
    "counterfactual_ablation_report.json",
    "delayed_credit_report.json",
    "manager_mutation_history.jsonl",
    "system_results.json",
    "lifecycle_decision_rows.jsonl",
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
    "pocket_feature_report_sample.json",
    "credit_assignment_report_sample.json",
    "pocket_evaluation_events_sample.jsonl",
    "lifecycle_decision_rows_sample.jsonl",
    "manager_mutation_history_sample.jsonl",
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


def recompute_from_lifecycle_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for system in sorted({row["system"] for row in rows}):
        chunk = [row for row in rows if row["system"] == system]
        core = [row for row in chunk if row["expected_lifecycle"] == "core"]
        dangerous = [row for row in chunk if row["expected_lifecycle"] in {"quarantine", "banned"}]
        selected = [row for row in chunk if row["predicted_lifecycle"] in {"core", "active", "specialist"}]
        beneficial = [row for row in chunk if row["expected_lifecycle"] in {"core", "active", "specialist"}]
        prune_false_positive = [
            row for row in beneficial if row["predicted_lifecycle"] in {"deprecated", "quarantine", "banned"}
        ]
        selected_cf = mean([row["counterfactual_gain"] for row in selected]) if selected else 0.0
        selected_cost = mean([row["cost"] for row in selected]) if selected else 0.0
        selected_harm = mean([row["wrong_commit_delta"] + row["downstream_harm"] for row in selected]) if selected else 0.0
        route_quality_delta = selected_cf - 0.20 * selected_cost - selected_harm
        delayed_poison = [row for row in chunk if row["pocket_id"] == "delayed_poison_cache"]
        hijacker = [row for row in chunk if row["pocket_id"] == "credit_hijacker_shadow"]
        out[system] = {
            "pocket_count": float(len(chunk)),
            "lifecycle_accuracy": mean([1.0 if row["correct"] else 0.0 for row in chunk]),
            "weighted_lifecycle_credit": mean([float(row["credit"]) for row in chunk]),
            "promote_correct_core": mean([1.0 if row["predicted_lifecycle"] == "core" else 0.0 for row in core]),
            "quarantine_dangerous_specialist": mean([1.0 if row["predicted_lifecycle"] in {"quarantine", "banned"} else 0.0 for row in dangerous]),
            "avoid_credit_hijack": 1.0 if hijacker and hijacker[0]["predicted_lifecycle"] in {"deprecated", "quarantine", "banned"} else 0.0,
            "delayed_harm_detection": 1.0 if delayed_poison and delayed_poison[0]["predicted_lifecycle"] in {"quarantine", "banned"} else 0.0,
            "cost_adjusted_utility": route_quality_delta,
            "route_quality_delta": route_quality_delta,
            "wrong_commit_delta": selected_harm,
            "prune_false_positive": len(prune_false_positive) / len(beneficial) if beneficial else 0.0,
            "selected_pocket_count": float(len(selected)),
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
    events = read_jsonl(sample_dir / "pocket_evaluation_events_sample.jsonl")
    rows = read_jsonl(sample_dir / "lifecycle_decision_rows_sample.jsonl")
    mutation = read_jsonl(sample_dir / "manager_mutation_history_sample.jsonl")
    if schema.get("milestone") != MILESTONE or schema.get("pocket_manager_lifecycle") is not True:
        failures.append("sample schema missing E49 marker")
    if schema.get("gradient_descent_used") is not False:
        failures.append("sample schema gradient flag is not false")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid sample decision {aggregate.get('decision')}")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("sample deterministic replay did not pass")
    if not events or not rows or not mutation:
        failures.append("sample events/rows/mutation empty")
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

    failures.extend(static_policy_check(Path("scripts/probes/run_e49_pocket_manager_credit_assignment_and_lifecycle_probe.py")))
    manifest = read_json(out / "backend_manifest.json")
    schema = read_json(out / "pocket_event_schema.json")
    aggregate = read_json(out / "aggregate_metrics.json")
    decision = read_json(out / "decision.json")
    summary = read_json(out / "summary.json")
    replay = read_json(out / "deterministic_replay.json")
    system_results = read_json(out / "system_results.json")
    credit_report = read_json(out / "credit_assignment_report.json")
    events = read_jsonl(out / "pocket_evaluation_events.jsonl")
    rows = read_jsonl(out / "lifecycle_decision_rows.jsonl")
    mutation = read_jsonl(out / "manager_mutation_history.jsonl")
    progress = read_jsonl(out / "progress.jsonl")
    heartbeat = read_jsonl(out / "hardware_heartbeat.jsonl")

    if manifest.get("milestone") != MILESTONE:
        failures.append("milestone mismatch")
    if manifest.get("gradient_descent_used") is not False or manifest.get("optimizer_used") is not False or manifest.get("backprop_used") is not False:
        failures.append("gradient/optimizer/backprop flags are not false")
    if schema.get("required_event") != "PocketEvaluationEvent":
        failures.append("missing PocketEvaluationEvent schema marker")
    if set(system_results) != SYSTEMS or set(credit_report) != SYSTEMS:
        failures.append("system/report set mismatch")
    if aggregate.get("decision") not in DECISIONS:
        failures.append(f"invalid decision {aggregate.get('decision')}")
    if decision.get("decision") != aggregate.get("decision") or summary.get("decision") != aggregate.get("decision"):
        failures.append("decision mismatch across artifacts")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("deterministic replay did not pass")
    if not events or not rows or not mutation or not progress or not heartbeat:
        failures.append("empty events/rows/mutation/progress/heartbeat artifact")
    required_event_fields = {
        "pocket_id",
        "pocket_version",
        "call_id",
        "cycle_id",
        "route_id",
        "input_footprint",
        "output_proposal_hash",
        "proposal_type",
        "agency_decision",
        "trace_ref",
        "cost",
        "immediate_outcome",
        "delayed_outcome",
        "counterfactual_without_pocket",
        "downstream_harm",
        "failure_mode",
    }
    if events and not required_event_fields.issubset(set(events[0])):
        failures.append("PocketEvaluationEvent missing required fields")

    recomputed = recompute_from_lifecycle_rows(rows)
    for system, metrics in recomputed.items():
        reported = system_results[system]["overall"]
        for key, value in metrics.items():
            if key in reported:
                compare_float(f"{system}.{key}", value, reported[key], failures)

    full_report = credit_report["full_event_credit_manager"]
    if full_report.get("accepted", 0) <= 0 or full_report.get("rejected", 0) <= 0:
        failures.append("full_event_credit_manager missing accepted/rejected mutation evidence")
    if full_report.get("rollback_count") != full_report.get("rejected"):
        failures.append("full_event_credit_manager rollback mismatch")
    if not full_report.get("parameter_diff_written") or not full_report.get("parameter_diff_hash"):
        failures.append("full_event_credit_manager missing parameter diff/hash")

    full = system_results["full_event_credit_manager"]["overall"]
    final_only = system_results["final_answer_only_score"]["overall"]
    if aggregate.get("decision") == "e49_pocket_manager_credit_lifecycle_positive":
        if full["lifecycle_accuracy"] < 0.90:
            failures.append("positive decision without lifecycle accuracy threshold")
        if full["avoid_credit_hijack"] != 1.0:
            failures.append("positive decision without avoiding credit hijack")
        if full["delayed_harm_detection"] != 1.0:
            failures.append("positive decision without delayed harm detection")
        if full["quarantine_dangerous_specialist"] < 0.95:
            failures.append("positive decision without dangerous quarantine")
        if final_only["avoid_credit_hijack"] >= 1.0:
            failures.append("positive decision but final-answer-only also avoids hijack")

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

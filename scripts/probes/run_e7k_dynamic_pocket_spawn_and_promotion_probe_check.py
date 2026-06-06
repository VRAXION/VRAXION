#!/usr/bin/env python3
"""Checker for E7K dynamic pocket spawn and promotion artifacts."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = "scripts/probes/run_e7k_dynamic_pocket_spawn_and_promotion_probe.py"
CHECKER = "scripts/probes/run_e7k_dynamic_pocket_spawn_and_promotion_probe_check.py"
SYSTEMS = (
    "fixed_library_no_spawn",
    "fixed_library_router_plus_repair",
    "oracle_spawn_scaffold",
    "random_spawn_control",
    "control_spawn_blank_pocket",
    "control_spawn_from_split",
    "control_spawn_from_composed_route",
    "control_spawn_plus_limited_repair",
    "dense_graph_danger_control",
)
MUTATION_SYSTEMS = (
    "control_spawn_blank_pocket",
    "control_spawn_from_split",
    "control_spawn_from_composed_route",
    "control_spawn_plus_limited_repair",
)
GRADIENT_SYSTEMS = ("dense_graph_danger_control",)
PHASES = (
    "phase_1_existing_library_sufficient",
    "phase_2_missing_reusable_transform",
    "phase_3_reuse_multiple_contexts",
    "phase_4_ood_counterfactual_generalization",
    "phase_5_damage_drift_repair",
)
SPLITS = ("train", "validation", "heldout", "ood", "counterfactual", "adversarial")
VALID_DECISIONS = {
    "e7k_dynamic_pocket_spawn_positive",
    "e7k_composed_route_pocket_spawn_positive",
    "e7k_split_spawn_positive",
    "e7k_spawn_needs_prior_scaffold",
    "e7k_spawn_artifact_or_task_too_easy",
    "e7k_spawn_overproduction_failure",
    "e7k_no_spawn_needed_existing_library_sufficient",
    "e7k_pocket_spawn_collapses_to_graph_soup",
    "e7k_leak_or_artifact_detected",
}
REQUIRED_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "spawn_mechanism_report.json",
    "spawn_promotion_report.json",
    "phase_spawn_winner_report.json",
    "system_results.json",
    "mutation_history.json",
    "training_history.json",
    "leakage_report.json",
    "deterministic_replay.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "progress.jsonl",
    "hardware_heartbeat.jsonl",
    "partial_aggregate_snapshot.json",
)
REQUIRED_MEAN_METRICS = (
    "eval_mean_spawn_value",
    "eval_mean_usefulness",
    "heldout_spawn_value",
    "ood_spawn_value",
    "counterfactual_spawn_value",
    "adversarial_spawn_value",
    "heldout_answer_accuracy",
    "heldout_route_accuracy",
    "heldout_spawn_precision",
    "heldout_spawn_recall",
    "heldout_unnecessary_spawn_rate",
    "heldout_promoted_pocket_count",
    "heldout_promoted_pocket_reuse_count",
    "heldout_spawned_pocket_average_K",
    "heldout_spawned_pocket_average_depth",
    "heldout_route_step_reduction",
    "heldout_route_cost_reduction",
    "heldout_freeze_survival",
    "heldout_local_repair_gain",
    "generalization_gap",
    "parameter_count",
)


def resolve_out(path: str | Path) -> Path:
    raw = Path(path)
    resolved = raw if raw.is_absolute() else (REPO_ROOT / raw).resolve()
    relative = resolved.relative_to(REPO_ROOT)
    if len(relative.parts) < 2 or relative.parts[0].lower() != "target" or relative.parts[1].lower() != "pilot_wave":
        raise ValueError("--out must stay under target/pilot_wave")
    return resolved


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def static_scan() -> list[str]:
    failures: list[str] = []
    runner = REPO_ROOT / RUNNER
    checker = REPO_ROOT / CHECKER
    for path in (runner, checker):
        if not path.exists():
            failures.append(f"MISSING_STATIC_FILE:{path.relative_to(REPO_ROOT).as_posix()}")
            return failures
    text = runner.read_text(encoding="utf-8")
    for token in (
        "E7K_DYNAMIC_POCKET_SPAWN_AND_PROMOTION_PROBE",
        "CALL(pocket_id, Flow[D]) -> Flow[D]",
        "control_spawn_from_composed_route",
        "control_spawn_plus_limited_repair",
        "dense_graph_danger_control",
        "hidden_missing_motif_id_used_as_model_input",
        "mutation_worker",
        "deterministic_replay",
        "start_hardware_monitor",
    ):
        if token not in text:
            failures.append(f"TOKEN_MISSING_IN_RUNNER:{token}")
    tree = ast.parse(text)
    mutation_functions = {"mutation_worker", "mutate_candidate", "candidate_learning_score", "bootstrap_candidates"}
    forbidden_calls = {"backward", "step", "zero_grad"}
    forbidden_optimizer_attrs = {"AdamW", "SGD", "RMSprop", "Optimizer"}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in mutation_functions:
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func = child.func
                    name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
                    if name in forbidden_calls:
                        failures.append(f"MUTATION_BACKPROP_CALL:{node.name}:{name}")
                if isinstance(child, ast.Attribute) and child.attr in forbidden_optimizer_attrs:
                    failures.append(f"MUTATION_OPTIMIZER_REFERENCE:{node.name}:{child.attr}")
    return failures


def check_artifacts(out: Path) -> list[str]:
    failures: list[str] = []
    for name in REQUIRED_ARTIFACTS:
        if not (out / name).exists():
            failures.append(f"MISSING_ARTIFACT:{name}")
    if failures:
        return failures
    manifest = load_json(out / "backend_manifest.json")
    task = load_json(out / "task_generation_report.json")
    mechanism = load_json(out / "spawn_mechanism_report.json")
    promotion = load_json(out / "spawn_promotion_report.json")
    phase_winner = load_json(out / "phase_spawn_winner_report.json")
    systems = load_json(out / "system_results.json")
    mutation = load_json(out / "mutation_history.json")
    training = load_json(out / "training_history.json")
    leakage = load_json(out / "leakage_report.json")
    replay = load_json(out / "deterministic_replay.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    decision = load_json(out / "decision.json")
    summary = load_json(out / "summary.json")
    seeds = [int(seed) for seed in manifest.get("settings", {}).get("seeds", [])]

    if manifest.get("milestone") != "E7K_DYNAMIC_POCKET_SPAWN_AND_PROMOTION_PROBE":
        failures.append("MANIFEST_MILESTONE_BAD")
    if tuple(manifest.get("systems", [])) != SYSTEMS:
        failures.append("MANIFEST_SYSTEMS_MISMATCH")
    if manifest.get("parallel_cpu_gpu_lanes") is not True:
        failures.append("PARALLEL_CPU_GPU_FLAG_BAD")
    if not seeds:
        failures.append("NO_SEEDS_RECORDED")
    if manifest.get("hardware_identity", {}).get("gpu", {}).get("cuda_available") is not True:
        failures.append("CUDA_HARDWARE_IDENTITY_MISSING")

    if task.get("public_inputs") != "microsegment_path_plus_phase_token_no_public_missing_motif_id":
        failures.append("TASK_PUBLIC_INPUTS_BAD")
    if task.get("hidden_missing_transform_used_for_eval_only") is not True:
        failures.append("TASK_HIDDEN_MOTIF_FLAG_BAD")
    if set(task.get("phases", [])) != set(PHASES):
        failures.append("TASK_PHASE_SET_BAD")
    for seed_counts in task.get("row_counts", {}).values():
        for phase in PHASES:
            for split in SPLITS:
                if seed_counts.get(phase, {}).get(split, 0) < 1:
                    failures.append(f"TASK_SPLIT_EMPTY:{phase}:{split}")
    if mechanism.get("external_interface") != "CALL(pocket_id, Flow[D]) -> Flow[D]":
        failures.append("FLOW_INTERFACE_BAD")
    if mechanism.get("dense_graph_allowed_for_mutation_systems") is not False:
        failures.append("MUTATION_DENSE_GRAPH_ALLOWED")
    if promotion.get("schema_version") != "e7k_spawn_promotion_report_v1":
        failures.append("SPAWN_PROMOTION_REPORT_BAD")
    if set(phase_winner.get("phases", {})) != set(PHASES):
        failures.append("PHASE_WINNER_REPORT_BAD")

    result_rows = systems.get("rows", [])
    expected_rows = len(seeds) * len(SYSTEMS)
    if len(result_rows) != expected_rows:
        failures.append(f"SYSTEM_RESULT_ROW_COUNT_MISMATCH:expected={expected_rows}:actual={len(result_rows)}")
    if {(row.get("system"), int(row.get("seed"))) for row in result_rows} != {(system, seed) for system in SYSTEMS for seed in seeds}:
        failures.append("SYSTEM_RESULT_SET_MISMATCH")
    aggregate_systems = aggregate.get("systems", {})
    if set(aggregate_systems) != set(SYSTEMS):
        failures.append("AGGREGATE_SYSTEM_SET_MISMATCH")
    for system in SYSTEMS:
        mean = aggregate_systems.get(system, {}).get("mean", {})
        for metric in REQUIRED_MEAN_METRICS:
            if metric not in mean:
                failures.append(f"AGGREGATE_MEAN_MISSING:{system}:{metric}")
        if aggregate_systems.get(system, {}).get("seed_count") != len(seeds):
            failures.append(f"AGGREGATE_SEED_COUNT_BAD:{system}")
    if set(aggregate.get("phase_winners", {})) != set(PHASES):
        failures.append("AGGREGATE_PHASE_WINNERS_BAD")

    for row in result_rows:
        for phase in PHASES:
            if phase not in row.get("phase_metrics", {}):
                failures.append(f"PHASE_METRICS_MISSING:{row.get('system')}:{row.get('seed')}:{phase}")
                continue
            for split in SPLITS:
                metrics = row["phase_metrics"][phase].get(split, {})
                for key in (
                    "answer_accuracy",
                    "route_accuracy",
                    "spawn_precision",
                    "spawn_recall",
                    "unnecessary_spawn_rate",
                    "promoted_pocket_count",
                    "promoted_pocket_reuse_count",
                    "spawned_pocket_average_K",
                    "spawned_pocket_average_depth",
                    "route_step_reduction",
                    "route_cost_reduction",
                    "freeze_survival",
                    "local_repair_gain",
                    "row_level_samples",
                ):
                    if key not in metrics:
                        failures.append(f"PHASE_SPLIT_METRIC_MISSING:{row.get('system')}:{phase}:{split}:{key}")
        for split in SPLITS:
            if not row.get("evals", {}).get(split, {}).get("row_level_samples"):
                failures.append(f"ROW_LEVEL_SAMPLES_MISSING:{row.get('system')}:{row.get('seed')}:{split}")

    mutation_rows = mutation.get("rows", [])
    if len(mutation_rows) != len(seeds) * len(MUTATION_SYSTEMS):
        failures.append("MUTATION_HISTORY_ROW_COUNT_MISMATCH")
    for row in mutation_rows:
        label = f"{row.get('system')}/seed{row.get('seed')}"
        if row.get("mutation_attempts", 0) < 1:
            failures.append(f"NO_MUTATION_ATTEMPTS:{label}")
        if row.get("accepted_mutations", 0) < 1:
            failures.append(f"NO_ACCEPTED_MUTATIONS:{label}")
        if row.get("rejected_mutations", 0) < 1:
            failures.append(f"NO_REJECTED_MUTATIONS:{label}")
        if row.get("rejected_mutations") != row.get("rollback_count"):
            failures.append(f"ROLLBACK_MISMATCH:{label}")
        if row.get("initial_candidate_hash") == row.get("final_candidate_hash"):
            failures.append(f"PARAMETER_DIFF_MISSING:{label}")
        if "promoted_pocket_count" not in row.get("final_candidate_summary", {}):
            failures.append(f"SPAWN_SUMMARY_MISSING:{label}")
        if row.get("failed_spawn_rollback_count", 0) < 1:
            failures.append(f"FAILED_SPAWN_ROLLBACK_MISSING:{label}")
        if not row.get("history"):
            failures.append(f"MUTATION_HISTORY_EMPTY:{label}")

    training_rows = training.get("rows", [])
    if len(training_rows) != len(seeds) * len(GRADIENT_SYSTEMS):
        failures.append("TRAINING_HISTORY_ROW_COUNT_MISMATCH")
    if not any(row.get("device") == "cuda" for row in training_rows):
        failures.append("NO_CUDA_GRADIENT_HISTORY")
    for row in training_rows:
        if not row.get("history"):
            failures.append(f"TRAINING_HISTORY_EMPTY:{row.get('system')}/seed{row.get('seed')}")

    if leakage.get("hidden_missing_motif_id_used_as_model_input") is not False:
        failures.append("LEAK_HIDDEN_MOTIF_INPUT_BAD")
    if leakage.get("dense_all_to_all_soft_routing_used_by_mutation_systems") is not False:
        failures.append("MUTATION_DENSE_GRAPH_FLAG_BAD")
    if leakage.get("random_spawn_control_passed") is not True and decision.get("decision") != "e7k_spawn_artifact_or_task_too_easy":
        failures.append("RANDOM_CONTROL_FAILED_WITHOUT_ARTIFACT_DECISION")
    if replay.get("internal_replay_passed") is not True:
        failures.append("DETERMINISTIC_REPLAY_FAILED")
    for name, row in replay.get("hash_comparisons", {}).items():
        if row.get("match") is not True:
            failures.append(f"REPLAY_HASH_MISMATCH:{name}")
    if decision.get("decision") not in VALID_DECISIONS:
        failures.append("DECISION_INVALID")
    if summary.get("decision") != decision.get("decision"):
        failures.append("SUMMARY_DECISION_MISMATCH")
    if summary.get("deterministic_replay_passed") is not True or decision.get("deterministic_replay_passed") is not True:
        failures.append("REPLAY_FLAG_BAD")

    events = {row.get("event") for row in jsonl_rows(out / "progress.jsonl")}
    for event in ("startup", "tasks_generated", "lanes_submitted", "gpu_lane_complete", "deterministic_replay_start", "deterministic_replay_complete", "final_artifacts_written"):
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")
    if not jsonl_rows(out / "hardware_heartbeat.jsonl"):
        failures.append("HARDWARE_HEARTBEAT_EMPTY")
    partial = load_json(out / "partial_aggregate_snapshot.json")
    if partial.get("completed_rows", 0) < len(seeds):
        failures.append("PARTIAL_AGGREGATE_TOO_WEAK")

    report = (out / "report.md").read_text(encoding="utf-8").lower()
    for forbidden in ("consciousness", "sentience", "agi", "model-scale", "natural-language reasoning"):
        if forbidden in report:
            failures.append(f"BROAD_CLAIM_IN_REPORT:{forbidden}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="target/pilot_wave/e7k_dynamic_pocket_spawn_and_promotion_probe")
    args = parser.parse_args()
    out = resolve_out(args.out)
    failures = static_scan()
    failures.extend(check_artifacts(out))
    report = {"schema_version": "e7k_checker_report_v1", "failure_count": len(failures), "failures": failures, "checked_root": out.relative_to(REPO_ROOT).as_posix()}
    (out / "checker_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    summary_path = out / "summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        summary["checker_failure_count"] = len(failures)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

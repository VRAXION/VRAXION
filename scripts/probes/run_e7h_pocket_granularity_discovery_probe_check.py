#!/usr/bin/env python3
"""Checker for E7H pocket granularity discovery artifacts."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = "scripts/probes/run_e7h_pocket_granularity_discovery_probe.py"
CHECKER = "scripts/probes/run_e7h_pocket_granularity_discovery_probe_check.py"
SYSTEMS = (
    "atomic_microsegment_router",
    "fixed_human_pockets",
    "fused_long_pipe",
    "mutation_discovered_pockets",
    "discovered_pockets_plus_router",
    "discovered_pockets_plus_limited_repair",
    "dense_graph_control",
    "random_boundary_control",
    "oracle_granularity_reference",
)
MUTATION_SYSTEMS = (
    "mutation_discovered_pockets",
    "discovered_pockets_plus_router",
    "discovered_pockets_plus_limited_repair",
)
GRADIENT_SYSTEMS = ("dense_graph_control",)
VALID_DECISIONS = {
    "e7h_mutation_discovers_reusable_pocket_granularity",
    "e7h_pocket_boundaries_need_prior_scaffold",
    "e7h_no_stable_pocket_granularity_detected",
    "e7h_long_pipe_needed_for_this_family",
    "e7h_pocket_discovery_collapses_to_graph_soup",
    "e7h_discovered_pockets_need_limited_repair",
    "e7h_leak_or_artifact_detected",
    "e7h_no_clear_granularity_winner",
}
REQUIRED_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "microsegment_inventory.json",
    "pocket_discovery_report.json",
    "freeze_reuse_repair_report.json",
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
    "heldout_answer_accuracy",
    "heldout_route_accuracy",
    "heldout_mean_route_steps",
    "heldout_compression_score",
    "heldout_irrelevant_branch_rate",
    "heldout_loop_rate",
    "heldout_discovered_pocket_count",
    "heldout_average_pocket_size",
    "heldout_reuse_count_per_pocket",
    "heldout_freeze_survival_score",
    "heldout_usefulness_score",
    "heldout_usefulness",
    "ood_usefulness",
    "counterfactual_usefulness",
    "adversarial_usefulness",
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
        "E7H_POCKET_GRANULARITY_DISCOVERY_PROBE",
        "mutation_discovered_pockets",
        "discovered_pockets_plus_router",
        "discovered_pockets_plus_limited_repair",
        "dense_graph_control",
        "merge_adjacent",
        "split_pocket",
        "freeze_unfreeze",
        "dense_all_to_all_soft_routing",
        "deterministic_replay",
        "start_hardware_monitor",
    ):
        if token not in text:
            failures.append(f"TOKEN_MISSING_IN_RUNNER:{token}")
    tree = ast.parse(text)
    mutation_functions = {"mutation_worker", "mutate_candidate", "candidate_score", "predict_from_pockets"}
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
    inventory = load_json(out / "microsegment_inventory.json")
    discovery = load_json(out / "pocket_discovery_report.json")
    freeze = load_json(out / "freeze_reuse_repair_report.json")
    systems = load_json(out / "system_results.json")
    mutation = load_json(out / "mutation_history.json")
    training = load_json(out / "training_history.json")
    leakage = load_json(out / "leakage_report.json")
    replay = load_json(out / "deterministic_replay.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    decision = load_json(out / "decision.json")
    summary = load_json(out / "summary.json")
    seeds = [int(seed) for seed in manifest.get("settings", {}).get("seeds", [])]

    if manifest.get("milestone") != "E7H_POCKET_GRANULARITY_DISCOVERY_PROBE":
        failures.append("MANIFEST_MILESTONE_BAD")
    if tuple(manifest.get("systems", [])) != SYSTEMS:
        failures.append("MANIFEST_SYSTEMS_MISMATCH")
    if manifest.get("parallel_cpu_gpu_lanes") is not True:
        failures.append("PARALLEL_CPU_GPU_FLAG_BAD")
    if manifest.get("hardware_identity", {}).get("gpu", {}).get("cuda_available") is not True:
        failures.append("CUDA_HARDWARE_IDENTITY_MISSING")
    if not seeds:
        failures.append("NO_SEEDS_RECORDED")

    if task.get("public_inputs") != "microsegment_path_only_no_public_pocket_ids":
        failures.append("TASK_PUBLIC_INPUTS_BAD")
    if task.get("hidden_pocket_ids_used_as_model_input") is not False:
        failures.append("HIDDEN_POCKET_INPUT_FLAG_BAD")
    if task.get("pocket_ids_are_hidden_for_eval_only") is not True:
        failures.append("HIDDEN_POCKET_EVAL_FLAG_BAD")
    for counts in task.get("row_counts", {}).values():
        for split in ("train", "validation", "heldout", "ood", "counterfactual", "adversarial"):
            if counts.get(split, 0) < 1:
                failures.append(f"TASK_SPLIT_EMPTY:{split}")
    if "dense_all_to_all_soft_routing" not in inventory.get("forbidden_mechanisms", []):
        failures.append("FORBIDDEN_MECHANISM_AUDIT_MISSING")
    if not inventory.get("natural_pockets_hidden_from_models"):
        failures.append("NATURAL_POCKET_AUDIT_MISSING")

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
    for row in result_rows:
        for split_name in ("train", "validation", "heldout", "ood", "counterfactual", "adversarial"):
            metrics = row.get("evals", {}).get(split_name)
            if not isinstance(metrics, dict):
                failures.append(f"SPLIT_METRICS_MISSING:{row.get('system')}:{row.get('seed')}:{split_name}")
                continue
            for key in ("answer_accuracy", "route_accuracy", "mean_route_steps", "discovered_pocket_count", "average_pocket_size", "reuse_count_per_pocket", "freeze_survival_score", "usefulness_score", "row_level_samples"):
                if key not in metrics:
                    failures.append(f"SPLIT_METRIC_MISSING:{row.get('system')}:{row.get('seed')}:{split_name}:{key}")
            if not metrics.get("row_level_samples"):
                failures.append(f"ROW_LEVEL_SAMPLES_MISSING:{row.get('system')}:{row.get('seed')}:{split_name}")

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
        if not row.get("accepted_by_operator") or not row.get("rejected_by_operator"):
            failures.append(f"OPERATOR_COUNTS_MISSING:{label}")
        if row.get("initial_candidate_hash") == row.get("final_candidate_hash"):
            failures.append(f"PARAMETER_DIFF_MISSING:{label}")
        summary_row = row.get("final_candidate_summary", {})
        if row.get("system") != "mutation_discovered_pockets" and summary_row.get("pocket_count", 0) < 1:
            failures.append(f"NO_DISCOVERED_POCKETS:{label}")
        if not row.get("history"):
            failures.append(f"MUTATION_HISTORY_EMPTY:{label}")
    if not discovery.get("mutation_system_summaries"):
        failures.append("DISCOVERY_SUMMARY_MISSING")

    training_rows = training.get("rows", [])
    if len(training_rows) != len(seeds) * len(GRADIENT_SYSTEMS):
        failures.append("TRAINING_HISTORY_ROW_COUNT_MISMATCH")
    if not any(row.get("device") == "cuda" for row in training_rows):
        failures.append("NO_CUDA_GRADIENT_HISTORY")
    for row in training_rows:
        if not row.get("history"):
            failures.append(f"TRAINING_HISTORY_EMPTY:{row.get('system')}/seed{row.get('seed')}")

    for key in ("discovered_freeze_survival", "repair_gain_over_router", "reuse_count_per_pocket", "local_repair_use_rate"):
        if key not in freeze:
            failures.append(f"FREEZE_REPAIR_METRIC_MISSING:{key}")
    if leakage.get("hidden_pocket_ids_used_as_model_input") is not False:
        failures.append("LEAK_HIDDEN_POCKET_INPUT_BAD")
    if leakage.get("dense_all_to_all_soft_routing_used_by_mutation_systems") is not False:
        failures.append("MUTATION_DENSE_GRAPH_FLAG_BAD")
    if leakage.get("random_control_passed") is not True and decision.get("decision") != "e7h_leak_or_artifact_detected":
        failures.append("LEAK_CONTROL_FAILED_WITHOUT_LEAK_DECISION")

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
    parser.add_argument("--out", default="target/pilot_wave/e7h_pocket_granularity_discovery_probe")
    args = parser.parse_args()
    out = resolve_out(args.out)
    failures = static_scan()
    failures.extend(check_artifacts(out))
    report = {
        "schema_version": "e7h_checker_report_v1",
        "failure_count": len(failures),
        "failures": failures,
        "checked_root": out.relative_to(REPO_ROOT).as_posix(),
    }
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

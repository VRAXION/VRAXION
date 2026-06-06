#!/usr/bin/env python3
"""Checker for E7I pocket-size optimum sweep artifacts."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = "scripts/probes/run_e7i_pocket_size_optimum_sweep.py"
CHECKER = "scripts/probes/run_e7i_pocket_size_optimum_sweep_check.py"
SYSTEMS = (
    "atomic_microsegment_router",
    "fixed_size_2_pockets",
    "fixed_size_3_pockets",
    "fixed_size_4_pockets",
    "mixed_size_2_3_pockets",
    "mixed_size_2_4_pockets",
    "mutation_discovered_variable_size_pockets",
    "fixed_human_pocket_scaffold",
    "fused_long_pipe",
    "dense_graph_control",
    "random_boundary_control",
    "oracle_family_granularity_reference",
)
FAMILIES = (
    "family_A_natural_size_2",
    "family_B_natural_size_3",
    "family_C_natural_size_4",
    "family_D_mixed_size_2_4",
    "family_E_no_stable_pocket_size",
    "family_F_decoy_pair_frequency",
    "family_G_reuse_sparse_family",
)
MUTATION_SYSTEMS = ("mutation_discovered_variable_size_pockets",)
GRADIENT_SYSTEMS = ("dense_graph_control",)
VALID_DECISIONS = {
    "e7i_stable_pocket_size_optimum_detected",
    "e7i_variable_pocket_granularity_preferred",
    "e7i_size2_was_generator_imprint",
    "e7i_pocket_size_needs_prior_scaffold",
    "e7i_atomic_microsegment_routing_preferred",
    "e7i_fused_pipe_overfit_detected",
    "e7i_pocket_granularity_collapses_to_graph_soup",
    "e7i_no_clear_size_frontier",
    "e7i_leak_or_artifact_detected",
}
REQUIRED_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "family_generation_report.json",
    "pocket_size_sweep_report.json",
    "family_winner_report.json",
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
    "eval_mean_usefulness",
    "heldout_usefulness",
    "ood_usefulness",
    "counterfactual_usefulness",
    "adversarial_usefulness",
    "heldout_answer_accuracy",
    "heldout_route_accuracy",
    "heldout_mean_route_steps",
    "heldout_average_pocket_size",
    "heldout_reuse_count_per_pocket",
    "heldout_irrelevant_branch_rate",
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
        "E7I_POCKET_SIZE_OPTIMUM_SWEEP",
        "fixed_size_2_pockets",
        "fixed_size_3_pockets",
        "fixed_size_4_pockets",
        "mutation_discovered_variable_size_pockets",
        "family_A_natural_size_2",
        "family_B_natural_size_3",
        "family_C_natural_size_4",
        "family_E_no_stable_pocket_size",
        "dense_graph_control",
        "deterministic_replay",
        "start_hardware_monitor",
    ):
        if token not in text:
            failures.append(f"TOKEN_MISSING_IN_RUNNER:{token}")
    tree = ast.parse(text)
    mutation_functions = {"mutation_worker", "mutate_candidate", "score_candidate", "candidate_pockets"}
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
    family_generation = load_json(out / "family_generation_report.json")
    sweep = load_json(out / "pocket_size_sweep_report.json")
    family_winner = load_json(out / "family_winner_report.json")
    systems = load_json(out / "system_results.json")
    mutation = load_json(out / "mutation_history.json")
    training = load_json(out / "training_history.json")
    leakage = load_json(out / "leakage_report.json")
    replay = load_json(out / "deterministic_replay.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    decision = load_json(out / "decision.json")
    summary = load_json(out / "summary.json")
    seeds = [int(seed) for seed in manifest.get("settings", {}).get("seeds", [])]

    if manifest.get("milestone") != "E7I_POCKET_SIZE_OPTIMUM_SWEEP":
        failures.append("MANIFEST_MILESTONE_BAD")
    if tuple(manifest.get("systems", [])) != SYSTEMS:
        failures.append("MANIFEST_SYSTEMS_MISMATCH")
    if manifest.get("parallel_cpu_gpu_lanes") is not True:
        failures.append("PARALLEL_CPU_GPU_FLAG_BAD")
    if not seeds:
        failures.append("NO_SEEDS_RECORDED")
    if manifest.get("hardware_identity", {}).get("gpu", {}).get("cuda_available") is not True:
        failures.append("CUDA_HARDWARE_IDENTITY_MISSING")

    if task.get("public_inputs") != "microsegment_path_plus_family_token_no_public_pocket_size_labels":
        failures.append("TASK_PUBLIC_INPUTS_BAD")
    if task.get("hidden_natural_sizes_used_for_eval_only") is not True:
        failures.append("HIDDEN_SIZE_EVAL_FLAG_BAD")
    if set(task.get("families", [])) != set(FAMILIES):
        failures.append("TASK_FAMILY_SET_BAD")
    for seed_counts in task.get("row_counts", {}).values():
        for family in FAMILIES:
            for split in ("train", "validation", "heldout", "ood", "counterfactual", "adversarial"):
                if seed_counts.get(family, {}).get(split, 0) < 1:
                    failures.append(f"TASK_SPLIT_EMPTY:{family}:{split}")
    if family_generation.get("size2_not_baked_into_all_families") is not True:
        failures.append("SIZE2_BIAS_AUDIT_MISSING")
    if set(family_generation.get("families", {})) != set(FAMILIES):
        failures.append("FAMILY_GENERATION_SET_BAD")

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
    if set(aggregate.get("family_winners", {})) != set(FAMILIES):
        failures.append("AGGREGATE_FAMILY_WINNERS_BAD")
    if set(family_winner.get("families", {})) != set(FAMILIES):
        failures.append("FAMILY_WINNER_REPORT_BAD")
    if not sweep.get("family_winners") or "variable_global_eval" not in sweep:
        failures.append("SWEEP_REPORT_MISSING_CORE")

    for row in result_rows:
        for family in FAMILIES:
            if family not in row.get("family_metrics", {}):
                failures.append(f"FAMILY_METRICS_MISSING:{row.get('system')}:{row.get('seed')}:{family}")
                continue
            for split in ("train", "validation", "heldout", "ood", "counterfactual", "adversarial"):
                metrics = row["family_metrics"][family].get(split, {})
                for key in ("answer_accuracy", "route_accuracy", "mean_route_steps", "average_pocket_size", "reuse_count_per_pocket", "usefulness_score", "row_level_samples"):
                    if key not in metrics:
                        failures.append(f"FAMILY_SPLIT_METRIC_MISSING:{row.get('system')}:{family}:{split}:{key}")
        for split in ("train", "validation", "heldout", "ood", "counterfactual", "adversarial"):
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
        if not row.get("final_candidate_summary", {}).get("family_size_policy"):
            failures.append(f"FAMILY_POLICY_MISSING:{label}")
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

    if leakage.get("hidden_natural_size_used_as_model_input") is not False:
        failures.append("LEAK_HIDDEN_SIZE_INPUT_BAD")
    if leakage.get("dense_all_to_all_soft_routing_used_by_mutation_systems") is not False:
        failures.append("MUTATION_DENSE_GRAPH_FLAG_BAD")
    if leakage.get("random_control_passed") is not True and decision.get("decision") != "e7i_leak_or_artifact_detected":
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
    parser.add_argument("--out", default="target/pilot_wave/e7i_pocket_size_optimum_sweep")
    args = parser.parse_args()
    out = resolve_out(args.out)
    failures = static_scan()
    failures.extend(check_artifacts(out))
    report = {"schema_version": "e7i_checker_report_v1", "failure_count": len(failures), "failures": failures, "checked_root": out.relative_to(REPO_ROOT).as_posix()}
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

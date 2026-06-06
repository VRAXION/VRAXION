#!/usr/bin/env python3
"""Checker for E7M anchor + working-copy crystallization probe."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = "scripts/probes/run_e7m_anchor_working_copy_crystallization_probe.py"
CHECKER = "scripts/probes/run_e7m_anchor_working_copy_crystallization_probe_check.py"
SYSTEMS = (
    "no_anchor_direct_mutation",
    "frozen_anchor_only",
    "frozen_anchor_plus_mutable_copy",
    "frozen_anchor_plus_mutable_copy_plus_pruning",
    "frozen_anchor_plus_mutable_copy_plus_prune_and_promote",
    "multi_copy_competition",
    "random_copy_control",
    "oracle_anchor_reference",
)
MUTATION_SYSTEMS = (
    "no_anchor_direct_mutation",
    "frozen_anchor_only",
    "frozen_anchor_plus_mutable_copy",
    "frozen_anchor_plus_mutable_copy_plus_pruning",
    "frozen_anchor_plus_mutable_copy_plus_prune_and_promote",
    "multi_copy_competition",
)
PHASES = (
    "phase_1_existing_library_sufficient",
    "phase_2_missing_reusable_transform",
    "phase_3_reuse_multiple_contexts",
    "phase_4_ood_counterfactual_generalization",
    "phase_5_damage_drift_repair",
)
SPLITS = ("train", "validation", "heldout", "ood", "counterfactual", "adversarial")
VALID_DECISIONS = {
    "e7m_anchor_working_copy_positive",
    "e7m_post_spawn_crystallization_positive",
    "e7m_safe_mutable_copy_promotion_positive",
    "e7m_multi_copy_competition_positive",
    "e7m_freeze_only_preferred_mutation_too_risky",
    "e7m_direct_mutation_sufficient_anchor_unneeded",
    "e7m_anchor_copy_overhead_too_high",
    "e7m_pruning_brittleness_detected",
    "e7m_promotion_guard_failure",
    "e7m_artifact_or_task_too_easy",
}
REQUIRED_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "lifecycle_contract_report.json",
    "anchor_working_copy_report.json",
    "crystallization_pruning_report.json",
    "promotion_guard_report.json",
    "system_results.json",
    "mutation_history.json",
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
    "eval_mean_raw_usefulness",
    "eval_mean_net_utility",
    "heldout_net_utility",
    "ood_net_utility",
    "counterfactual_net_utility",
    "adversarial_net_utility",
    "heldout_raw_usefulness",
    "heldout_answer_accuracy",
    "heldout_route_accuracy",
    "heldout_anchor_survival_rate",
    "heldout_mutable_copy_improvement_rate",
    "heldout_promotion_precision",
    "heldout_bad_promotion_rate",
    "heldout_discard_rate",
    "heldout_prune_compression_ratio",
    "heldout_post_prune_utility_delta",
    "heldout_maintenance_cost",
    "heldout_library_size",
    "heldout_copy_count",
    "heldout_junk_pocket_rate",
    "heldout_recovery_from_drift",
    "heldout_delayed_feedback_regret",
    "heldout_minimal_stable_pocket_size",
    "heldout_minimal_stable_pocket_cost",
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
        "E7M_ANCHOR_WORKING_COPY_CRYSTALLIZATION_PROBE",
        "Frozen anchors must never be overwritten directly",
        "frozen_anchor_plus_mutable_copy_plus_prune_and_promote",
        "multi_copy_competition",
        "net_utility_formula",
        "spawn -> validate -> crystallize/prune",
        "mutation_worker",
        "mutate_candidate",
        "promote_best_copy",
        "deterministic_replay",
        "start_hardware_monitor",
        "gpu_lane_reason",
    ):
        if token not in text:
            failures.append(f"TOKEN_MISSING_IN_RUNNER:{token}")
    tree = ast.parse(text)
    mutation_functions = {"mutation_worker", "mutate_candidate", "candidate_learning_score", "bootstrap_candidates", "promote_best_copy"}
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
    lifecycle = load_json(out / "lifecycle_contract_report.json")
    anchor = load_json(out / "anchor_working_copy_report.json")
    pruning = load_json(out / "crystallization_pruning_report.json")
    promotion = load_json(out / "promotion_guard_report.json")
    systems = load_json(out / "system_results.json")
    mutation = load_json(out / "mutation_history.json")
    leakage = load_json(out / "leakage_report.json")
    replay = load_json(out / "deterministic_replay.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    decision = load_json(out / "decision.json")
    summary = load_json(out / "summary.json")
    seeds = [int(seed) for seed in manifest.get("settings", {}).get("seeds", [])]

    if manifest.get("milestone") != "E7M_ANCHOR_WORKING_COPY_CRYSTALLIZATION_PROBE":
        failures.append("MANIFEST_MILESTONE_BAD")
    if tuple(manifest.get("systems", [])) != SYSTEMS:
        failures.append("MANIFEST_SYSTEMS_MISMATCH")
    if tuple(manifest.get("mutation_systems", [])) != MUTATION_SYSTEMS:
        failures.append("MANIFEST_MUTATION_SYSTEMS_MISMATCH")
    if manifest.get("source_milestone") != "E7L_SPAWN_REPAIR_COST_AND_NOISY_HEALTH_FALSIFICATION":
        failures.append("SOURCE_MILESTONE_BAD")
    if manifest.get("parallel_cpu_lanes") is not True:
        failures.append("PARALLEL_CPU_FLAG_BAD")
    if not seeds:
        failures.append("NO_SEEDS_RECORDED")
    if manifest.get("gpu_lane") is not False:
        failures.append("GPU_LANE_SHOULD_BE_FALSE_FOR_E7M")

    if task.get("schema_version") != "e7m_task_generation_report_v1":
        failures.append("TASK_REPORT_SCHEMA_BAD")
    if set(task.get("phases", [])) != set(PHASES):
        failures.append("TASK_PHASE_SET_BAD")
    for seed_counts in task.get("row_counts", {}).values():
        for phase in PHASES:
            for split in SPLITS:
                if seed_counts.get(phase, {}).get(split, 0) < 1:
                    failures.append(f"TASK_SPLIT_EMPTY:{phase}:{split}")

    if lifecycle.get("frozen_anchor_overwrite_allowed") is not False:
        failures.append("ANCHOR_OVERWRITE_ALLOWED")
    if "raw_usefulness - spawn_cost - repair_cost - prune_cost" not in lifecycle.get("net_utility_formula", ""):
        failures.append("NET_UTILITY_FORMULA_MISSING")
    if anchor.get("frozen_anchor_overwrite_detected") is not False:
        failures.append("FROZEN_ANCHOR_OVERWRITE_DETECTED")
    if pruning.get("pruning_happened_on_working_copies_or_pre_anchor_candidates_only") is not True:
        failures.append("PRUNING_TARGET_CONTRACT_BAD")
    if len(promotion.get("promotion_guard", [])) < 5:
        failures.append("PROMOTION_GUARD_TOO_SHORT")

    result_rows = systems.get("rows", [])
    expected_rows = len(seeds) * len(SYSTEMS)
    if len(result_rows) != expected_rows:
        failures.append(f"SYSTEM_RESULT_ROW_COUNT_MISMATCH:expected={expected_rows}:actual={len(result_rows)}")
    if {(row.get("system"), int(row.get("seed"))) for row in result_rows} != {(system, seed) for system in SYSTEMS for seed in seeds}:
        failures.append("SYSTEM_RESULT_SET_MISMATCH")
    aggregate_systems = aggregate.get("systems", {})
    if set(aggregate_systems) != set(SYSTEMS):
        failures.append("AGGREGATE_SYSTEM_SET_MISMATCH")
    if aggregate.get("best_non_oracle_system") not in SYSTEMS or aggregate.get("best_non_oracle_system") == "oracle_anchor_reference":
        failures.append("BEST_NON_ORACLE_BAD")
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
        if "evals" not in row or "phase_metrics" not in row:
            failures.append(f"ROW_LEVEL_EVAL_MISSING:{row.get('system')}:{row.get('seed')}")
            continue
        for split in SPLITS:
            if not row["evals"].get(split, {}).get("row_level_samples"):
                failures.append(f"ROW_SAMPLES_MISSING:{row.get('system')}:{row.get('seed')}:{split}")
        for phase in PHASES:
            if phase not in row["phase_metrics"]:
                failures.append(f"PHASE_METRICS_MISSING:{row.get('system')}:{row.get('seed')}:{phase}")
                continue
            for split in SPLITS:
                metrics = row["phase_metrics"][phase].get(split, {})
                for key in (
                    "raw_usefulness",
                    "net_utility",
                    "answer_accuracy",
                    "route_accuracy",
                    "anchor_survival_rate",
                    "mutable_copy_improvement_rate",
                    "promotion_precision",
                    "bad_promotion_rate",
                    "discard_rate",
                    "prune_compression_ratio",
                    "post_prune_utility_delta",
                    "maintenance_cost",
                    "library_size",
                    "copy_count",
                    "junk_pocket_rate",
                    "recovery_from_drift",
                    "delayed_feedback_regret",
                    "minimal_stable_pocket_size",
                    "minimal_stable_pocket_cost",
                ):
                    if key not in metrics:
                        failures.append(f"PHASE_METRIC_MISSING:{row.get('system')}:{row.get('seed')}:{phase}:{split}:{key}")
        if row["system"] != "random_copy_control" and row["system"] != "oracle_anchor_reference":
            if not row.get("anchor_version_history") and row["system"] != "no_anchor_direct_mutation":
                failures.append(f"ANCHOR_VERSION_HISTORY_MISSING:{row.get('system')}:{row.get('seed')}")
            if not row.get("lifecycle_state_hash"):
                failures.append(f"LIFECYCLE_HASH_MISSING:{row.get('system')}:{row.get('seed')}")
        if row["system"] in {"frozen_anchor_plus_mutable_copy", "frozen_anchor_plus_mutable_copy_plus_pruning", "frozen_anchor_plus_mutable_copy_plus_prune_and_promote", "multi_copy_competition"}:
            if not row.get("working_copy_lineage"):
                failures.append(f"WORKING_COPY_LINEAGE_MISSING:{row.get('system')}:{row.get('seed')}")

    mutation_rows = mutation.get("rows", [])
    expected_mutation_rows = len(seeds) * len(MUTATION_SYSTEMS)
    if len(mutation_rows) != expected_mutation_rows:
        failures.append(f"MUTATION_HISTORY_ROW_COUNT_MISMATCH:expected={expected_mutation_rows}:actual={len(mutation_rows)}")
    if {(row.get("system"), int(row.get("seed"))) for row in mutation_rows} != {(system, seed) for system in MUTATION_SYSTEMS for seed in seeds}:
        failures.append("MUTATION_HISTORY_SET_MISMATCH")
    for row in mutation_rows:
        attempts = int(row.get("mutation_attempts", 0))
        accepted = int(row.get("accepted_mutations", 0))
        rejected = int(row.get("rejected_mutations", 0))
        rollback = int(row.get("rollback_count", -1))
        if attempts <= 0 or accepted <= 0 or rejected <= 0:
            failures.append(f"MUTATION_COUNTS_BAD:{row.get('system')}:{row.get('seed')}")
        if rollback != rejected:
            failures.append(f"ROLLBACK_MISMATCH:{row.get('system')}:{row.get('seed')}")
        if row.get("initial_candidate_hash") == row.get("final_candidate_hash"):
            failures.append(f"PARAMETER_DIFF_MISSING:{row.get('system')}:{row.get('seed')}")
        if not row.get("parameter_diff_hash"):
            failures.append(f"PARAMETER_DIFF_HASH_MISSING:{row.get('system')}:{row.get('seed')}")
        if not row.get("history"):
            failures.append(f"MUTATION_HISTORY_EMPTY:{row.get('system')}:{row.get('seed')}")

    if leakage.get("hidden_missing_motif_id_used_as_model_input") is not False:
        failures.append("LEAKAGE_HIDDEN_MOTIF_PUBLIC")
    if leakage.get("mutation_system_uses_optimizer_or_backprop") is not False:
        failures.append("MUTATION_OPTIMIZER_LEAKAGE")
    if leakage.get("random_copy_control_passed") is not True:
        failures.append("RANDOM_CONTROL_FAILED")
    if replay.get("internal_replay_passed") is not True:
        failures.append("DETERMINISTIC_REPLAY_FAILED")
    for name, item in replay.get("hash_comparisons", {}).items():
        if item.get("match") is not True:
            failures.append(f"REPLAY_HASH_MISMATCH:{name}")

    if decision.get("decision") not in VALID_DECISIONS:
        failures.append("DECISION_LABEL_BAD")
    if summary.get("decision") != decision.get("decision"):
        failures.append("SUMMARY_DECISION_MISMATCH")
    if summary.get("best_non_oracle_system") != aggregate.get("best_non_oracle_system"):
        failures.append("SUMMARY_BEST_MISMATCH")
    if summary.get("deterministic_replay_passed") is not True or decision.get("deterministic_replay_passed") is not True:
        failures.append("REPLAY_FLAG_NOT_PROPAGATED")

    progress = jsonl_rows(out / "progress.jsonl")
    events = {row.get("event") for row in progress}
    for event in ("startup", "tasks_generated", "lanes_submitted", "mutation_generation", "mutation_job_complete", "deterministic_replay_start", "deterministic_replay_complete", "final_artifacts_written"):
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")
    heartbeat_rows = jsonl_rows(out / "hardware_heartbeat.jsonl")
    if not heartbeat_rows:
        failures.append("HARDWARE_HEARTBEAT_EMPTY")
    if "completed_rows" not in load_json(out / "partial_aggregate_snapshot.json"):
        failures.append("PARTIAL_AGGREGATE_BAD")

    report_text = (out / "report.md").read_text(encoding="utf-8").lower()
    for forbidden in ("consciousness", "sentience", "model-scale claim"):
        if forbidden in report_text:
            failures.append(f"BROAD_CLAIM_FORBIDDEN:{forbidden}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="target/pilot_wave/e7m_anchor_working_copy_crystallization_probe")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()
    out = resolve_out(args.out)
    failures = static_scan()
    if out.exists():
        failures.extend(check_artifacts(out))
    else:
        failures.append(f"OUT_ROOT_MISSING:{out.relative_to(REPO_ROOT).as_posix()}")
    payload = {
        "schema_version": "e7m_checker_result_v1",
        "out": out.relative_to(REPO_ROOT).as_posix(),
        "failure_count": len(failures),
        "failures": failures,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.write_summary and out.exists():
        summary_path = out / "summary.json"
        if summary_path.exists():
            summary = load_json(summary_path)
            summary["checker_failure_count"] = len(failures)
            summary["checker_failures"] = failures[:20]
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Checker for E7G addressable chapter-skip router artifacts."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = "scripts/probes/run_e7g_addressable_chapter_skip_router_probe.py"
CHECKER = "scripts/probes/run_e7g_addressable_chapter_skip_router_probe_check.py"
SYSTEMS = (
    "sequential_pipe_scan",
    "fixed_short_pipe_router",
    "fused_long_pipe_path_model",
    "addressable_chapter_router_mutation",
    "addressable_router_sparse_call_prior",
    "dense_graph_soft_router_gradient",
    "random_segment_walk_control",
    "oracle_chapter_skip_reference",
)
GRADIENT_SYSTEMS = ("dense_graph_soft_router_gradient",)
MUTATION_SYSTEMS = ("addressable_chapter_router_mutation", "addressable_router_sparse_call_prior")
VALID_DECISIONS = {
    "e7g_addressable_chapter_skip_confirmed",
    "e7g_sequential_scan_sufficient",
    "e7g_fused_path_model_sufficient",
    "e7g_dense_graph_soft_router_preferred",
    "e7g_overbranching_or_loop_failure",
    "e7g_leak_or_artifact_detected",
    "e7g_no_clear_chapter_skip_winner",
}
REQUIRED_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "chapter_library_report.json",
    "addressable_skip_report.json",
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
    "heldout_skip_efficiency",
    "heldout_irrelevant_branch_rate",
    "heldout_loop_rate",
    "heldout_mean_steps",
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
        "E7G_ADDRESSABLE_CHAPTER_SKIP_ROUTER_PROBE",
        "addressable_chapter_router_mutation",
        "addressable_router_sparse_call_prior",
        "dense_graph_soft_router_gradient",
        "sequential_pipe_scan",
        "chapter_library_report",
        "addressable_skip_report",
        "deterministic_replay",
        "start_hardware_monitor",
        "return_to_router_after_each_call",
    ):
        if token not in text:
            failures.append(f"TOKEN_MISSING_IN_RUNNER:{token}")
    tree = ast.parse(text)
    mutation_functions = {"mutation_worker", "mutate_candidate", "score_candidate", "predict_candidate"}
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
    chapters = load_json(out / "chapter_library_report.json")
    skip = load_json(out / "addressable_skip_report.json")
    systems = load_json(out / "system_results.json")
    mutation = load_json(out / "mutation_history.json")
    training = load_json(out / "training_history.json")
    leakage = load_json(out / "leakage_report.json")
    replay = load_json(out / "deterministic_replay.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    decision = load_json(out / "decision.json")
    summary = load_json(out / "summary.json")
    seeds = [int(seed) for seed in manifest.get("settings", {}).get("seeds", [])]

    if manifest.get("milestone") != "E7G_ADDRESSABLE_CHAPTER_SKIP_ROUTER_PROBE":
        failures.append("MANIFEST_MILESTONE_BAD")
    if tuple(manifest.get("systems", [])) != SYSTEMS:
        failures.append("MANIFEST_SYSTEMS_MISMATCH")
    if manifest.get("parallel_cpu_gpu_lanes") is not True:
        failures.append("PARALLEL_CPU_GPU_FLAG_BAD")
    if not seeds:
        failures.append("NO_SEEDS_RECORDED")
    if manifest.get("hardware_identity", {}).get("gpu", {}).get("cuda_available") is not True:
        failures.append("CUDA_HARDWARE_IDENTITY_MISSING")

    if task.get("chapter_ids_are_explicit_task_addresses") is not True:
        failures.append("TASK_ADDRESS_FLAG_BAD")
    if task.get("hidden_correct_path_used_as_private_input") is not False:
        failures.append("HIDDEN_PATH_FLAG_BAD")
    for counts in task.get("row_counts", {}).values():
        for split in ("train", "validation", "heldout", "ood", "counterfactual", "adversarial"):
            if counts.get(split, 0) < 1:
                failures.append(f"TASK_SPLIT_EMPTY:{split}")
    if chapters.get("contract", {}).get("return_to_router_after_each_call") is not True:
        failures.append("RETURN_TO_ROUTER_CONTRACT_BAD")
    if chapters.get("not_a_free_dense_graph") is not True:
        failures.append("DENSE_GRAPH_BOUNDARY_FLAG_BAD")

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
            for key in ("answer_accuracy", "route_accuracy", "skip_efficiency", "irrelevant_branch_rate", "loop_rate", "usefulness_score", "row_level_samples"):
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
        if not row.get("parameter_diff_hash"):
            failures.append(f"PARAMETER_DIFF_HASH_MISSING:{label}")
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

    if skip.get("interpretation_boundary") != "addressable_chapter_skip_controlled_proxy":
        failures.append("SKIP_BOUNDARY_BAD")
    for key in ("addressable_minus_sequential_heldout", "addressable_minus_fused_ood", "sequential_mean_steps", "addressable_mean_steps"):
        if key not in skip:
            failures.append(f"SKIP_METRIC_MISSING:{key}")
    if leakage.get("random_control_passed") is not True and decision.get("decision") != "e7g_leak_or_artifact_detected":
        failures.append("LEAK_CONTROL_FAILED_WITHOUT_LEAK_DECISION")
    if leakage.get("hidden_correct_path_used_as_private_input") is not False:
        failures.append("LEAK_HIDDEN_PATH_FLAG_BAD")

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
    heartbeat_rows = jsonl_rows(out / "hardware_heartbeat.jsonl")
    if not heartbeat_rows:
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
    parser.add_argument("--out", default="target/pilot_wave/e7g_addressable_chapter_skip_router_probe")
    args = parser.parse_args()
    out = resolve_out(args.out)
    failures = static_scan()
    failures.extend(check_artifacts(out))
    report = {
        "schema_version": "e7g_checker_report_v1",
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

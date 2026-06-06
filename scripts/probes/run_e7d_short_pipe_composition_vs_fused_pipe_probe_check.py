#!/usr/bin/env python3
"""Checker for E7D short-pipe composition versus fused-pipe artifacts."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = "scripts/probes/run_e7d_short_pipe_composition_vs_fused_pipe_probe.py"
CHECKER = "scripts/probes/run_e7d_short_pipe_composition_vs_fused_pipe_probe_check.py"
SYSTEMS = (
    "monolithic_matrix_core_gradient",
    "monolithic_mutation_model",
    "fused_long_pipe_gradient_router",
    "fused_long_pipe_mutation_router",
    "short_pipe_no_router_between",
    "short_pipe_router_composition",
    "router_plus_limited_pocket_repair",
    "random_router_control",
    "oracle_short_pipe_reference",
)
GRADIENT_SYSTEMS = ("monolithic_matrix_core_gradient", "fused_long_pipe_gradient_router")
MUTATION_SYSTEMS = (
    "monolithic_mutation_model",
    "fused_long_pipe_mutation_router",
    "short_pipe_no_router_between",
    "short_pipe_router_composition",
    "router_plus_limited_pocket_repair",
)
VALID_DECISIONS = {
    "e7d_short_pipe_router_flow_preferred",
    "e7d_fused_long_pipe_required",
    "e7d_hybrid_common_fused_plus_short_preferred",
    "e7d_monolithic_sufficient_or_task_too_easy",
    "e7d_router_overhead_failure",
    "e7d_leak_or_artifact_detected",
    "e7d_no_clear_topology_winner",
}
REQUIRED_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "pipe_topology_report.json",
    "flow_state_report.json",
    "system_results.json",
    "mutation_history.json",
    "training_history.json",
    "composition_report.json",
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
    "answer_accuracy",
    "route_accuracy",
    "composition_accuracy",
    "shortcut_rate",
    "usefulness_score",
    "step_penalized_usefulness",
    "latency_steps",
    "generalization_gap",
    "heldout_usefulness",
    "ood_usefulness",
    "counterfactual_usefulness",
    "adversarial_usefulness",
    "ood_route_accuracy",
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
        "E7D_SHORT_PIPE_COMPOSITION_VS_FUSED_PIPE_PROBE",
        "fused_long_pipe_mutation_router",
        "short_pipe_router_composition",
        "short_pipe_no_router_between",
        "router_plus_limited_pocket_repair",
        "ood_unseen_pair_compositions_used",
        "branch_after_first",
        "deterministic_replay",
        "start_hardware_monitor",
    ):
        if token not in text:
            failures.append(f"TOKEN_MISSING_IN_RUNNER:{token}")
    tree = ast.parse(text)
    mutation_functions = {"mutation_worker", "mutate_candidate", "predict_candidate", "score_candidate", "evaluate_candidate_full"}
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
    topology = load_json(out / "pipe_topology_report.json")
    flow = load_json(out / "flow_state_report.json")
    systems = load_json(out / "system_results.json")
    mutation = load_json(out / "mutation_history.json")
    training = load_json(out / "training_history.json")
    composition = load_json(out / "composition_report.json")
    leakage = load_json(out / "leakage_report.json")
    replay = load_json(out / "deterministic_replay.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    decision = load_json(out / "decision.json")
    summary = load_json(out / "summary.json")

    seeds = [int(seed) for seed in manifest.get("settings", {}).get("seeds", [])]
    if manifest.get("milestone") != "E7D_SHORT_PIPE_COMPOSITION_VS_FUSED_PIPE_PROBE":
        failures.append("MANIFEST_MILESTONE_BAD")
    if tuple(manifest.get("systems", [])) != SYSTEMS:
        failures.append("MANIFEST_SYSTEMS_MISMATCH")
    if manifest.get("parallel_cpu_gpu_lanes") is not True:
        failures.append("PARALLEL_CPU_GPU_FLAG_BAD")
    if manifest.get("hardware_identity", {}).get("gpu", {}).get("cuda_available") is not True:
        failures.append("CUDA_HARDWARE_IDENTITY_MISSING")
    if len(seeds) < 1:
        failures.append("NO_SEEDS_RECORDED")

    if task.get("target_pair_route_not_in_raw_features") is not True:
        failures.append("TARGET_PAIR_ROUTE_RAW_LEAK_FLAG_BAD")
    if task.get("ood_unseen_pair_compositions_used") is not True:
        failures.append("OOD_UNSEEN_PAIR_FLAG_BAD")
    if not task.get("ood_heldout_pairs") or not task.get("seen_train_pairs"):
        failures.append("PAIR_SPLIT_MISSING")
    for counts in task.get("row_counts", {}).values():
        for split in ("train", "validation", "heldout", "ood", "counterfactual", "adversarial"):
            if counts.get(split, 0) < 1:
                failures.append(f"TASK_SPLIT_EMPTY:{split}")

    if topology.get("short_pipe_library", {}).get("router_return_between_pipes") is not True:
        failures.append("SHORT_PIPE_ROUTER_RETURN_FLAG_BAD")
    if topology.get("fused_pipe_library", {}).get("unseen_pair_generalization_required") is not True:
        failures.append("FUSED_OOD_FLAG_BAD")
    if "branch_flag_after_first_pipe" not in flow.get("flow_state_fields", []):
        failures.append("FLOW_STATE_BRANCH_FIELD_MISSING")

    result_rows = systems.get("rows", [])
    if len(result_rows) != len(seeds) * len(SYSTEMS):
        failures.append(f"SYSTEM_RESULT_ROW_COUNT_MISMATCH:expected={len(seeds) * len(SYSTEMS)}:actual={len(result_rows)}")
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
        for split in ("train", "validation", "heldout", "ood", "counterfactual", "adversarial"):
            metrics = row.get("evals", {}).get(split)
            if not isinstance(metrics, dict):
                failures.append(f"SPLIT_METRICS_MISSING:{row.get('system')}:{row.get('seed')}:{split}")
                continue
            for key in ("answer_accuracy", "route_accuracy", "composition_accuracy", "shortcut_rate", "usefulness_score", "step_penalized_usefulness", "latency_steps", "row_level_samples"):
                if key not in metrics:
                    failures.append(f"SPLIT_METRIC_MISSING:{row.get('system')}:{row.get('seed')}:{split}:{key}")
            if not metrics.get("row_level_samples"):
                failures.append(f"ROW_LEVEL_SAMPLES_MISSING:{row.get('system')}:{row.get('seed')}:{split}")

    mutation_rows = mutation.get("rows", [])
    if len(mutation_rows) != len(seeds) * len(MUTATION_SYSTEMS):
        failures.append("MUTATION_HISTORY_ROW_COUNT_MISMATCH")
    for row in mutation_rows:
        label = f"{row.get('system')}/seed{row.get('seed')}"
        if row.get("mutation_attempts", 0) < 1:
            failures.append(f"NO_MUTATION_ATTEMPTS:{label}")
        if row.get("rejected_mutations", 0) < 1:
            failures.append(f"NO_REJECTED_MUTATIONS:{label}")
        if row.get("rejected_mutations") != row.get("rollback_count"):
            failures.append(f"ROLLBACK_MISMATCH:{label}")
        if not row.get("accepted_by_operator") or not row.get("rejected_by_operator"):
            failures.append(f"OPERATOR_COUNTS_MISSING:{label}")
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

    if composition.get("interpretation_boundary") != "short_pipe_router_flow_vs_fused_pipe_controlled_proxy":
        failures.append("COMPOSITION_BOUNDARY_BAD")
    if "router_between_gain_over_no_router" not in composition:
        failures.append("ROUTER_GAIN_METRIC_MISSING")
    if leakage.get("random_control_passed") is not True and decision.get("decision") != "e7d_leak_or_artifact_detected":
        failures.append("LEAK_CONTROL_FAILED_WITHOUT_LEAK_DECISION")
    if leakage.get("hidden_correct_pair_route_used_as_input") is not False:
        failures.append("HIDDEN_PAIR_ROUTE_FLAG_BAD")

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
    for event in (
        "startup",
        "tasks_generated",
        "lanes_submitted",
        "gradient_job_start",
        "gradient_epoch",
        "mutation_generation",
        "mutation_job_complete",
        "gpu_lane_complete",
        "deterministic_replay_complete",
        "final_artifacts_written",
    ):
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")
    hardware_rows = jsonl_rows(out / "hardware_heartbeat.jsonl")
    if not hardware_rows:
        failures.append("HARDWARE_HEARTBEAT_EMPTY")
    elif not any(row.get("hardware", {}).get("gpu", {}).get("cuda_available") is True for row in hardware_rows):
        failures.append("HARDWARE_HEARTBEAT_GPU_MISSING")
    for dirname in ("partial_status", "mutation_history_snapshots"):
        folder = out / dirname
        if not folder.exists() or not any(folder.rglob("*.json")):
            failures.append(f"PARTIAL_OUTPUT_DIR_EMPTY:{dirname}")
    partial = load_json(out / "partial_aggregate_snapshot.json")
    if partial.get("completed_rows", 0) < len(seeds):
        failures.append("PARTIAL_AGGREGATE_BAD")

    report_text = (out / "report.md").read_text(encoding="utf-8").lower()
    for forbidden in ("consciousness", "sentience", "agi", "model-scale", "natural-language reasoning"):
        if forbidden in report_text:
            failures.append(f"FORBIDDEN_REPORT_CLAIM_TOKEN:{forbidden}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="target/pilot_wave/e7d_short_pipe_composition_vs_fused_pipe_probe")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = resolve_out(args.out)
    failures = static_scan() + check_artifacts(out)
    report = {
        "schema_version": "e7d_checker_report_v1",
        "out": out.relative_to(REPO_ROOT).as_posix(),
        "failure_count": len(failures),
        "failures": failures,
    }
    path = out / "checker_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

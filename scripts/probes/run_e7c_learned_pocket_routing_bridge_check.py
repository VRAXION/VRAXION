#!/usr/bin/env python3
"""Checker for E7C learned pocket routing bridge artifacts."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = "scripts/probes/run_e7c_learned_pocket_routing_bridge.py"
CHECKER = "scripts/probes/run_e7c_learned_pocket_routing_bridge_check.py"
SYSTEMS = (
    "monolithic_backprop_model",
    "monolithic_mutation_model",
    "learned_pockets_gradient_router",
    "learned_pockets_mutation_router",
    "learned_binary_pockets_mutation_router",
    "router_plus_limited_pocket_repair",
    "random_router_control",
    "oracle_learned_pocket_router_reference",
    "oracle_symbolic_reference",
)
GRADIENT_SYSTEMS = ("monolithic_backprop_model", "learned_pockets_gradient_router")
MUTATION_SYSTEMS = (
    "monolithic_mutation_model",
    "learned_pockets_mutation_router",
    "learned_binary_pockets_mutation_router",
    "router_plus_limited_pocket_repair",
)
VALID_DECISIONS = {
    "e7c_learned_pocket_mutation_router_viable",
    "e7c_binary_learned_pocket_router_viable",
    "e7c_gradient_router_only_viable",
    "e7c_symbolic_only_scaffold_detected",
    "e7c_learned_pocket_quality_bottleneck",
    "e7c_leak_or_artifact_detected",
}
REQUIRED_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "learned_pocket_training_report.json",
    "pocket_library_report.json",
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
        "E7C_LEARNED_POCKET_ROUTING_BRIDGE",
        "PocketNet",
        "learned_pocket_training_report",
        "learned_pockets_mutation_router",
        "learned_binary_pockets_mutation_router",
        "oracle_learned_pocket_router_reference",
        "start_hardware_monitor",
        "deterministic_replay",
    ):
        if token not in text:
            failures.append(f"TOKEN_MISSING_IN_RUNNER:{token}")
    tree = ast.parse(text)
    mutation_wrappers = {"run_mutation_job", "control_results", "aggregate_results"}
    forbidden_calls = {"backward", "step", "zero_grad"}
    forbidden_optimizer_attrs = {"AdamW", "SGD", "RMSprop", "Optimizer"}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in mutation_wrappers:
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func = child.func
                    name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
                    if name in forbidden_calls:
                        failures.append(f"MUTATION_WRAPPER_BACKPROP_CALL:{node.name}:{name}")
                if isinstance(child, ast.Attribute) and child.attr in forbidden_optimizer_attrs:
                    failures.append(f"MUTATION_WRAPPER_OPTIMIZER_REFERENCE:{node.name}:{child.attr}")
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
    pocket_train = load_json(out / "learned_pocket_training_report.json")
    pocket = load_json(out / "pocket_library_report.json")
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
    if manifest.get("milestone") != "E7C_LEARNED_POCKET_ROUTING_BRIDGE":
        failures.append("MANIFEST_MILESTONE_BAD")
    if tuple(manifest.get("systems", [])) != SYSTEMS:
        failures.append("MANIFEST_SYSTEMS_MISMATCH")
    if manifest.get("parallel_cpu_gpu_lanes") is not True:
        failures.append("PARALLEL_CPU_GPU_FLAG_BAD")
    if manifest.get("hardware_identity", {}).get("gpu", {}).get("cuda_available") is not True:
        failures.append("CUDA_HARDWARE_IDENTITY_MISSING")

    if task.get("learned_pocket_outputs_used_for_router_inputs") is not True:
        failures.append("LEARNED_POCKET_INPUT_FLAG_BAD")
    if task.get("symbolic_task_ground_truth_retained") is not True:
        failures.append("SYMBOLIC_GROUND_TRUTH_FLAG_BAD")
    for counts in task.get("row_counts", {}).values():
        for split in ("train", "validation", "heldout", "ood", "counterfactual", "adversarial"):
            if counts.get(split, 0) < 1:
                failures.append(f"TASK_SPLIT_EMPTY:{split}")

    pocket_rows = pocket_train.get("rows", [])
    if len(pocket_rows) != len(seeds):
        failures.append("POCKET_TRAINING_ROW_COUNT_MISMATCH")
    for row in pocket_rows:
        label = f"seed{row.get('seed')}"
        if row.get("device") != "cuda":
            failures.append(f"POCKET_NOT_CUDA:{label}")
        if not row.get("history"):
            failures.append(f"POCKET_HISTORY_EMPTY:{label}")
        if not row.get("state_hash"):
            failures.append(f"POCKET_STATE_HASH_MISSING:{label}")
        for split, metrics in row.get("split_metrics", {}).items():
            for key in ("candidate_answer_accuracy", "branch_accuracy", "oracle_learned_route_answer_ceiling"):
                if key not in metrics:
                    failures.append(f"POCKET_SPLIT_METRIC_MISSING:{label}:{split}:{key}")
    if pocket.get("pocket_source") != "separately_learned_frozen_pocket_models":
        failures.append("POCKET_SOURCE_BAD")
    if pocket.get("learned_pockets_are_frozen_for_router_training") is not True:
        failures.append("POCKET_FROZEN_FLAG_BAD")

    result_rows = systems.get("rows", [])
    if len(result_rows) != len(seeds) * len(SYSTEMS):
        failures.append("SYSTEM_RESULT_ROW_COUNT_MISMATCH")
    if {(row.get("system"), int(row.get("seed"))) for row in result_rows} != {(system, seed) for system in SYSTEMS for seed in seeds}:
        failures.append("SYSTEM_RESULT_SET_MISMATCH")
    aggregate_systems = aggregate.get("systems", {})
    if set(aggregate_systems) != set(SYSTEMS):
        failures.append("AGGREGATE_SYSTEM_SET_MISMATCH")
    for system in SYSTEMS:
        mean = aggregate_systems.get(system, {}).get("mean", {})
        for metric in ("answer_accuracy", "route_accuracy", "composition_accuracy", "shortcut_rate", "usefulness_score", "adversarial_usefulness", "parameter_count"):
            if metric not in mean:
                failures.append(f"AGGREGATE_MEAN_MISSING:{system}:{metric}")
    for row in result_rows:
        for split in ("train", "validation", "heldout", "ood", "counterfactual", "adversarial"):
            metrics = row.get("evals", {}).get(split)
            if not isinstance(metrics, dict):
                failures.append(f"SPLIT_METRICS_MISSING:{row.get('system')}:{row.get('seed')}:{split}")
                continue
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
    training_rows = training.get("rows", [])
    if len(training_rows) != len(seeds) * len(GRADIENT_SYSTEMS):
        failures.append("TRAINING_HISTORY_ROW_COUNT_MISMATCH")
    if not any(row.get("device") == "cuda" for row in training_rows):
        failures.append("NO_CUDA_GRADIENT_HISTORY")

    if composition.get("interpretation_boundary") != "routing_over_separately_learned_frozen_pocket_outputs":
        failures.append("COMPOSITION_BOUNDARY_BAD")
    if leakage.get("random_control_passed") is not True and decision.get("decision") != "e7c_leak_or_artifact_detected":
        failures.append("LEAK_CONTROL_FAILED_WITHOUT_LEAK_DECISION")
    if leakage.get("hidden_correct_route_index_used_as_input") is not False:
        failures.append("HIDDEN_ROUTE_INDEX_FLAG_BAD")

    if replay.get("internal_replay_passed") is not True:
        failures.append("DETERMINISTIC_REPLAY_FAILED")
    for name, row in replay.get("hash_comparisons", {}).items():
        if row.get("match") is not True:
            failures.append(f"REPLAY_HASH_MISMATCH:{name}")
    if decision.get("decision") not in VALID_DECISIONS:
        failures.append("DECISION_INVALID")
    if summary.get("decision") != decision.get("decision"):
        failures.append("SUMMARY_DECISION_MISMATCH")
    if summary.get("deterministic_replay_passed") is not True:
        failures.append("SUMMARY_REPLAY_FLAG_BAD")

    events = {row.get("event") for row in jsonl_rows(out / "progress.jsonl")}
    for event in (
        "startup",
        "pocket_training_start",
        "pocket_epoch",
        "pocket_training_complete",
        "learned_tasks_ready",
        "lanes_submitted",
        "gradient_job_start",
        "mutation_generation",
        "mutation_job_complete",
        "gpu_lane_complete",
        "deterministic_replay_complete",
        "final_artifacts_written",
    ):
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")
    hardware_rows = jsonl_rows(out / "hardware_heartbeat.jsonl")
    if not hardware_rows or not any(row.get("hardware", {}).get("gpu", {}).get("cuda_available") is True for row in hardware_rows):
        failures.append("HARDWARE_HEARTBEAT_GPU_MISSING")
    for dirname in ("partial_status", "mutation_history_snapshots"):
        folder = out / dirname
        if not folder.exists() or not any(folder.rglob("*.json")):
            failures.append(f"PARTIAL_OUTPUT_DIR_EMPTY:{dirname}")

    report_text = (out / "report.md").read_text(encoding="utf-8").lower()
    for forbidden in ("consciousness", "sentience", "agi", "model-scale", "natural-language reasoning"):
        if forbidden in report_text:
            failures.append(f"FORBIDDEN_REPORT_CLAIM_TOKEN:{forbidden}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="target/pilot_wave/e7c_learned_pocket_routing_bridge")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = resolve_out(args.out)
    failures = static_scan() + check_artifacts(out)
    report = {"schema_version": "e7c_checker_report_v1", "out": out.relative_to(REPO_ROOT).as_posix(), "failure_count": len(failures), "failures": failures}
    path = out / "checker_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

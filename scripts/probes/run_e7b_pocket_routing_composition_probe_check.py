#!/usr/bin/env python3
"""Checker for E7B pocket routing composition probe artifacts."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = "scripts/probes/run_e7b_pocket_routing_composition_probe.py"
CHECKER = "scripts/probes/run_e7b_pocket_routing_composition_probe_check.py"
SYSTEMS = (
    "monolithic_backprop_model",
    "monolithic_mutation_model",
    "frozen_pockets_gradient_router",
    "frozen_pockets_mutation_router",
    "frozen_pockets_binary_router",
    "router_plus_limited_pocket_repair",
    "random_router_control",
    "oracle_pocket_router_reference",
)
GRADIENT_SYSTEMS = ("monolithic_backprop_model", "frozen_pockets_gradient_router")
MUTATION_SYSTEMS = (
    "monolithic_mutation_model",
    "frozen_pockets_mutation_router",
    "frozen_pockets_binary_router",
    "router_plus_limited_pocket_repair",
)
VALID_DECISIONS = {
    "e7b_mutation_router_composition_viable",
    "e7b_gradient_only_composition_viable",
    "e7b_monolithic_mutation_sufficient_or_task_too_easy",
    "e7b_pocket_router_no_advantage_detected",
    "e7b_router_leak_or_artifact_detected",
}
REQUIRED_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
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
REQUIRED_MEAN_METRICS = (
    "answer_accuracy",
    "route_accuracy",
    "composition_accuracy",
    "shortcut_rate",
    "usefulness_score",
    "generalization_gap",
    "heldout_usefulness",
    "ood_usefulness",
    "counterfactual_usefulness",
    "adversarial_usefulness",
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
        "E7B_POCKET_ROUTING_COMPOSITION_PROBE",
        "gpu_lane_worker",
        "mutation_worker",
        "hardware_heartbeat",
        "frozen_pockets_mutation_router",
        "router_plus_limited_pocket_repair",
        "oracle_pocket_router_reference",
        "deterministic_replay",
    ):
        if token not in text:
            failures.append(f"TOKEN_MISSING_IN_RUNNER:{token}")
    tree = ast.parse(text)
    mutation_functions = {"mutation_worker", "mutate_candidate", "score_candidate", "predict_candidate", "router_logits"}
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


def check_system_rows(failures: list[str], manifest: dict[str, Any], results: dict[str, Any], aggregate: dict[str, Any]) -> None:
    seeds = [int(seed) for seed in manifest.get("settings", {}).get("seeds", [])]
    rows = results.get("rows", [])
    if len(rows) != len(seeds) * len(SYSTEMS):
        failures.append(f"SYSTEM_RESULT_ROW_COUNT_MISMATCH:expected={len(seeds) * len(SYSTEMS)}:actual={len(rows)}")
    seen = {(row.get("system"), int(row.get("seed"))) for row in rows}
    expected = {(system, seed) for system in SYSTEMS for seed in seeds}
    if seen != expected:
        failures.append("SYSTEM_RESULT_SET_MISMATCH")
    systems = aggregate.get("systems", {})
    if set(systems) != set(SYSTEMS):
        failures.append("AGGREGATE_SYSTEM_SET_MISMATCH")
    for system in SYSTEMS:
        mean = systems.get(system, {}).get("mean", {})
        for metric in REQUIRED_MEAN_METRICS:
            if metric not in mean:
                failures.append(f"AGGREGATE_MEAN_METRIC_MISSING:{system}:{metric}")
        if systems.get(system, {}).get("seed_count") != len(seeds):
            failures.append(f"AGGREGATE_SEED_COUNT_BAD:{system}")
    for row in rows:
        for split in ("train", "validation", "heldout", "ood", "counterfactual", "adversarial"):
            metrics = row.get("evals", {}).get(split)
            if not isinstance(metrics, dict):
                failures.append(f"SPLIT_METRICS_MISSING:{row.get('system')}:{row.get('seed')}:{split}")
                continue
            for key in ("answer_accuracy", "route_accuracy", "composition_accuracy", "shortcut_rate", "usefulness_score", "row_level_samples"):
                if key not in metrics:
                    failures.append(f"SPLIT_METRIC_MISSING:{row.get('system')}:{row.get('seed')}:{split}:{key}")


def check_mutation_and_training(failures: list[str], manifest: dict[str, Any], mutation: dict[str, Any], training: dict[str, Any]) -> None:
    seeds = [int(seed) for seed in manifest.get("settings", {}).get("seeds", [])]
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


def check_artifacts(out: Path) -> list[str]:
    failures: list[str] = []
    for name in REQUIRED_ARTIFACTS:
        if not (out / name).exists():
            failures.append(f"MISSING_ARTIFACT:{name}")
    if failures:
        return failures
    manifest = load_json(out / "backend_manifest.json")
    task = load_json(out / "task_generation_report.json")
    pocket = load_json(out / "pocket_library_report.json")
    results = load_json(out / "system_results.json")
    mutation = load_json(out / "mutation_history.json")
    training = load_json(out / "training_history.json")
    composition = load_json(out / "composition_report.json")
    leakage = load_json(out / "leakage_report.json")
    replay = load_json(out / "deterministic_replay.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    decision = load_json(out / "decision.json")
    summary = load_json(out / "summary.json")

    if manifest.get("milestone") != "E7B_POCKET_ROUTING_COMPOSITION_PROBE":
        failures.append("MANIFEST_MILESTONE_BAD")
    if tuple(manifest.get("systems", [])) != SYSTEMS:
        failures.append("MANIFEST_SYSTEMS_MISMATCH")
    if manifest.get("parallel_cpu_gpu_lanes") is not True:
        failures.append("PARALLEL_CPU_GPU_FLAG_BAD")
    hardware = manifest.get("hardware_identity", {})
    if hardware.get("cpu_count", 0) < 2:
        failures.append("HARDWARE_CPU_COUNT_BAD")
    if hardware.get("gpu", {}).get("cuda_available") is not True:
        failures.append("HARDWARE_CUDA_NOT_RECORDED")

    for split_counts in task.get("row_counts", {}).values():
        for split in ("train", "validation", "heldout", "ood", "counterfactual", "adversarial"):
            if split_counts.get(split, 0) < 1:
                failures.append(f"TASK_SPLIT_EMPTY:{split}")
    if pocket.get("pocket_source") != "frozen_deterministic_symbolic_pockets_for_routing_isolation":
        failures.append("POCKET_SOURCE_BAD")
    if pocket.get("learned_pocket_claim_deferred") is not True:
        failures.append("POCKET_BOUNDARY_FLAG_BAD")

    check_system_rows(failures, manifest, results, aggregate)
    check_mutation_and_training(failures, manifest, mutation, training)

    if composition.get("interpretation_boundary") != "routing_over_frozen_symbolic_pockets_only":
        failures.append("COMPOSITION_BOUNDARY_BAD")
    if leakage.get("random_control_passed") is not True and decision.get("decision") != "e7b_router_leak_or_artifact_detected":
        failures.append("LEAKAGE_CONTROL_FAILED_WITHOUT_LEAK_DECISION")
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
    if partial.get("completed_rows", 0) < len(manifest.get("settings", {}).get("seeds", [])):
        failures.append("PARTIAL_AGGREGATE_BAD")

    report_text = (out / "report.md").read_text(encoding="utf-8").lower()
    for forbidden in ("consciousness", "sentience", "agi", "model-scale", "natural-language reasoning"):
        if forbidden in report_text:
            failures.append(f"FORBIDDEN_REPORT_CLAIM_TOKEN:{forbidden}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="target/pilot_wave/e7b_pocket_routing_composition_probe")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = resolve_out(args.out)
    failures = static_scan() + check_artifacts(out)
    report = {
        "schema_version": "e7b_checker_report_v1",
        "out": out.relative_to(REPO_ROOT).as_posix(),
        "failure_count": len(failures),
        "failures": failures,
    }
    write_path = out / "checker_report.json"
    write_path.parent.mkdir(parents=True, exist_ok=True)
    write_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

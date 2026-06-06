#!/usr/bin/env python3
"""Checker for E7N real numeric pocket core bridge probe."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = "scripts/probes/run_e7n_real_numeric_pocket_core_bridge_probe.py"
CHECKER = "scripts/probes/run_e7n_real_numeric_pocket_core_bridge_probe_check.py"
SYSTEMS = (
    "symbolic_proxy_pocket_reference",
    "float_numeric_pocket_backprop",
    "quantized_numeric_pocket_int8",
    "quantized_numeric_pocket_int4",
    "quantized_numeric_pocket_ternary",
    "quantized_numeric_pocket_binary",
    "quantized_pocket_plus_mutation_repair",
    "quantized_pocket_plus_prune_crystallize",
    "quantized_pocket_plus_repair_plus_prune",
    "random_pocket_control",
)
MUTATION_SYSTEMS = (
    "quantized_pocket_plus_mutation_repair",
    "quantized_pocket_plus_prune_crystallize",
    "quantized_pocket_plus_repair_plus_prune",
)
SPLITS = ("train", "validation", "heldout", "ood", "counterfactual", "adversarial")
REQUIRED_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "numeric_pocket_training_report.json",
    "quantization_report.json",
    "mutation_repair_report.json",
    "pruning_crystallization_report.json",
    "pocket_registry_report.json",
    "router_call_report.json",
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
    "eval_mean_answer_accuracy",
    "eval_mean_router_call_usefulness",
    "heldout_accuracy",
    "ood_accuracy",
    "counterfactual_accuracy",
    "adversarial_accuracy",
    "parameter_count",
    "active_parameter_count",
    "bit_budget",
)
VALID_DECISIONS = {
    "e7n_real_numeric_pocket_viable",
    "e7n_quantized_numeric_pocket_viable",
    "e7n_mutation_repair_numeric_pocket_positive",
    "e7n_numeric_pocket_crystallization_positive",
    "e7n_binary_numeric_pocket_viable",
    "e7n_float_only_numeric_pocket",
    "e7n_numeric_pocket_bridge_failed",
}


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
    if failures:
        return failures
    text = runner.read_text(encoding="utf-8")
    for token in (
        "E7N_REAL_NUMERIC_POCKET_CORE_BRIDGE_PROBE",
        "CALL(pocket_id, Flow[D]) -> Flow[D]",
        "NumericPocketCore",
        "input adapter D->K",
        "mutation_repair",
        "prune_state",
        "quantize_state",
        "deterministic_replay",
        "hardware_heartbeat",
        "symbolic_segment_proxy_primary",
    ):
        if token not in text:
            failures.append(f"TOKEN_MISSING_IN_RUNNER:{token}")
    tree = ast.parse(text)
    mutation_functions = {"mutate_binary_state", "mutation_repair", "prune_state", "derived_worker", "quick_score"}
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
    training = load_json(out / "numeric_pocket_training_report.json")
    quant = load_json(out / "quantization_report.json")
    repair = load_json(out / "mutation_repair_report.json")
    pruning = load_json(out / "pruning_crystallization_report.json")
    registry = load_json(out / "pocket_registry_report.json")
    router = load_json(out / "router_call_report.json")
    systems = load_json(out / "system_results.json")
    mutation = load_json(out / "mutation_history.json")
    leakage = load_json(out / "leakage_report.json")
    replay = load_json(out / "deterministic_replay.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    decision = load_json(out / "decision.json")
    summary = load_json(out / "summary.json")
    seeds = [int(seed) for seed in manifest.get("settings", {}).get("seeds", [])]

    if manifest.get("milestone") != "E7N_REAL_NUMERIC_POCKET_CORE_BRIDGE_PROBE":
        failures.append("MANIFEST_MILESTONE_BAD")
    if tuple(manifest.get("systems", [])) != SYSTEMS:
        failures.append("MANIFEST_SYSTEMS_MISMATCH")
    if tuple(manifest.get("mutation_systems", [])) != MUTATION_SYSTEMS:
        failures.append("MANIFEST_MUTATION_SYSTEMS_MISMATCH")
    if manifest.get("real_numeric_matrices_used") is not True:
        failures.append("REAL_NUMERIC_MATRICES_FLAG_BAD")
    if manifest.get("symbolic_segment_proxy_primary") is not False:
        failures.append("SYMBOLIC_PROXY_PRIMARY_BAD")
    if manifest.get("mutation_backprop_allowed") is not False:
        failures.append("MUTATION_BACKPROP_FLAG_BAD")
    if not seeds:
        failures.append("NO_SEEDS_RECORDED")

    if task.get("schema_version") != "e7n_task_generation_report_v1":
        failures.append("TASK_SCHEMA_BAD")
    if task.get("flow_dim") != 40 or task.get("class_count") != 8:
        failures.append("TASK_DIMENSIONS_BAD")
    for seed in seeds:
        row_counts = task.get("row_counts", {}).get(str(seed), {})
        for split in SPLITS:
            if int(row_counts.get(split, 0)) <= 0:
                failures.append(f"TASK_SPLIT_EMPTY:{seed}:{split}")

    for name, artifact in (
        ("training", training),
        ("quantization", quant),
        ("repair", repair),
        ("pruning", pruning),
        ("registry", registry),
        ("router", router),
        ("systems", systems),
        ("mutation", mutation),
        ("leakage", leakage),
        ("aggregate", aggregate),
    ):
        if not str(artifact.get("schema_version", "")).startswith("e7n_"):
            failures.append(f"SCHEMA_BAD:{name}")

    result_rows = systems.get("rows", [])
    expected_rows = len(seeds) * len(SYSTEMS)
    if len(result_rows) != expected_rows:
        failures.append(f"SYSTEM_RESULT_ROW_COUNT_BAD:expected={expected_rows}:actual={len(result_rows)}")
    if {(row.get("system"), int(row.get("seed"))) for row in result_rows} != {(system, seed) for system in SYSTEMS for seed in seeds}:
        failures.append("SYSTEM_RESULT_SET_BAD")
    aggregate_systems = aggregate.get("systems", {})
    if set(aggregate_systems) != set(SYSTEMS):
        failures.append("AGGREGATE_SYSTEM_SET_BAD")
    for system in SYSTEMS:
        mean = aggregate_systems.get(system, {}).get("mean", {})
        for metric in REQUIRED_MEAN_METRICS:
            if metric not in mean:
                failures.append(f"AGGREGATE_MEAN_MISSING:{system}:{metric}")
        if aggregate_systems.get(system, {}).get("seed_count") != len(seeds):
            failures.append(f"AGGREGATE_SEED_COUNT_BAD:{system}")

    for row in result_rows:
        system = row.get("system")
        seed = row.get("seed")
        if "evals" not in row:
            failures.append(f"ROW_EVALS_MISSING:{system}:{seed}")
            continue
        for split in SPLITS:
            split_eval = row["evals"].get(split, {})
            if not split_eval.get("row_level_samples"):
                failures.append(f"ROW_SAMPLES_MISSING:{system}:{seed}:{split}")
            for key in ("answer_accuracy", "route_accuracy", "router_call_usefulness", "raw_usefulness", "bit_budget", "active_parameter_count", "latency_cost_estimate", "stability_under_repeated_calls"):
                if key not in split_eval:
                    failures.append(f"SPLIT_METRIC_MISSING:{system}:{seed}:{split}:{key}")
        if system not in {"symbolic_proxy_pocket_reference", "random_pocket_control"}:
            summary_state = row.get("state_summary", {})
            if not summary_state.get("array_hashes"):
                failures.append(f"STATE_ARRAY_HASHES_MISSING:{system}:{seed}")
            if not row.get("state_hash"):
                failures.append(f"STATE_HASH_MISSING:{system}:{seed}")
            if int(row.get("parameter_count", 0)) <= 0:
                failures.append(f"PARAMETER_COUNT_BAD:{system}:{seed}")

    if len(training.get("rows", [])) != len(seeds):
        failures.append("TRAINING_ROW_COUNT_BAD")
    for row in training.get("rows", []):
        if row.get("backprop_used") is not True:
            failures.append(f"TRAINING_BACKPROP_FLAG_BAD:{row.get('seed')}")
        if not row.get("training_history"):
            failures.append(f"TRAINING_HISTORY_MISSING:{row.get('seed')}")
        if row.get("device") not in {"cuda", "cpu"}:
            failures.append(f"TRAINING_DEVICE_BAD:{row.get('seed')}")

    if len(quant.get("rows", {})) != len(seeds):
        failures.append("QUANTIZATION_SEED_COUNT_BAD")
    for seed, seed_report in quant.get("rows", {}).items():
        for precision in ("int8", "int4", "ternary", "binary"):
            if precision not in seed_report:
                failures.append(f"QUANTIZATION_PRECISION_MISSING:{seed}:{precision}")

    mutation_rows = mutation.get("rows", [])
    expected_mutation_rows = len(seeds) * len(MUTATION_SYSTEMS)
    if len(mutation_rows) != expected_mutation_rows:
        failures.append(f"MUTATION_ROW_COUNT_BAD:expected={expected_mutation_rows}:actual={len(mutation_rows)}")
    for row in mutation_rows:
        system = row.get("system")
        seed = row.get("seed")
        attempts = int(row.get("mutation_attempts", 0))
        accepted = int(row.get("accepted_mutations", 0))
        rejected = int(row.get("rejected_mutations", 0))
        rollback = int(row.get("rollback_count", -1))
        if attempts <= 0 or "accepted_mutations" not in row or "rejected_mutations" not in row:
            failures.append(f"MUTATION_COUNTS_BAD:{system}:{seed}")
        if accepted + rejected != attempts:
            failures.append(f"MUTATION_ACCEPT_REJECT_SUM_BAD:{system}:{seed}")
        if rollback != rejected:
            failures.append(f"ROLLBACK_MISMATCH:{system}:{seed}")
        if row.get("initial_candidate_hash") == row.get("final_candidate_hash"):
            failures.append(f"PARAMETER_DIFF_MISSING:{system}:{seed}")
        if not row.get("parameter_diff_hash"):
            failures.append(f"PARAMETER_DIFF_HASH_MISSING:{system}:{seed}")
        if not row.get("history"):
            failures.append(f"MUTATION_HISTORY_EMPTY:{system}:{seed}")

    registry_rows = registry.get("rows", [])
    if len(registry_rows) != expected_rows:
        failures.append("REGISTRY_ROW_COUNT_BAD")
    router_rows = router.get("rows", [])
    if len(router_rows) != expected_rows:
        failures.append("ROUTER_ROW_COUNT_BAD")

    if leakage.get("target_class_in_input") is not False:
        failures.append("LEAK_TARGET_CLASS_IN_INPUT")
    if leakage.get("row_id_used_as_input") is not False:
        failures.append("LEAK_ROW_ID_USED")
    if leakage.get("symbolic_proxy_reference_primary") is not False:
        failures.append("LEAK_SYMBOLIC_PROXY_PRIMARY")
    if leakage.get("mutation_system_uses_optimizer_or_backprop") is not False:
        failures.append("LEAK_MUTATION_BACKPROP")
    if float(leakage.get("random_control_accuracy", 1.0)) > 0.35:
        failures.append("RANDOM_CONTROL_TOO_HIGH")

    if replay.get("internal_replay_passed") is not True:
        failures.append("DETERMINISTIC_REPLAY_FAILED")
    for name, item in replay.get("hash_comparisons", {}).items():
        if item.get("match") is not True:
            failures.append(f"REPLAY_HASH_MISMATCH:{name}")

    if decision.get("decision") not in VALID_DECISIONS:
        failures.append("DECISION_LABEL_BAD")
    if summary.get("decision") != decision.get("decision"):
        failures.append("SUMMARY_DECISION_MISMATCH")
    if summary.get("best_non_control_system") != aggregate.get("best_non_control_system"):
        failures.append("SUMMARY_BEST_MISMATCH")
    if summary.get("deterministic_replay_passed") is not True or decision.get("deterministic_replay_passed") is not True:
        failures.append("REPLAY_FLAG_NOT_PROPAGATED")

    progress = jsonl_rows(out / "progress.jsonl")
    events = {row.get("event") for row in progress}
    for event in ("startup", "tasks_generated", "gradient_epoch", "seed_pretrain_complete", "mutation_jobs_submitted", "mutation_generation", "mutation_job_complete", "deterministic_replay_start", "deterministic_replay_complete", "final_artifacts_written"):
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")
    if not jsonl_rows(out / "hardware_heartbeat.jsonl"):
        failures.append("HARDWARE_HEARTBEAT_EMPTY")
    if "completed_rows" not in load_json(out / "partial_aggregate_snapshot.json"):
        failures.append("PARTIAL_AGGREGATE_BAD")

    report_text = (out / "report.md").read_text(encoding="utf-8").lower()
    for banned in ("agi", "consciousness", "model-scale claim", "raw-language proof"):
        if banned in report_text and "does not" not in report_text:
            failures.append(f"BROAD_CLAIM_IN_REPORT:{banned}")
    return failures


def write_summary(out: Path, failures: list[str]) -> None:
    summary_path = out / "summary.json"
    if not summary_path.exists():
        return
    summary = load_json(summary_path)
    summary["checker_failure_count"] = len(failures)
    summary["checker_failures"] = failures
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="target/pilot_wave/e7n_real_numeric_pocket_core_bridge_probe")
    parser.add_argument("--write-summary", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = resolve_out(args.out)
    failures = static_scan()
    if out.exists():
        failures.extend(check_artifacts(out))
    else:
        failures.append(f"OUT_MISSING:{out}")
    if args.write_summary and out.exists():
        write_summary(out, failures)
    result = {"schema_version": "e7n_checker_result_v1", "out": out.relative_to(REPO_ROOT).as_posix(), "failure_count": len(failures), "failures": failures}
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

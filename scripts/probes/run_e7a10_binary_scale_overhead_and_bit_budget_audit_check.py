#!/usr/bin/env python3
"""Checker for E7A10 binary scale overhead and bit-budget audit artifacts."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = "scripts/probes/run_e7a10_binary_scale_overhead_and_bit_budget_audit.py"
CHECKER = "scripts/probes/run_e7a10_binary_scale_overhead_and_bit_budget_audit_check.py"
METHODS = (
    "float32_matrix_core",
    "int8_direct",
    "int4_direct",
    "int3_direct",
    "ternary_block_scale_qat",
    "binary_direct_block_scale",
    "binary_minimal_scale_qat",
    "binary_global_scale_qat",
    "binary_block_scale_qat",
    "binary_channel_scale_qat",
    "binary_channel_scale_qat_paramwise_freeze",
)
MUTATION_METHODS = ("binary_channel_scale_qat_paramwise_freeze",)
VALID_DECISIONS = {
    "e7a10_binary_same_budget_preferred",
    "e7a10_binary_scale_overhead_required",
    "e7a10_int4_quality_path_preferred",
    "e7a10_ternary_balanced_path_preferred",
    "e7a10_global_or_block_binary_viable",
    "e7a10_binary_width_scaling_not_worth_it",
    "e7a10_invalid_artifact_detected",
}
BASE_REQUIRED = (
    "e7a10_backend_manifest.json",
    "e7a10_task_generation_report.json",
    "e7a10_method_comparison_report.json",
    "e7a10_scale_overhead_report.json",
    "e7a10_bit_budget_width_scaling_report.json",
    "e7a10_mutation_history.json",
    "e7a10_no_synthetic_metric_audit.json",
    "e7a10_runtime_report.json",
    "e7a10_deterministic_replay_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "progress.jsonl",
    "e7a10_row_level_eval_sample_heldout.json",
    "e7a10_row_level_eval_sample_ood.json",
    "e7a10_row_level_eval_sample_counterfactual.json",
    "e7a10_row_level_eval_sample_adversarial.json",
)
REQUIRED_ROW_METRICS = (
    "width",
    "matrix_shape",
    "matrix_cells",
    "parameter_count",
    "eval_accuracy",
    "train_accuracy",
    "validation_accuracy",
    "heldout_accuracy",
    "ood_accuracy",
    "counterfactual_accuracy",
    "adversarial_accuracy",
    "generalization_gap",
    "training_mode",
    "solve_passed",
    "bit_cost",
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


def progress_rows(out: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (out / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def ast_scan() -> list[str]:
    failures: list[str] = []
    runner = REPO_ROOT / RUNNER
    checker = REPO_ROOT / CHECKER
    for path in (runner, checker):
        if not path.exists():
            failures.append(f"MISSING_STATIC_FILE:{path.relative_to(REPO_ROOT).as_posix()}")
            return failures
    text = runner.read_text(encoding="utf-8")
    for token in (
        "binary_minimal_scale_qat",
        "binary_global_scale_qat",
        "binary_block_scale_qat",
        "binary_channel_scale_qat",
        "binary_channel_scale_qat_paramwise_freeze",
        "scale_storage_mode",
        "reference_int4_budget_bits",
        "ProcessPoolExecutor",
        "deterministic_replay",
    ):
        if token not in text:
            failures.append(f"TOKEN_MISSING_IN_RUNNER:{token}")
    tree = ast.parse(text)
    forbidden_calls = {"backward", "step", "zero_grad"}
    mutation_function_names = {"run_freeze_worker"}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in mutation_function_names:
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func = child.func
                    name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
                    if name in forbidden_calls:
                        failures.append(f"MUTATION_REPAIR_BACKPROP_CALL:{node.name}:{name}")
                if isinstance(child, ast.Attribute) and child.attr in {"AdamW", "SGD", "RMSprop"}:
                    failures.append(f"MUTATION_REPAIR_OPTIMIZER_REFERENCE:{node.name}:{child.attr}")
    return failures


def check_artifacts(out: Path) -> list[str]:
    failures: list[str] = []
    for name in BASE_REQUIRED:
        if not (out / name).exists():
            failures.append(f"MISSING_ARTIFACT:{name}")
    if failures:
        return failures

    manifest = load_json(out / "e7a10_backend_manifest.json")
    task = load_json(out / "e7a10_task_generation_report.json")
    comparison = load_json(out / "e7a10_method_comparison_report.json")
    scale = load_json(out / "e7a10_scale_overhead_report.json")
    budget = load_json(out / "e7a10_bit_budget_width_scaling_report.json")
    mutation = load_json(out / "e7a10_mutation_history.json")
    no_synth = load_json(out / "e7a10_no_synthetic_metric_audit.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    replay = load_json(out / "e7a10_deterministic_replay_report.json")
    decision = load_json(out / "decision.json")
    summary = load_json(out / "summary.json")
    widths = [int(width) for width in manifest.get("widths", [])]
    width_keys = {str(width) for width in widths}

    if tuple(manifest.get("methods", [])) != METHODS:
        failures.append("MANIFEST_METHODS_MISMATCH")
    if manifest.get("new_architecture_added") is not False:
        failures.append("NEW_ARCHITECTURE_FLAG_BAD")
    if manifest.get("parallel_replay_supported") is not True:
        failures.append("PARALLEL_REPLAY_FLAG_MISSING")
    if manifest.get("broad_claims_intentionally_deferred") is not True:
        failures.append("BROAD_CLAIMS_DEFER_FLAG_MISSING")
    if int(manifest.get("reference_width", -1)) not in widths:
        failures.append("REFERENCE_WIDTH_NOT_IN_WIDTHS")

    if task.get("inherits_task_from") != "E7A9_BINARY_FREEZE_POLICY_UPPER_BOUND_AUDIT":
        failures.append("TASK_INHERITANCE_BAD")
    for split in ("train", "validation", "heldout", "ood", "counterfactual", "adversarial"):
        if task.get("splits", {}).get(split, {}).get("row_count", 0) < 1:
            failures.append(f"SPLIT_EMPTY:{split}")

    systems = aggregate.get("systems", {})
    if set(systems) != set(METHODS):
        failures.append("AGGREGATE_METHOD_SET_MISMATCH")
    for method in METHODS:
        if set(systems.get(method, {})) != width_keys:
            failures.append(f"METHOD_WIDTHS_MISMATCH:{method}")
        for width in widths:
            row = systems.get(method, {}).get(str(width), {})
            for key in REQUIRED_ROW_METRICS:
                if key not in row:
                    failures.append(f"ROW_METRIC_MISSING:{method}:{width}:{key}")
            bit = row.get("bit_cost", {})
            for key in ("total_bit_cost", "compression_vs_float32", "nominal_weight_bits", "scale_bit_cost", "scale_count"):
                if key not in bit:
                    failures.append(f"BIT_COST_FIELD_MISSING:{method}:{width}:{key}")
            if method in MUTATION_METHODS:
                for key in ("mutation_attempt_count", "accepted_mutation_count", "rejected_mutation_count", "rollback_count"):
                    if key not in row:
                        failures.append(f"MUTATION_METRIC_MISSING:{method}:{width}:{key}")

    if set(comparison.get("best_by_method", {})) != set(METHODS):
        failures.append("COMPARISON_METHODS_MISMATCH")
    for key in ("best_binary_same_bit_budget", "best_binary_any_budget"):
        if key not in comparison:
            failures.append(f"COMPARISON_FIELD_MISSING:{key}")
    if budget.get("reference_int4_budget_bits", 0) <= 0:
        failures.append("REFERENCE_BUDGET_MISSING")
    if not budget.get("best_binary_same_bit_budget"):
        failures.append("BEST_BINARY_SAME_BUDGET_MISSING")
    if len(budget.get("all_budget_rows", {})) < len(METHODS) * max(1, len(widths)):
        failures.append("BUDGET_ROWS_INCOMPLETE")
    if not scale.get("scale_count_includes_stored_float32_scales"):
        failures.append("SCALE_COUNT_POLICY_MISSING")
    expected_scale_rows = len(METHODS) * len(widths)
    if len(scale.get("rows", {})) != expected_scale_rows:
        failures.append("SCALE_ROW_COUNT_MISMATCH")

    expected_histories = len(widths) * len(MUTATION_METHODS)
    if len(mutation.get("rows", {})) != expected_histories:
        failures.append("MUTATION_HISTORY_RUN_COUNT_MISMATCH")
    for name, hist in mutation.get("rows", {}).items():
        if hist.get("mutation_attempt_count", 0) < 1:
            failures.append(f"NO_MUTATION_ATTEMPTS:{name}")
        if hist.get("rejected_mutation_count", 0) < 1:
            failures.append(f"NO_REJECTED_MUTATIONS:{name}")
        if hist.get("rejected_mutation_count") != hist.get("rollback_count"):
            failures.append(f"ROLLBACK_MISMATCH:{name}")
        if hist.get("accepted_mutation_count", 0) < 1:
            failures.append(f"NO_ACCEPTED_MUTATION:{name}")

    if no_synth.get("generated_from_row_level_eval") is not True:
        failures.append("NO_SYNTH_ROW_EVAL_FLAG_BAD")
    if no_synth.get("row_level_samples_present") is not True:
        failures.append("ROW_LEVEL_SAMPLE_FLAG_BAD")
    if no_synth.get("mutation_repair_used_optimizer_or_backprop") is not False:
        failures.append("MUTATION_BACKPROP_FLAG_BAD")
    if no_synth.get("scale_overhead_counted_in_bit_budget") is not True:
        failures.append("SCALE_OVERHEAD_AUDIT_FLAG_BAD")
    if no_synth.get("same_budget_width_scaling_evaluated") is not True:
        failures.append("WIDTH_SCALING_AUDIT_FLAG_BAD")
    if no_synth.get("new_architecture_added") is not False:
        failures.append("NEW_ARCHITECTURE_AUDIT_FLAG_BAD")
    if no_synth.get("hardcoded_improvement_flags_present") is not False:
        failures.append("HARDCODED_IMPROVEMENT_FLAG_BAD")

    if replay.get("internal_replay_passed") is not True:
        failures.append("DETERMINISTIC_REPLAY_FAILED")
    for artifact, row in replay.get("hash_comparisons", {}).items():
        if row.get("match") is not True:
            failures.append(f"REPLAY_HASH_MISMATCH:{artifact}")
    if decision.get("decision") not in VALID_DECISIONS:
        failures.append("DECISION_NOT_ALLOWED")
    if decision.get("deterministic_replay_passed") is not True:
        failures.append("DECISION_REPLAY_FLAG_BAD")
    if summary.get("decision") != decision.get("decision"):
        failures.append("SUMMARY_DECISION_MISMATCH")

    events = {row.get("event") for row in progress_rows(out)}
    for event in ("startup", "task_generated", "scale_qat_epoch", "paramwise_freeze_step", "deterministic_replay_complete", "final_artifacts_written"):
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")

    report_text = (out / "report.md").read_text(encoding="utf-8").lower()
    for forbidden in ("consciousness", "sentience", "agi", "model-scale", "natural-language reasoning"):
        if forbidden in report_text:
            failures.append(f"FORBIDDEN_REPORT_CLAIM_TOKEN:{forbidden}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="target/pilot_wave/e7a10_binary_scale_overhead_and_bit_budget_audit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = resolve_out(args.out)
    failures = ast_scan() + check_artifacts(out)
    report = {
        "schema_version": "e7a10_checker_report_v1",
        "out": out.relative_to(REPO_ROOT).as_posix(),
        "failure_count": len(failures),
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

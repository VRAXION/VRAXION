#!/usr/bin/env python3
"""Checker for E7A9 binary freeze policy upper-bound audit artifacts."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = "scripts/probes/run_e7a9_binary_freeze_policy_upper_bound_audit.py"
CHECKER = "scripts/probes/run_e7a9_binary_freeze_policy_upper_bound_audit_check.py"
METHODS = (
    "float32_matrix_core",
    "int8_direct",
    "int4_direct",
    "ternary_qat_reference",
    "binary_direct",
    "binary_qat_baseline",
    "binary_qat_best_effort",
    "binary_distance_paramwise_freeze",
    "binary_sensitivity_paramwise_freeze",
    "binary_qat_warmstart_paramwise_freeze",
    "binary_direct_mutation_repair",
    "mixed_input_int4_state_binary_output_int4",
    "mixed_input_ternary_state_binary_output_int4",
)
MUTATION_METHODS = (
    "binary_distance_paramwise_freeze",
    "binary_sensitivity_paramwise_freeze",
    "binary_qat_warmstart_paramwise_freeze",
    "binary_direct_mutation_repair",
)
FREEZE_METHODS = (
    "binary_distance_paramwise_freeze",
    "binary_sensitivity_paramwise_freeze",
    "binary_qat_warmstart_paramwise_freeze",
)
VALID_DECISIONS = {
    "e7a9_binary_quality_competitive",
    "e7a9_binary_not_quality_competitive",
    "e7a9_mixed_precision_matrix_core_preferred",
    "e7a9_binary_paramwise_freeze_positive",
    "e7a9_qat_upper_bound_remains_preferred",
    "e7a9_invalid_artifact_detected",
}
BASE_REQUIRED = (
    "e7a9_backend_manifest.json",
    "e7a9_task_generation_report.json",
    "e7a9_method_comparison_report.json",
    "e7a9_binary_freeze_schedule_report.json",
    "e7a9_qat_upper_bound_report.json",
    "e7a9_precision_tradeoff_report.json",
    "e7a9_mixed_precision_report.json",
    "e7a9_mutation_history.json",
    "e7a9_no_synthetic_metric_audit.json",
    "e7a9_runtime_report.json",
    "e7a9_deterministic_replay_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "progress.jsonl",
    "e7a9_row_level_eval_sample_heldout.json",
    "e7a9_row_level_eval_sample_ood.json",
    "e7a9_row_level_eval_sample_counterfactual.json",
    "e7a9_row_level_eval_sample_adversarial.json",
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
        "train_binary_qat_best_effort",
        "binary_distance_paramwise_freeze",
        "binary_sensitivity_paramwise_freeze",
        "binary_qat_warmstart_paramwise_freeze",
        "make_mixed_candidate",
        "deterministic_replay",
        "ProcessPoolExecutor",
    ):
        if token not in text:
            failures.append(f"TOKEN_MISSING_IN_RUNNER:{token}")
    tree = ast.parse(text)
    forbidden_calls = {"backward", "step", "zero_grad"}
    mutation_function_names = {
        "tiny_repair_step",
        "run_paramwise_freeze",
        "run_direct_binary_repair",
        "mutate_binary_candidate",
        "score_candidate",
    }
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

    manifest = load_json(out / "e7a9_backend_manifest.json")
    task = load_json(out / "e7a9_task_generation_report.json")
    comparison = load_json(out / "e7a9_method_comparison_report.json")
    freeze = load_json(out / "e7a9_binary_freeze_schedule_report.json")
    qat = load_json(out / "e7a9_qat_upper_bound_report.json")
    tradeoff = load_json(out / "e7a9_precision_tradeoff_report.json")
    mixed = load_json(out / "e7a9_mixed_precision_report.json")
    mutation = load_json(out / "e7a9_mutation_history.json")
    no_synth = load_json(out / "e7a9_no_synthetic_metric_audit.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    replay = load_json(out / "e7a9_deterministic_replay_report.json")
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

    if task.get("inherits_task_from") != "E7A8_PROGRESSIVE_QUANT_FREEZE_PLATEAU_REPAIR":
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
            for key in ("total_bit_cost", "compression_vs_float32", "nominal_weight_bits"):
                if key not in bit:
                    failures.append(f"BIT_COST_FIELD_MISSING:{method}:{width}:{key}")
            if method in MUTATION_METHODS:
                for key in ("mutation_attempt_count", "accepted_mutation_count", "rejected_mutation_count", "rollback_count"):
                    if key not in row:
                        failures.append(f"MUTATION_METRIC_MISSING:{method}:{width}:{key}")
            if method in FREEZE_METHODS:
                for key in ("freeze_rounds_count", "final_frozen_parameter_ratio", "input_projection_frozen_ratio"):
                    if key not in row:
                        failures.append(f"FREEZE_METRIC_MISSING:{method}:{width}:{key}")

    if set(comparison.get("best_by_method", {})) != set(METHODS):
        failures.append("COMPARISON_METHODS_MISMATCH")
    for key in ("best_binary_method", "best_mixed_method"):
        if key not in comparison:
            failures.append(f"COMPARISON_FIELD_MISSING:{key}")

    if set(freeze.get("rows", {})) != width_keys:
        failures.append("FREEZE_WIDTHS_MISMATCH")
    for width in widths:
        rows = freeze.get("rows", {}).get(str(width), {})
        if set(rows) != set(FREEZE_METHODS):
            failures.append(f"FREEZE_METHODS_MISMATCH:{width}")
        for method, row in rows.items():
            if not row.get("freeze_history"):
                failures.append(f"FREEZE_HISTORY_EMPTY:{method}:{width}")
            if "parameter_diff" not in row:
                failures.append(f"FREEZE_PARAMETER_DIFF_MISSING:{method}:{width}")

    if set(qat.get("rows", {})) != width_keys:
        failures.append("QAT_WIDTHS_MISMATCH")
    for width in widths:
        row = qat.get("rows", {}).get(str(width), {})
        for key in ("binary_qat_baseline", "binary_qat_best_effort", "ternary_qat_reference"):
            if key not in row:
                failures.append(f"QAT_FIELD_MISSING:{width}:{key}")

    for key in ("float32_matrix_core", "int8_direct", "int4_direct", "ternary_qat_reference", "binary_qat_best_effort"):
        if key not in tradeoff.get("rows", {}):
            failures.append(f"TRADEOFF_ROW_MISSING:{key}")
    for key in ("mixed_input_int4_state_binary_output_int4", "mixed_input_ternary_state_binary_output_int4", "best_mixed_method"):
        if key not in mixed.get("rows", {}):
            failures.append(f"MIXED_ROW_MISSING:{key}")

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
    for event in ("startup", "task_generated", "best_effort_qat_epoch", "paramwise_freeze_step", "deterministic_replay_complete", "final_artifacts_written"):
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")

    report_text = (out / "report.md").read_text(encoding="utf-8").lower()
    for forbidden in ("consciousness", "sentience", "agi", "model-scale", "natural-language reasoning"):
        if forbidden in report_text:
            failures.append(f"FORBIDDEN_REPORT_CLAIM_TOKEN:{forbidden}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="target/pilot_wave/e7a9_binary_freeze_policy_upper_bound_audit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = resolve_out(args.out)
    failures = ast_scan() + check_artifacts(out)
    report = {
        "schema_version": "e7a9_checker_report_v1",
        "out": out.relative_to(REPO_ROOT).as_posix(),
        "failure_count": len(failures),
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

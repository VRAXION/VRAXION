#!/usr/bin/env python3
"""Checker for E7A8 progressive quant freeze plateau repair artifacts."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = "scripts/probes/run_e7a8_progressive_quant_freeze_plateau_repair.py"
CHECKER = "scripts/probes/run_e7a8_progressive_quant_freeze_plateau_repair_check.py"
METHODS = (
    "baseline_float_matrix_core",
    "direct_low_bit_quant",
    "post_quant_mutation_repair",
    "distance_only_progressive_freeze",
    "sensitivity_aware_progressive_freeze",
    "input_projection_aware_progressive_freeze",
    "blockwise_scale_mutation_repair",
    "qat_reference",
)
PROGRESSIVE_METHODS = (
    "distance_only_progressive_freeze",
    "sensitivity_aware_progressive_freeze",
    "input_projection_aware_progressive_freeze",
)
MUTATION_METHODS = (
    "post_quant_mutation_repair",
    "distance_only_progressive_freeze",
    "sensitivity_aware_progressive_freeze",
    "input_projection_aware_progressive_freeze",
    "blockwise_scale_mutation_repair",
)
VALID_DECISIONS = {
    "e7a8_input_projection_aware_progressive_freeze_positive",
    "e7a8_sensitivity_aware_freeze_positive",
    "e7a8_distance_only_freeze_sufficient",
    "e7a8_blockwise_scale_repair_positive",
    "e7a8_progressive_freeze_no_advantage",
    "e7a8_progressive_freeze_overfit_or_brittle",
    "e7a8_invalid_artifact_detected",
}
BASE_REQUIRED = (
    "e7a8_backend_manifest.json",
    "e7a8_task_generation_report.json",
    "e7a8_method_comparison_report.json",
    "e7a8_freeze_schedule_report.json",
    "e7a8_input_projection_damage_recovery_report.json",
    "e7a8_mutation_repair_report.json",
    "e7a8_mutation_history.json",
    "e7a8_no_synthetic_metric_audit.json",
    "e7a8_runtime_report.json",
    "e7a8_deterministic_replay_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "progress.jsonl",
    "e7a8_row_level_eval_sample_heldout.json",
    "e7a8_row_level_eval_sample_ood.json",
    "e7a8_row_level_eval_sample_counterfactual.json",
    "e7a8_row_level_eval_sample_adversarial.json",
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
        "run_progressive_freeze",
        "blockwise_scale_mutation_repair",
        "input_projection_aware_progressive_freeze",
        "rollback_needed",
        "plateau_patience",
        "deterministic_replay",
        "ProcessPoolExecutor",
        "train_qat_core",
    ):
        if token not in text:
            failures.append(f"TOKEN_MISSING_IN_RUNNER:{token}")
    tree = ast.parse(text)
    forbidden_calls = {"backward", "step", "zero_grad"}
    mutation_function_names = {
        "repair_loop",
        "run_post_quant_repair",
        "run_progressive_freeze",
        "run_scale_repair",
        "mutate_candidate_q",
        "mutate_candidate_scale",
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

    manifest = load_json(out / "e7a8_backend_manifest.json")
    task = load_json(out / "e7a8_task_generation_report.json")
    comparison = load_json(out / "e7a8_method_comparison_report.json")
    freeze = load_json(out / "e7a8_freeze_schedule_report.json")
    input_report = load_json(out / "e7a8_input_projection_damage_recovery_report.json")
    repair = load_json(out / "e7a8_mutation_repair_report.json")
    mutation = load_json(out / "e7a8_mutation_history.json")
    no_synth = load_json(out / "e7a8_no_synthetic_metric_audit.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    replay = load_json(out / "e7a8_deterministic_replay_report.json")
    decision = load_json(out / "decision.json")
    summary = load_json(out / "summary.json")
    widths = [int(width) for width in manifest.get("widths", [])]
    width_keys = {str(width) for width in widths}
    levels = tuple(manifest.get("target_levels", []))

    if tuple(manifest.get("methods", [])) != METHODS:
        failures.append("MANIFEST_METHODS_MISMATCH")
    if not levels:
        failures.append("TARGET_LEVELS_EMPTY")
    if set(levels) - {"int3", "ternary", "binary"}:
        failures.append("TARGET_LEVELS_UNEXPECTED")
    if manifest.get("new_architecture_added") is not False:
        failures.append("NEW_ARCHITECTURE_FLAG_BAD")
    if manifest.get("cpu_repair_and_qat_overlap_supported") is not True:
        failures.append("OVERLAP_SUPPORT_FLAG_MISSING")
    if manifest.get("parallel_replay_supported") is not True:
        failures.append("PARALLEL_REPLAY_FLAG_MISSING")
    if manifest.get("broad_claims_intentionally_deferred") is not True:
        failures.append("BROAD_CLAIMS_DEFER_FLAG_MISSING")

    if task.get("inherits_task_from") != "E7A7_LOW_BIT_REPAIR_OPERATOR_AUDIT":
        failures.append("TASK_INHERITANCE_BAD")
    for split in ("train", "validation", "heldout", "ood", "counterfactual", "adversarial"):
        if task.get("splits", {}).get(split, {}).get("row_count", 0) < 1:
            failures.append(f"SPLIT_EMPTY:{split}")

    systems = aggregate.get("systems", {})
    if set(systems) != set(METHODS):
        failures.append("AGGREGATE_METHOD_SET_MISMATCH")
    if set(systems.get("baseline_float_matrix_core", {})) != width_keys:
        failures.append("FLOAT_WIDTH_ROWS_MISMATCH")
    for width in widths:
        row = systems.get("baseline_float_matrix_core", {}).get(str(width), {})
        for key in REQUIRED_ROW_METRICS:
            if key not in row:
                failures.append(f"FLOAT_ROW_METRIC_MISSING:{width}:{key}")
    for method in METHODS:
        if method == "baseline_float_matrix_core":
            continue
        if set(systems.get(method, {})) != set(levels):
            failures.append(f"METHOD_LEVELS_MISMATCH:{method}")
        for level in levels:
            if set(systems.get(method, {}).get(level, {})) != width_keys:
                failures.append(f"METHOD_WIDTHS_MISMATCH:{method}:{level}")
            for width in widths:
                row = systems.get(method, {}).get(level, {}).get(str(width), {})
                for key in REQUIRED_ROW_METRICS:
                    if key not in row:
                        failures.append(f"ROW_METRIC_MISSING:{method}:{level}:{width}:{key}")
                if method in MUTATION_METHODS:
                    for key in ("mutation_attempt_count", "accepted_mutation_count", "rejected_mutation_count", "rollback_count"):
                        if key not in row:
                            failures.append(f"MUTATION_METRIC_MISSING:{method}:{level}:{width}:{key}")
                if method in PROGRESSIVE_METHODS:
                    for key in ("freeze_rounds_count", "final_frozen_parameter_ratio", "input_projection_frozen_ratio", "recurrent_state_frozen_ratio", "output_head_frozen_ratio"):
                        if key not in row:
                            failures.append(f"FREEZE_METRIC_MISSING:{method}:{level}:{width}:{key}")

    if set(comparison.get("rows", {})) != set(levels):
        failures.append("COMPARISON_LEVELS_MISMATCH")
    for level in levels:
        rows = comparison.get("rows", {}).get(level, {})
        for method in METHODS:
            if method == "baseline_float_matrix_core":
                continue
            if method not in rows:
                failures.append(f"COMPARISON_METHOD_MISSING:{level}:{method}")
        if "best_non_qat_method" not in rows:
            failures.append(f"BEST_NON_QAT_MISSING:{level}")

    if set(freeze.get("rows", {})) != set(levels):
        failures.append("FREEZE_LEVELS_MISMATCH")
    for level in levels:
        if set(freeze.get("rows", {}).get(level, {})) != width_keys:
            failures.append(f"FREEZE_WIDTHS_MISMATCH:{level}")
        for width in widths:
            rows = freeze.get("rows", {}).get(level, {}).get(str(width), {})
            if set(rows) != set(PROGRESSIVE_METHODS):
                failures.append(f"FREEZE_METHODS_MISMATCH:{level}:{width}")
            for method, row in rows.items():
                if not row.get("freeze_history"):
                    failures.append(f"FREEZE_HISTORY_EMPTY:{level}:{width}:{method}")
                if "parameter_diff" not in row:
                    failures.append(f"FREEZE_PARAMETER_DIFF_MISSING:{level}:{width}:{method}")

    if set(input_report.get("rows", {})) != set(levels):
        failures.append("INPUT_REPORT_LEVELS_MISMATCH")
    for level in levels:
        if set(input_report.get("rows", {}).get(level, {})) != width_keys:
            failures.append(f"INPUT_REPORT_WIDTHS_MISMATCH:{level}")
        for width in widths:
            row = input_report.get("rows", {}).get(level, {}).get(str(width), {})
            for key in ("input_projection_sensitivity", "input_projection_aware_metrics", "distance_only_metrics", "blockwise_scale_metrics"):
                if key not in row:
                    failures.append(f"INPUT_REPORT_FIELD_MISSING:{level}:{width}:{key}")

    if repair.get("all_mutation_methods_have_attempts") is not True:
        failures.append("MUTATION_ATTEMPT_FLAG_BAD")
    if repair.get("all_rejected_mutations_rolled_back") is not True:
        failures.append("ROLLBACK_FLAG_BAD")
    if repair.get("at_least_one_mutation_accepted") is not True:
        failures.append("NO_ACCEPTED_MUTATION")
    expected_histories = len(levels) * len(widths) * len(MUTATION_METHODS)
    if len(mutation.get("rows", {})) != expected_histories:
        failures.append("MUTATION_HISTORY_RUN_COUNT_MISMATCH")
    for name, hist in mutation.get("rows", {}).items():
        if hist.get("mutation_attempt_count", 0) < 1:
            failures.append(f"NO_MUTATION_ATTEMPTS:{name}")
        if hist.get("rejected_mutation_count", 0) < 1:
            failures.append(f"NO_REJECTED_MUTATIONS:{name}")
        if hist.get("rejected_mutation_count") != hist.get("rollback_count"):
            failures.append(f"ROLLBACK_MISMATCH:{name}")

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
    for event in ("startup", "task_generated", "repair_generation", "freeze_round", "deterministic_replay_complete", "final_artifacts_written"):
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")

    report_text = (out / "report.md").read_text(encoding="utf-8").lower()
    for forbidden in ("consciousness", "sentience", "agi", "model-scale", "natural-language reasoning"):
        if forbidden in report_text:
            failures.append(f"FORBIDDEN_REPORT_CLAIM_TOKEN:{forbidden}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="target/pilot_wave/e7a8_progressive_quant_freeze_plateau_repair")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = resolve_out(args.out)
    failures = ast_scan() + check_artifacts(out)
    report = {
        "schema_version": "e7a8_checker_report_v1",
        "out": out.relative_to(REPO_ROOT).as_posix(),
        "failure_count": len(failures),
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

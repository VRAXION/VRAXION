#!/usr/bin/env python3
"""Checker for E7A5 operator-cell matrix incremental scan artifacts."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = "scripts/probes/run_e7a5_operator_cell_matrix_incremental_scan.py"
CHECKER = "scripts/probes/run_e7a5_operator_cell_matrix_incremental_scan_check.py"
VARIANTS = (
    "plain_matrix_core_baseline",
    "soft_mask_matrix",
    "edge_bias_shared_activation",
    "per_cell_activation_soft_mixture",
    "source_target_trace_operand_cell",
)
SYSTEMS = (*VARIANTS, "random_control")
VALID_DECISIONS = {
    "e7a5_edge_bias_shared_activation_positive",
    "e7a5_per_cell_activation_positive",
    "e7a5_operand_cell_positive",
    "e7a5_operator_cell_no_advantage_detected",
    "e7a5_operator_cell_overfit_or_search_noise",
    "e7a5_mutation_repair_value_confirmed",
    "e7a5_invalid_artifact_detected",
}
BASE_REQUIRED = (
    "e7a5_backend_manifest.json",
    "e7a5_task_generation_report.json",
    "e7a5_incremental_comparison_report.json",
    "e7a5_quantization_report.json",
    "e7a5_mutation_repair_report.json",
    "e7a5_operator_cell_audit.json",
    "e7a5_training_history.json",
    "e7a5_mutation_history.json",
    "e7a5_no_synthetic_metric_audit.json",
    "e7a5_deterministic_replay_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "progress.jsonl",
    "e7a5_row_level_eval_sample_heldout.json",
    "e7a5_row_level_eval_sample_ood.json",
    "e7a5_row_level_eval_sample_counterfactual.json",
    "e7a5_row_level_eval_sample_adversarial.json",
    "e7a5_row_level_eval_sample_stress.json",
)
HASH_ARTIFACTS = (
    "e7a5_task_generation_report.json",
    "e7a5_incremental_comparison_report.json",
    "e7a5_quantization_report.json",
    "e7a5_mutation_repair_report.json",
    "e7a5_operator_cell_audit.json",
    "e7a5_no_synthetic_metric_audit.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
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
    "stress_accuracy",
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


def required_artifacts(out: Path) -> list[str]:
    manifest = load_json(out / "e7a5_backend_manifest.json")
    widths = [int(width) for width in manifest.get("widths", [])]
    variants = tuple(manifest.get("variants", []))
    names = list(BASE_REQUIRED)
    for width in widths:
        names.append(f"e7a5_candidate_random_control_width{width}_summary.json")
        for variant in variants:
            names.append(f"e7a5_candidate_{variant}_width{width}_summary.json")
            names.append(f"e7a5_training_history_{variant}_width{width}.json")
            names.append(f"e7a5_float_state_{variant}_width{width}.json")
            names.append(f"e7a5_candidate_{variant}_quantized_width{width}.json")
            names.append(f"e7a5_candidate_{variant}_quantized_width{width}_summary.json")
            names.append(f"e7a5_mutation_history_{variant}_width{width}.json")
            names.append(f"e7a5_candidate_{variant}_mutation_repair_width{width}_initial.json")
            names.append(f"e7a5_candidate_{variant}_mutation_repair_width{width}_final.json")
            names.append(f"e7a5_candidate_{variant}_mutation_repair_width{width}_summary.json")
            names.append(f"e7a5_parameter_diff_{variant}_mutation_repair_width{width}.json")
    return names


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
        *VARIANTS,
        "OperatorCellCore",
        "activation_mix",
        "quantize_state_dict",
        "run_quantized_repair",
        "mutation_repair_used_optimizer_or_backprop",
    ):
        if token not in text:
            failures.append(f"TOKEN_MISSING_IN_RUNNER:{token}")
    if "consciousness" in text.lower() or "agi" in text.lower():
        failures.append("FORBIDDEN_FINAL_CLAIM_TOKEN_IN_RUNNER")
    tree = ast.parse(text)
    torch_import_seen = False
    adamw_seen = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "torch":
                    torch_import_seen = True
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "torch":
            torch_import_seen = True
        if isinstance(node, ast.Attribute) and node.attr == "AdamW":
            adamw_seen = True
    if not torch_import_seen:
        failures.append("TORCH_IMPORT_MISSING")
    if not adamw_seen:
        failures.append("ADAMW_MISSING_FOR_FLOAT_VARIANTS")
    forbidden_calls = {"backward", "step", "zero_grad"}
    mutation_function_names = {"run_quantized_repair", "mutate_quantized", "score_quantized", "quantized_forward"}
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
    if not (out / "e7a5_backend_manifest.json").exists():
        return ["MISSING_ARTIFACT:e7a5_backend_manifest.json"]
    for name in required_artifacts(out):
        if not (out / name).exists():
            failures.append(f"MISSING_ARTIFACT:{name}")
    if failures:
        return failures

    manifest = load_json(out / "e7a5_backend_manifest.json")
    task = load_json(out / "e7a5_task_generation_report.json")
    comparison = load_json(out / "e7a5_incremental_comparison_report.json")
    quant = load_json(out / "e7a5_quantization_report.json")
    repair = load_json(out / "e7a5_mutation_repair_report.json")
    op_audit = load_json(out / "e7a5_operator_cell_audit.json")
    training = load_json(out / "e7a5_training_history.json")
    mutation = load_json(out / "e7a5_mutation_history.json")
    no_synth = load_json(out / "e7a5_no_synthetic_metric_audit.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    replay = load_json(out / "e7a5_deterministic_replay_report.json")
    decision = load_json(out / "decision.json")
    summary = load_json(out / "summary.json")
    widths = [int(width) for width in manifest.get("widths", [])]

    if tuple(manifest.get("variants", [])) != VARIANTS:
        failures.append("MANIFEST_VARIANTS_MISMATCH")
    if tuple(manifest.get("systems", [])) != SYSTEMS:
        failures.append("MANIFEST_SYSTEMS_MISMATCH")
    if manifest.get("final_e7_verdict_intentionally_deferred") is not True:
        failures.append("MANIFEST_FINAL_E7_NOT_DEFERRED")
    if manifest.get("cpu_repair_and_gpu_gradient_overlap_supported") is not True:
        failures.append("OVERLAP_SUPPORT_FLAG_MISSING")

    if task.get("inherits_task_from") != "E7A3_NEURAL_MATRIX_SUBSTRATE_HARNESS":
        failures.append("TASK_INHERITANCE_BAD")
    for split in ("train", "validation", "heldout", "ood", "counterfactual", "adversarial", "stress"):
        if task.get("splits", {}).get(split, {}).get("row_count", 0) < 1:
            failures.append(f"SPLIT_EMPTY:{split}")

    systems = aggregate.get("systems", {})
    if set(systems) != {"float", "quantized", "repair", "random_control"}:
        failures.append("AGGREGATE_MODE_SET_MISMATCH")
    for mode in ("float", "quantized", "repair"):
        if set(systems.get(mode, {})) != set(VARIANTS):
            failures.append(f"AGGREGATE_VARIANTS_MISMATCH:{mode}")
        for variant in VARIANTS:
            rows = systems.get(mode, {}).get(variant, {})
            if set(rows) != {str(width) for width in widths}:
                failures.append(f"WIDTH_ROWS_MISMATCH:{mode}:{variant}")
            for width in widths:
                row = rows.get(str(width), {})
                for key in REQUIRED_ROW_METRICS:
                    if key not in row:
                        failures.append(f"ROW_METRIC_MISSING:{mode}:{variant}:{width}:{key}")
                if row.get("parameter_count", 0) < 1:
                    failures.append(f"PARAMETER_COUNT_MISSING:{mode}:{variant}:{width}")
                if mode == "repair":
                    for key in ("mutation_attempt_count", "accepted_mutation_count", "rejected_mutation_count", "rollback_count"):
                        if key not in row:
                            failures.append(f"REPAIR_MUTATION_METRIC_MISSING:{variant}:{width}:{key}")
    if set(systems.get("random_control", {})) != {str(width) for width in widths}:
        failures.append("RANDOM_WIDTH_ROWS_MISMATCH")

    if comparison.get("baseline_variant") != "plain_matrix_core_baseline":
        failures.append("COMPARISON_BASELINE_BAD")
    if set(comparison.get("rows", {})) != set(VARIANTS):
        failures.append("COMPARISON_VARIANTS_MISMATCH")
    for variant, row in comparison.get("rows", {}).items():
        for key in ("float_best", "quantized_best", "repair_best", "float_delta_vs_baseline", "parameter_ratio_vs_baseline", "param_normalized_delta"):
            if key not in row:
                failures.append(f"COMPARISON_FIELD_MISSING:{variant}:{key}")

    if set(quant.get("rows", {})) != set(VARIANTS):
        failures.append("QUANT_VARIANTS_MISMATCH")
    for variant in VARIANTS:
        if set(quant.get("rows", {}).get(variant, {})) != {str(width) for width in widths}:
            failures.append(f"QUANT_WIDTHS_MISMATCH:{variant}")

    if repair.get("all_repair_runs_have_accept_reject_rollback") is not True:
        failures.append("REPAIR_ACCEPT_REJECT_ROLLBACK_AGGREGATE_BAD")
    if set(repair.get("rows", {})) != set(VARIANTS):
        failures.append("REPAIR_VARIANTS_MISMATCH")
    for variant in VARIANTS:
        if set(repair.get("rows", {}).get(variant, {})) != {str(width) for width in widths}:
            failures.append(f"REPAIR_WIDTHS_MISMATCH:{variant}")
        for width in widths:
            hist = load_json(out / f"e7a5_mutation_history_{variant}_width{width}.json")
            if hist.get("mutation_attempt_count", 0) < 1:
                failures.append(f"NO_REPAIR_MUTATION_ATTEMPTS:{variant}:{width}")
            if hist.get("accepted_mutation_count", 0) < 1:
                failures.append(f"NO_REPAIR_ACCEPTED_MUTATIONS:{variant}:{width}")
            if hist.get("rejected_mutation_count", 0) < 1:
                failures.append(f"NO_REPAIR_REJECTED_MUTATIONS:{variant}:{width}")
            if hist.get("rejected_mutation_count") != hist.get("rollback_count"):
                failures.append(f"REPAIR_ROLLBACK_MISMATCH:{variant}:{width}")
            if len(hist.get("history", [])) != manifest.get("settings", {}).get("repair_generations"):
                failures.append(f"REPAIR_GENERATION_COUNT_MISMATCH:{variant}:{width}")
            diff = load_json(out / f"e7a5_parameter_diff_{variant}_mutation_repair_width{width}.json")
            if "actual_parameter_diff_found" not in diff:
                failures.append(f"REPAIR_PARAMETER_DIFF_FIELD_MISSING:{variant}:{width}")

    if set(training.get("variants", {})) != set(VARIANTS):
        failures.append("TRAINING_HISTORY_VARIANTS_MISMATCH")
    for variant in VARIANTS:
        if set(training.get("variants", {}).get(variant, {})) != {str(width) for width in widths}:
            failures.append(f"TRAINING_HISTORY_WIDTHS_MISMATCH:{variant}")
        for width in widths:
            rows = training.get("variants", {}).get(variant, {}).get(str(width), [])
            if len(rows) != manifest.get("settings", {}).get("gradient_epochs"):
                failures.append(f"TRAINING_EPOCH_COUNT_MISMATCH:{variant}:{width}")
    if set(mutation.get("variants", {})) != set(VARIANTS):
        failures.append("MUTATION_HISTORY_VARIANTS_MISMATCH")

    if op_audit.get("incremental_one_freedom_at_a_time") is not True:
        failures.append("OPERATOR_INCREMENTAL_AUDIT_BAD")
    if op_audit.get("full_formula_combinatorics_used") is not False:
        failures.append("FORMULA_COMBINATORICS_USED")
    if set(op_audit.get("overfit_or_search_noise_flags", {})) != set(VARIANTS):
        failures.append("OVERFIT_FLAGS_VARIANTS_MISMATCH")

    if no_synth.get("generated_from_row_level_eval") is not True:
        failures.append("AUDIT_NOT_ROW_LEVEL")
    if no_synth.get("row_level_samples_present") is not True:
        failures.append("ROW_LEVEL_SAMPLE_AUDIT_BAD")
    if no_synth.get("backprop_used_for_float_variants") is not True:
        failures.append("BACKPROP_SCOPE_AUDIT_BAD")
    if no_synth.get("mutation_repair_used_optimizer_or_backprop") is not False:
        failures.append("MUTATION_REPAIR_BACKPROP_AUDIT_BAD")
    if no_synth.get("hardcoded_improvement_flags_present") is not False:
        failures.append("HARDCODED_IMPROVEMENT_FLAG_PRESENT")
    if no_synth.get("final_e7_verdict_intentionally_deferred") is not True:
        failures.append("AUDIT_FINAL_E7_NOT_DEFERRED")

    for split in ("heldout", "ood", "counterfactual", "adversarial", "stress"):
        sample = load_json(out / f"e7a5_row_level_eval_sample_{split}.json")
        if sample.get("split") != split:
            failures.append(f"ROW_SAMPLE_SPLIT_BAD:{split}")
        for variant in VARIANTS:
            if set(sample.get("samples", {}).get(variant, {})) != {str(width) for width in widths}:
                failures.append(f"ROW_SAMPLE_WIDTHS_MISMATCH:{split}:{variant}")
            for width in widths:
                cell = sample.get("samples", {}).get(variant, {}).get(str(width), {})
                if not cell.get("float") or not cell.get("quantized") or not cell.get("repair"):
                    failures.append(f"ROW_SAMPLE_EMPTY:{split}:{variant}:{width}")

    if replay.get("internal_replay_passed") is not True:
        failures.append("INTERNAL_REPLAY_FAILED")
    replay_root_raw = replay.get("replay_work_root")
    if not replay_root_raw:
        failures.append("REPLAY_WORK_ROOT_MISSING")
    else:
        replay_root = (REPO_ROOT / replay_root_raw).resolve()
        try:
            relative = replay_root.relative_to(out)
        except ValueError:
            relative = None
        if relative is None or not replay_root.exists():
            failures.append("REPLAY_WORK_ROOT_BAD")
        elif not (replay_root / "progress.jsonl").exists():
            failures.append("REPLAY_PROGRESS_MISSING")
        else:
            replay_events = {row.get("event") for row in progress_rows(replay_root)}
            for event in ("startup", "task_generated", "repair_generation"):
                if event not in replay_events:
                    failures.append(f"REPLAY_PROGRESS_EVENT_MISSING:{event}")
    if set(replay.get("hash_comparisons", {})) != set(HASH_ARTIFACTS):
        failures.append("REPLAY_HASH_ARTIFACT_SET_MISMATCH")
    for name, row in replay.get("hash_comparisons", {}).items():
        if row.get("match") is not True:
            failures.append(f"REPLAY_HASH_MISMATCH:{name}")
    if decision.get("deterministic_replay_passed") is not True:
        failures.append("DECISION_REPLAY_FLAG_FALSE")
    if decision.get("decision") not in VALID_DECISIONS:
        failures.append("INVALID_DECISION")
    if decision.get("final_e7_verdict_intentionally_deferred") is not True:
        failures.append("DECISION_FINAL_E7_NOT_DEFERRED")
    if summary.get("decision") != decision.get("decision"):
        failures.append("SUMMARY_DECISION_MISMATCH")
    if summary.get("deterministic_replay_passed") is not True:
        failures.append("SUMMARY_REPLAY_FLAG_FALSE")

    report_text = (out / "report.md").read_text(encoding="utf-8").lower()
    for forbidden in ("consciousness", "agi", "model-scale behavior", "natural-language reasoning proof"):
        if forbidden in report_text:
            failures.append(f"FORBIDDEN_REPORT_CLAIM:{forbidden}")

    rows = progress_rows(out)
    events = {row.get("event") for row in rows}
    for event in ("startup", "task_generated", "gradient_epoch", "repair_generation", "final_artifacts_written"):
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")
    if len(rows) < 5:
        failures.append("PROGRESS_TOO_SPARSE")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="target/pilot_wave/e7a5_operator_cell_matrix_incremental_scan")
    args = parser.parse_args()
    out = resolve_out(args.out)
    failures = ast_scan() + check_artifacts(out)
    report = {
        "schema_version": "e7a5_checker_report_v1",
        "out": out.relative_to(REPO_ROOT).as_posix(),
        "failure_count": len(failures),
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    write_path = out / "e7a5_checker_report.json"
    if out.exists():
        write_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

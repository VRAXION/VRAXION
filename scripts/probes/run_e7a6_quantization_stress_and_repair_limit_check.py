#!/usr/bin/env python3
"""Checker for E7A6 quantization stress and repair limit artifacts."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = "scripts/probes/run_e7a6_quantization_stress_and_repair_limit.py"
CHECKER = "scripts/probes/run_e7a6_quantization_stress_and_repair_limit_check.py"
QUANT_LEVELS = ("int8", "int4", "int3", "ternary", "binary")
SYSTEMS = ("float32_matrix_core", "quantized_no_repair", "quantized_mutation_repair", "random_control")
VALID_DECISIONS = {
    "e7a6_int8_only_stable",
    "e7a6_int4_stable_without_repair",
    "e7a6_int3_or_lower_stable_without_repair",
    "e7a6_mutation_repair_recovers_low_bit_core",
    "e7a6_mutation_repair_partial_low_bit_recovery",
    "e7a6_mutation_repair_not_useful",
    "e7a6_quantization_breakpoint_mapped",
    "e7a6_invalid_artifact_detected",
}
BASE_REQUIRED = (
    "e7a6_backend_manifest.json",
    "e7a6_task_generation_report.json",
    "e7a6_float_training_report.json",
    "e7a6_quantization_stress_report.json",
    "e7a6_mutation_repair_report.json",
    "e7a6_frontier_report.json",
    "e7a6_mutation_history.json",
    "e7a6_no_synthetic_metric_audit.json",
    "e7a6_deterministic_replay_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "progress.jsonl",
    "e7a6_row_level_eval_sample_heldout.json",
    "e7a6_row_level_eval_sample_ood.json",
    "e7a6_row_level_eval_sample_counterfactual.json",
    "e7a6_row_level_eval_sample_adversarial.json",
)
HASH_ARTIFACTS = (
    "e7a6_task_generation_report.json",
    "e7a6_float_training_report.json",
    "e7a6_quantization_stress_report.json",
    "e7a6_mutation_repair_report.json",
    "e7a6_frontier_report.json",
    "e7a6_no_synthetic_metric_audit.json",
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
    manifest = load_json(out / "e7a6_backend_manifest.json")
    widths = [int(width) for width in manifest.get("widths", [])]
    levels = tuple(manifest.get("quant_levels", []))
    names = list(BASE_REQUIRED)
    for width in widths:
        names.append(f"e7a6_candidate_float32_width{width}_summary.json")
        names.append(f"e7a6_training_history_float32_width{width}.json")
        names.append(f"e7a6_float_state_width{width}.json")
        names.append(f"e7a6_candidate_random_control_width{width}_summary.json")
        for level in levels:
            names.append(f"e7a6_candidate_{level}_width{width}.json")
            names.append(f"e7a6_candidate_{level}_no_repair_width{width}_summary.json")
            names.append(f"e7a6_mutation_history_{level}_width{width}.json")
            names.append(f"e7a6_candidate_{level}_mutation_repair_width{width}_initial.json")
            names.append(f"e7a6_candidate_{level}_mutation_repair_width{width}_final.json")
            names.append(f"e7a6_candidate_{level}_mutation_repair_width{width}_summary.json")
            names.append(f"e7a6_parameter_diff_{level}_mutation_repair_width{width}.json")
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
        *QUANT_LEVELS,
        "FloatMatrixCore",
        "quantize_state_dict",
        "run_quantized_repair",
        "mutation_repair_used_optimizer_or_backprop",
        "parallel_replay_supported",
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
        failures.append("ADAMW_MISSING_FOR_FLOAT32_BASELINE")
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
    if not (out / "e7a6_backend_manifest.json").exists():
        return ["MISSING_ARTIFACT:e7a6_backend_manifest.json"]
    for name in required_artifacts(out):
        if not (out / name).exists():
            failures.append(f"MISSING_ARTIFACT:{name}")
    if failures:
        return failures

    manifest = load_json(out / "e7a6_backend_manifest.json")
    task = load_json(out / "e7a6_task_generation_report.json")
    training = load_json(out / "e7a6_float_training_report.json")
    stress = load_json(out / "e7a6_quantization_stress_report.json")
    repair = load_json(out / "e7a6_mutation_repair_report.json")
    frontier = load_json(out / "e7a6_frontier_report.json")
    mutation = load_json(out / "e7a6_mutation_history.json")
    no_synth = load_json(out / "e7a6_no_synthetic_metric_audit.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    replay = load_json(out / "e7a6_deterministic_replay_report.json")
    decision = load_json(out / "decision.json")
    summary = load_json(out / "summary.json")
    widths = [int(width) for width in manifest.get("widths", [])]

    if tuple(manifest.get("systems", [])) != SYSTEMS:
        failures.append("MANIFEST_SYSTEMS_MISMATCH")
    if tuple(manifest.get("quant_levels", [])) != QUANT_LEVELS:
        failures.append("MANIFEST_QUANT_LEVELS_MISMATCH")
    if manifest.get("final_e7_verdict_intentionally_deferred") is not True:
        failures.append("MANIFEST_FINAL_E7_NOT_DEFERRED")
    if manifest.get("cpu_repair_and_gpu_gradient_overlap_supported") is not True:
        failures.append("OVERLAP_SUPPORT_FLAG_MISSING")
    if manifest.get("parallel_replay_supported") is not True:
        failures.append("PARALLEL_REPLAY_FLAG_MISSING")

    if task.get("inherits_task_from") != "E7A3_NEURAL_MATRIX_SUBSTRATE_HARNESS":
        failures.append("TASK_INHERITANCE_BAD")
    for split in ("train", "validation", "heldout", "ood", "counterfactual", "adversarial"):
        if task.get("splits", {}).get(split, {}).get("row_count", 0) < 1:
            failures.append(f"SPLIT_EMPTY:{split}")

    systems = aggregate.get("systems", {})
    if set(systems) != {"float32", "quantized", "repair", "random_control"}:
        failures.append("AGGREGATE_MODE_SET_MISMATCH")
    if set(systems.get("float32", {})) != {str(width) for width in widths}:
        failures.append("FLOAT_WIDTH_ROWS_MISMATCH")
    if set(systems.get("random_control", {})) != {str(width) for width in widths}:
        failures.append("RANDOM_WIDTH_ROWS_MISMATCH")
    for width in widths:
        row = systems.get("float32", {}).get(str(width), {})
        for key in REQUIRED_ROW_METRICS:
            if key not in row:
                failures.append(f"FLOAT_ROW_METRIC_MISSING:{width}:{key}")
    for mode in ("quantized", "repair"):
        if set(systems.get(mode, {})) != set(QUANT_LEVELS):
            failures.append(f"AGGREGATE_LEVELS_MISMATCH:{mode}")
        for level in QUANT_LEVELS:
            rows = systems.get(mode, {}).get(level, {})
            if set(rows) != {str(width) for width in widths}:
                failures.append(f"WIDTH_ROWS_MISMATCH:{mode}:{level}")
            for width in widths:
                row = rows.get(str(width), {})
                for key in REQUIRED_ROW_METRICS:
                    if key not in row:
                        failures.append(f"ROW_METRIC_MISSING:{mode}:{level}:{width}:{key}")
                if row.get("parameter_count", 0) < 1:
                    failures.append(f"PARAMETER_COUNT_MISSING:{mode}:{level}:{width}")
                if mode == "repair":
                    for key in ("mutation_attempt_count", "accepted_mutation_count", "rejected_mutation_count", "rollback_count"):
                        if key not in row:
                            failures.append(f"REPAIR_MUTATION_METRIC_MISSING:{level}:{width}:{key}")

    if training.get("system") != "float32_matrix_core":
        failures.append("TRAINING_REPORT_SYSTEM_BAD")
    if set(training.get("widths", {})) != {str(width) for width in widths}:
        failures.append("TRAINING_WIDTHS_MISMATCH")
    for width in widths:
        rows = training.get("widths", {}).get(str(width), {}).get("history", [])
        if len(rows) != manifest.get("settings", {}).get("gradient_epochs"):
            failures.append(f"TRAINING_EPOCH_COUNT_MISMATCH:{width}")

    if set(stress.get("rows", {})) != set(QUANT_LEVELS):
        failures.append("STRESS_LEVELS_MISMATCH")
    for level in QUANT_LEVELS:
        if set(stress.get("rows", {}).get(level, {})) != {str(width) for width in widths}:
            failures.append(f"STRESS_WIDTHS_MISMATCH:{level}")
        for width in widths:
            row = stress.get("rows", {}).get(level, {}).get(str(width), {})
            for key in ("float_eval_accuracy", "quantized_eval_accuracy", "repair_eval_accuracy", "quantization_drop", "repair_drop", "repair_delta_vs_quantized", "stable_without_repair", "stable_after_repair"):
                if key not in row:
                    failures.append(f"STRESS_FIELD_MISSING:{level}:{width}:{key}")

    if repair.get("all_repair_runs_have_accept_reject_rollback") is not True:
        failures.append("REPAIR_ACCEPT_REJECT_ROLLBACK_AGGREGATE_BAD")
    if set(repair.get("rows", {})) != set(QUANT_LEVELS):
        failures.append("REPAIR_LEVELS_MISMATCH")
    for level in QUANT_LEVELS:
        if set(repair.get("rows", {}).get(level, {})) != {str(width) for width in widths}:
            failures.append(f"REPAIR_WIDTHS_MISMATCH:{level}")
        for width in widths:
            hist = load_json(out / f"e7a6_mutation_history_{level}_width{width}.json")
            if hist.get("mutation_attempt_count", 0) < 1:
                failures.append(f"NO_REPAIR_MUTATION_ATTEMPTS:{level}:{width}")
            if hist.get("accepted_mutation_count", 0) < 1:
                failures.append(f"NO_REPAIR_ACCEPTED_MUTATIONS:{level}:{width}")
            if hist.get("rejected_mutation_count", 0) < 1:
                failures.append(f"NO_REPAIR_REJECTED_MUTATIONS:{level}:{width}")
            if hist.get("rejected_mutation_count") != hist.get("rollback_count"):
                failures.append(f"REPAIR_ROLLBACK_MISMATCH:{level}:{width}")
            if len(hist.get("history", [])) != manifest.get("settings", {}).get("repair_generations"):
                failures.append(f"REPAIR_GENERATION_COUNT_MISMATCH:{level}:{width}")
            diff = load_json(out / f"e7a6_parameter_diff_{level}_mutation_repair_width{width}.json")
            if "actual_parameter_diff_found" not in diff:
                failures.append(f"REPAIR_PARAMETER_DIFF_FIELD_MISSING:{level}:{width}")

    if set(mutation.get("levels", {})) != set(QUANT_LEVELS):
        failures.append("MUTATION_HISTORY_LEVELS_MISMATCH")
    if set(frontier.get("best_level_rows", {})) != set(QUANT_LEVELS):
        failures.append("FRONTIER_LEVELS_MISMATCH")

    if no_synth.get("generated_from_row_level_eval") is not True:
        failures.append("AUDIT_NOT_ROW_LEVEL")
    if no_synth.get("row_level_samples_present") is not True:
        failures.append("ROW_LEVEL_SAMPLE_AUDIT_BAD")
    if no_synth.get("backprop_used_for_float32_matrix_core") is not True:
        failures.append("BACKPROP_SCOPE_AUDIT_BAD")
    if no_synth.get("mutation_repair_used_optimizer_or_backprop") is not False:
        failures.append("MUTATION_REPAIR_BACKPROP_AUDIT_BAD")
    if no_synth.get("hardcoded_improvement_flags_present") is not False:
        failures.append("HARDCODED_IMPROVEMENT_FLAG_PRESENT")
    if no_synth.get("final_e7_verdict_intentionally_deferred") is not True:
        failures.append("AUDIT_FINAL_E7_NOT_DEFERRED")

    for split in ("heldout", "ood", "counterfactual", "adversarial"):
        sample = load_json(out / f"e7a6_row_level_eval_sample_{split}.json")
        if sample.get("split") != split:
            failures.append(f"ROW_SAMPLE_SPLIT_BAD:{split}")
        if set(sample.get("samples", {})) != {str(width) for width in widths}:
            failures.append(f"ROW_SAMPLE_WIDTHS_MISMATCH:{split}")
        for width in widths:
            cell = sample.get("samples", {}).get(str(width), {})
            if not cell.get("float32"):
                failures.append(f"ROW_SAMPLE_FLOAT_EMPTY:{split}:{width}")
            if set(cell.get("quantized", {})) != set(QUANT_LEVELS):
                failures.append(f"ROW_SAMPLE_QUANT_LEVELS_BAD:{split}:{width}")
            if set(cell.get("repair", {})) != set(QUANT_LEVELS):
                failures.append(f"ROW_SAMPLE_REPAIR_LEVELS_BAD:{split}:{width}")
            for level in QUANT_LEVELS:
                if not cell.get("quantized", {}).get(level):
                    failures.append(f"ROW_SAMPLE_QUANT_EMPTY:{split}:{width}:{level}")
                if not cell.get("repair", {}).get(level):
                    failures.append(f"ROW_SAMPLE_REPAIR_EMPTY:{split}:{width}:{level}")

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

    events = {row.get("event") for row in progress_rows(out)}
    for event in ("startup", "task_generated", "gradient_epoch", "repair_generation", "deterministic_replay_start", "deterministic_replay_complete", "final_artifacts_written"):
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="target/pilot_wave/e7a6_quantization_stress_and_repair_limit")
    args = parser.parse_args()
    out = resolve_out(args.out)
    failures = ast_scan() + check_artifacts(out)
    report = {
        "schema_version": "e7a6_checker_report_v1",
        "out": out.relative_to(REPO_ROOT).as_posix(),
        "failure_count": len(failures),
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if out.exists():
        (out / "e7a6_checker_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

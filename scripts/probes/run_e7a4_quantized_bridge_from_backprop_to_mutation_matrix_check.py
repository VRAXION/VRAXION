#!/usr/bin/env python3
"""Checker for E7A4 quantized bridge artifacts."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = "scripts/probes/run_e7a4_quantized_bridge_from_backprop_to_mutation_matrix.py"
CHECKER = "scripts/probes/run_e7a4_quantized_bridge_from_backprop_to_mutation_matrix_check.py"
SYSTEMS = (
    "float_mlp_backprop_reference",
    "float_matrix_core_backprop",
    "quantized_matrix_core_no_repair",
    "quantized_matrix_core_mutation_repair",
    "random_control",
)
VALID_DECISIONS = {
    "e7a4_reference_not_solved_redesign_required",
    "e7a4_float_matrix_core_not_learned",
    "e7a4_quantized_matrix_core_preserved_without_repair",
    "e7a4_mutation_repair_improves_quantized_matrix_core",
    "e7a4_mutation_repair_recovers_quantized_matrix_core",
    "e7a4_quantization_breaks_and_mutation_repair_failed",
    "e7a4_task_too_easy_or_leaky",
}
BASE_REQUIRED = (
    "e7a4_backend_manifest.json",
    "e7a4_task_generation_report.json",
    "e7a4_bridge_report.json",
    "e7a4_quantization_report.json",
    "e7a4_mutation_repair_report.json",
    "e7a4_training_history.json",
    "e7a4_mutation_history.json",
    "e7a4_no_synthetic_metric_audit.json",
    "e7a4_deterministic_replay_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "progress.jsonl",
    "e7a4_row_level_eval_sample_heldout.json",
    "e7a4_row_level_eval_sample_ood.json",
    "e7a4_row_level_eval_sample_counterfactual.json",
    "e7a4_row_level_eval_sample_adversarial.json",
)
HASH_ARTIFACTS = (
    "e7a4_task_generation_report.json",
    "e7a4_bridge_report.json",
    "e7a4_quantization_report.json",
    "e7a4_mutation_repair_report.json",
    "e7a4_no_synthetic_metric_audit.json",
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
    manifest = load_json(out / "e7a4_backend_manifest.json")
    widths = [int(width) for width in manifest.get("widths", [])]
    names = list(BASE_REQUIRED)
    for width in widths:
        for system in SYSTEMS:
            names.append(f"e7a4_candidate_{system}_width{width}_summary.json")
        names.append(f"e7a4_training_history_float_matrix_core_backprop_width{width}.json")
        names.append(f"e7a4_float_matrix_core_state_width{width}.json")
        names.append(f"e7a4_candidate_quantized_matrix_core_no_repair_width{width}.json")
        names.append(f"e7a4_mutation_history_quantized_matrix_core_mutation_repair_width{width}.json")
        names.append(f"e7a4_candidate_quantized_matrix_core_mutation_repair_width{width}_initial.json")
        names.append(f"e7a4_candidate_quantized_matrix_core_mutation_repair_width{width}_final.json")
        names.append(f"e7a4_parameter_diff_quantized_matrix_core_mutation_repair_width{width}.json")
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
    for token in (*SYSTEMS, "FloatMatrixCore", "quantize_state_dict", "run_quantized_repair"):
        if token not in text:
            failures.append(f"TOKEN_MISSING_IN_RUNNER:{token}")
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
        failures.append("ADAMW_MISSING_FOR_FLOAT_SYSTEMS")
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
    if not (out / "e7a4_backend_manifest.json").exists():
        return ["MISSING_ARTIFACT:e7a4_backend_manifest.json"]
    for name in required_artifacts(out):
        if not (out / name).exists():
            failures.append(f"MISSING_ARTIFACT:{name}")
    if failures:
        return failures

    manifest = load_json(out / "e7a4_backend_manifest.json")
    task = load_json(out / "e7a4_task_generation_report.json")
    bridge = load_json(out / "e7a4_bridge_report.json")
    quant = load_json(out / "e7a4_quantization_report.json")
    repair = load_json(out / "e7a4_mutation_repair_report.json")
    training = load_json(out / "e7a4_training_history.json")
    mutation = load_json(out / "e7a4_mutation_history.json")
    audit = load_json(out / "e7a4_no_synthetic_metric_audit.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    replay = load_json(out / "e7a4_deterministic_replay_report.json")
    decision = load_json(out / "decision.json")
    summary = load_json(out / "summary.json")
    widths = [int(width) for width in manifest.get("widths", [])]

    if tuple(manifest.get("systems", [])) != SYSTEMS:
        failures.append("MANIFEST_SYSTEMS_MISMATCH")
    if manifest.get("final_e7_verdict_intentionally_deferred") is not True:
        failures.append("MANIFEST_FINAL_E7_NOT_DEFERRED")
    if manifest.get("cpu_repair_and_gpu_gradient_overlap_supported") is not True:
        failures.append("OVERLAP_SUPPORT_FLAG_MISSING")

    if task.get("inherits_task_from") != "E7A3_NEURAL_MATRIX_SUBSTRATE_HARNESS":
        failures.append("TASK_INHERITANCE_BAD")
    for split in ("train", "validation", "heldout", "ood", "counterfactual", "adversarial"):
        if task.get("splits", {}).get(split, {}).get("row_count", 0) < 1:
            failures.append(f"SPLIT_EMPTY:{split}")

    if set(aggregate.get("systems", {})) != set(SYSTEMS):
        failures.append("AGGREGATE_SYSTEMS_MISMATCH")
    for system in SYSTEMS:
        rows = aggregate.get("systems", {}).get(system, {})
        if set(rows) != {str(width) for width in widths}:
            failures.append(f"WIDTH_ROWS_MISMATCH:{system}")
        for width in widths:
            row = rows.get(str(width), {})
            for key in REQUIRED_ROW_METRICS:
                if key not in row:
                    failures.append(f"ROW_METRIC_MISSING:{system}:{width}:{key}")
            if system != "random_control" and row.get("parameter_count", 0) < 1:
                failures.append(f"PARAMETER_COUNT_MISSING:{system}:{width}")

    if set(bridge.get("systems", [])) != set(SYSTEMS):
        failures.append("BRIDGE_SYSTEMS_MISMATCH")
    if set(bridge.get("smallest_passing_width", {})) != set(SYSTEMS):
        failures.append("BRIDGE_PASSING_WIDTH_MISMATCH")
    if set(quant.get("rows", {})) != {str(width) for width in widths}:
        failures.append("QUANTIZATION_WIDTHS_MISMATCH")

    if repair.get("all_repair_runs_have_accept_reject_rollback") is not True:
        failures.append("REPAIR_ACCEPT_REJECT_ROLLBACK_AGGREGATE_BAD")
    if set(repair.get("rows", {})) != {str(width) for width in widths}:
        failures.append("REPAIR_WIDTHS_MISMATCH")
    for width in widths:
        hist = load_json(out / f"e7a4_mutation_history_quantized_matrix_core_mutation_repair_width{width}.json")
        if hist.get("mutation_attempt_count", 0) < 1:
            failures.append(f"NO_REPAIR_MUTATION_ATTEMPTS:{width}")
        if hist.get("accepted_mutation_count", 0) < 1:
            failures.append(f"NO_REPAIR_ACCEPTED_MUTATIONS:{width}")
        if hist.get("rejected_mutation_count", 0) < 1:
            failures.append(f"NO_REPAIR_REJECTED_MUTATIONS:{width}")
        if hist.get("rejected_mutation_count") != hist.get("rollback_count"):
            failures.append(f"REPAIR_ROLLBACK_MISMATCH:{width}")
        if len(hist.get("history", [])) != manifest.get("settings", {}).get("repair_generations"):
            failures.append(f"REPAIR_GENERATION_COUNT_MISMATCH:{width}")
        diff = load_json(out / f"e7a4_parameter_diff_quantized_matrix_core_mutation_repair_width{width}.json")
        if "actual_parameter_diff_found" not in diff:
            failures.append(f"REPAIR_PARAMETER_DIFF_FIELD_MISSING:{width}")

    for system in ("float_mlp_backprop_reference", "float_matrix_core_backprop"):
        if set(training.get("systems", {}).get(system, {})) != {str(width) for width in widths}:
            failures.append(f"TRAINING_HISTORY_WIDTHS_MISMATCH:{system}")
        for width in widths:
            rows = training.get("systems", {}).get(system, {}).get(str(width), [])
            if len(rows) != manifest.get("settings", {}).get("gradient_epochs"):
                failures.append(f"TRAINING_EPOCH_COUNT_MISMATCH:{system}:{width}")
    if set(mutation.get("systems", {}).get("quantized_matrix_core_mutation_repair", {})) != {str(width) for width in widths}:
        failures.append("MUTATION_HISTORY_WIDTHS_MISMATCH")

    if audit.get("generated_from_row_level_eval") is not True:
        failures.append("AUDIT_NOT_ROW_LEVEL")
    if audit.get("row_level_samples_present") is not True:
        failures.append("ROW_LEVEL_SAMPLE_AUDIT_BAD")
    if audit.get("backprop_used_for_float_systems") is not True:
        failures.append("BACKPROP_SCOPE_AUDIT_BAD")
    if audit.get("mutation_repair_used_optimizer_or_backprop") is not False:
        failures.append("MUTATION_REPAIR_BACKPROP_AUDIT_BAD")
    if audit.get("hardcoded_improvement_flags_present") is not False:
        failures.append("HARDCODED_IMPROVEMENT_FLAG_PRESENT")
    if audit.get("final_e7_verdict_intentionally_deferred") is not True:
        failures.append("AUDIT_FINAL_E7_NOT_DEFERRED")

    if replay.get("internal_replay_passed") is not True:
        failures.append("INTERNAL_REPLAY_FAILED")
    if decision.get("deterministic_replay_passed") is not True:
        failures.append("DECISION_REPLAY_FLAG_BAD")
    for name, row in replay.get("hash_comparisons", {}).items():
        if name in HASH_ARTIFACTS and row.get("match") is not True:
            failures.append(f"REPLAY_HASH_MISMATCH:{name}")

    if decision.get("decision") not in VALID_DECISIONS:
        failures.append("DECISION_NOT_VALID")
    if decision.get("final_e7_verdict_intentionally_deferred") is not True:
        failures.append("DECISION_FINAL_E7_NOT_DEFERRED")
    if summary.get("final_e7_verdict_intentionally_deferred") is not True:
        failures.append("SUMMARY_FINAL_E7_NOT_DEFERRED")

    events = {row.get("event") for row in progress_rows(out)}
    for event in ("startup", "task_generated", "reference_complete", "matrix_core_gradient_epoch", "repair_generation", "final_artifacts_written"):
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")
    if manifest.get("settings", {}).get("execution_mode") == "parallel" and "repair_lane_ready" not in events:
        failures.append("PARALLEL_REPAIR_PROGRESS_MISSING")

    for split in ("heldout", "ood", "counterfactual", "adversarial"):
        sample = load_json(out / f"e7a4_row_level_eval_sample_{split}.json")
        if set(sample.get("samples", {})) != set(SYSTEMS):
            failures.append(f"ROW_SAMPLE_SYSTEMS_MISMATCH:{split}")

    report_text = (out / "report.md").read_text(encoding="utf-8").lower()
    for forbidden in ("final e7 architecture confirmed", "agi confirmed", "consciousness confirmed", "model-scale proof"):
        if forbidden in report_text:
            failures.append(f"FORBIDDEN_REPORT_CLAIM:{forbidden}")

    failures.extend(ast_scan())
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = resolve_out(args.out)
    failures = check_artifacts(out)
    print(json.dumps({"failure_count": len(failures), "failures": failures, "out": out.as_posix()}, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

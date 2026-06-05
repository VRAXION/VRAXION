#!/usr/bin/env python3
"""Checker for E7A7 low-bit repair operator audit artifacts."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = "scripts/probes/run_e7a7_low_bit_repair_operator_audit.py"
CHECKER = "scripts/probes/run_e7a7_low_bit_repair_operator_audit_check.py"
AUDIT_LEVELS = ("int3", "ternary", "binary")
LOW_BIT_LEVELS = ("ternary", "binary")
BLOCKS = ("input_projection", "recurrent_state", "carry_gate", "state_bias", "output_head")
VALID_DECISIONS = {
    "e7a7_sensitive_block_repair_sufficient",
    "e7a7_output_or_state_bottleneck_identified",
    "e7a7_qat_preferred_over_post_repair",
    "e7a7_repair_operator_bottleneck_detected",
    "e7a7_low_bit_information_limit_detected",
    "e7a7_low_bit_breakpoint_audit_complete",
    "e7a7_invalid_artifact_detected",
}
BASE_REQUIRED = (
    "e7a7_backend_manifest.json",
    "e7a7_task_generation_report.json",
    "e7a7_block_damage_report.json",
    "e7a7_block_restore_report.json",
    "e7a7_repair_operator_report.json",
    "e7a7_qat_report.json",
    "e7a7_low_bit_bottleneck_report.json",
    "e7a7_mutation_history.json",
    "e7a7_no_synthetic_metric_audit.json",
    "e7a7_deterministic_replay_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "progress.jsonl",
    "e7a7_row_level_eval_sample_heldout.json",
    "e7a7_row_level_eval_sample_ood.json",
    "e7a7_row_level_eval_sample_counterfactual.json",
    "e7a7_row_level_eval_sample_adversarial.json",
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
        "run_targeted_repair",
        "mutate_quantized_targeted",
        "fake_quant_tensor",
        "train_qat_core",
        "block_restored_to_int8",
        "block_only_low_bit",
        "ProcessPoolExecutor",
        "deterministic_replay",
    ):
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
        failures.append("ADAMW_MISSING_FOR_QAT_CONTROL")
    forbidden_calls = {"backward", "step", "zero_grad"}
    mutation_function_names = {
        "run_targeted_repair",
        "mutate_quantized_targeted",
        "score_candidate",
        "quantized_eval_result",
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

    manifest = load_json(out / "e7a7_backend_manifest.json")
    task = load_json(out / "e7a7_task_generation_report.json")
    damage = load_json(out / "e7a7_block_damage_report.json")
    restore = load_json(out / "e7a7_block_restore_report.json")
    repair = load_json(out / "e7a7_repair_operator_report.json")
    qat = load_json(out / "e7a7_qat_report.json")
    bottleneck = load_json(out / "e7a7_low_bit_bottleneck_report.json")
    mutation = load_json(out / "e7a7_mutation_history.json")
    no_synth = load_json(out / "e7a7_no_synthetic_metric_audit.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    replay = load_json(out / "e7a7_deterministic_replay_report.json")
    decision = load_json(out / "decision.json")
    summary = load_json(out / "summary.json")
    widths = [int(width) for width in manifest.get("widths", [])]
    width_keys = {str(width) for width in widths}

    if tuple(manifest.get("audit_levels", [])) != AUDIT_LEVELS:
        failures.append("MANIFEST_AUDIT_LEVELS_MISMATCH")
    if tuple(manifest.get("low_bit_levels", [])) != LOW_BIT_LEVELS:
        failures.append("MANIFEST_LOW_BIT_LEVELS_MISMATCH")
    if set(manifest.get("blocks", {})) != set(BLOCKS):
        failures.append("MANIFEST_BLOCKS_MISMATCH")
    if manifest.get("cpu_repair_and_gradient_overlap_supported") is not True:
        failures.append("OVERLAP_SUPPORT_FLAG_MISSING")
    if manifest.get("parallel_replay_supported") is not True:
        failures.append("PARALLEL_REPLAY_FLAG_MISSING")
    if manifest.get("broad_claims_intentionally_deferred") is not True:
        failures.append("BROAD_CLAIMS_DEFER_FLAG_MISSING")

    if task.get("inherits_task_from") != "E7A6_QUANTIZATION_STRESS_AND_REPAIR_LIMIT":
        failures.append("TASK_INHERITANCE_BAD")
    for split in ("train", "validation", "heldout", "ood", "counterfactual", "adversarial"):
        if task.get("splits", {}).get(split, {}).get("row_count", 0) < 1:
            failures.append(f"SPLIT_EMPTY:{split}")

    systems = aggregate.get("systems", {})
    for section in ("float32", "low_bit", "block_only", "block_restore", "repair", "qat"):
        if section not in systems:
            failures.append(f"AGGREGATE_SECTION_MISSING:{section}")
    if set(systems.get("float32", {})) != width_keys:
        failures.append("FLOAT_WIDTH_ROWS_MISMATCH")
    for width in widths:
        row = systems.get("float32", {}).get(str(width), {})
        for key in REQUIRED_ROW_METRICS:
            if key not in row:
                failures.append(f"FLOAT_ROW_METRIC_MISSING:{width}:{key}")
    for level in AUDIT_LEVELS:
        if set(systems.get("low_bit", {}).get(level, {})) != width_keys:
            failures.append(f"LOW_BIT_WIDTHS_MISMATCH:{level}")
        if set(systems.get("block_only", {}).get(level, {})) != width_keys:
            failures.append(f"BLOCK_ONLY_WIDTHS_MISMATCH:{level}")
        if set(systems.get("block_restore", {}).get(level, {})) != width_keys:
            failures.append(f"BLOCK_RESTORE_WIDTHS_MISMATCH:{level}")
        if set(systems.get("repair", {}).get(level, {})) != width_keys:
            failures.append(f"REPAIR_WIDTHS_MISMATCH:{level}")
        for width in widths:
            for row_name, row in (("low_bit", systems.get("low_bit", {}).get(level, {}).get(str(width), {})),):
                for key in REQUIRED_ROW_METRICS:
                    if key not in row:
                        failures.append(f"ROW_METRIC_MISSING:{row_name}:{level}:{width}:{key}")
            if set(systems.get("block_only", {}).get(level, {}).get(str(width), {})) != set(BLOCKS):
                failures.append(f"BLOCK_ONLY_BLOCKS_MISMATCH:{level}:{width}")
            if set(systems.get("block_restore", {}).get(level, {}).get(str(width), {})) != set(BLOCKS):
                failures.append(f"BLOCK_RESTORE_BLOCKS_MISMATCH:{level}:{width}")
            repair_rows = systems.get("repair", {}).get(level, {}).get(str(width), {})
            expected_repair_count = 1 + len(BLOCKS) + 1
            if len(repair_rows) != expected_repair_count:
                failures.append(f"REPAIR_ROW_COUNT_MISMATCH:{level}:{width}:{len(repair_rows)}")
            if "full" not in repair_rows:
                failures.append(f"FULL_REPAIR_ROW_MISSING:{level}:{width}")
    for level in LOW_BIT_LEVELS:
        if set(systems.get("qat", {}).get(level, {})) != width_keys:
            failures.append(f"QAT_WIDTHS_MISMATCH:{level}")
        for width in widths:
            row = systems.get("qat", {}).get(level, {}).get(str(width), {})
            for key in REQUIRED_ROW_METRICS:
                if key not in row:
                    failures.append(f"QAT_ROW_METRIC_MISSING:{level}:{width}:{key}")

    for report_name, report in (("damage", damage), ("restore", restore), ("bottleneck", bottleneck)):
        if set(report.get("rows", {})) != set(AUDIT_LEVELS):
            failures.append(f"{report_name.upper()}_LEVELS_MISMATCH")
        for level in AUDIT_LEVELS:
            if set(report.get("rows", {}).get(level, {})) != width_keys:
                failures.append(f"{report_name.upper()}_WIDTHS_MISMATCH:{level}")
    for level in AUDIT_LEVELS:
        for width in widths:
            if set(damage.get("rows", {}).get(level, {}).get(str(width), {})) != set(BLOCKS):
                failures.append(f"DAMAGE_BLOCKS_MISMATCH:{level}:{width}")
            if set(restore.get("rows", {}).get(level, {}).get(str(width), {})) != set(BLOCKS):
                failures.append(f"RESTORE_BLOCKS_MISMATCH:{level}:{width}")
            brow = bottleneck.get("rows", {}).get(level, {}).get(str(width), {})
            for key in ("top_restore_block", "top_restore_gain", "full_repair_eval_accuracy", "best_targeted_eval_accuracy", "best_pair_eval_accuracy"):
                if key not in brow:
                    failures.append(f"BOTTLENECK_FIELD_MISSING:{level}:{width}:{key}")

    if repair.get("all_repair_runs_have_mutation_attempts") is not True:
        failures.append("REPAIR_ATTEMPT_FLAG_BAD")
    if repair.get("all_repair_runs_have_reject_rollback") is not True:
        failures.append("REPAIR_ROLLBACK_FLAG_BAD")
    if repair.get("at_least_one_repair_mutation_accepted") is not True:
        failures.append("NO_REPAIR_ACCEPTED_MUTATION")
    if len(mutation.get("rows", {})) != len(widths) * len(AUDIT_LEVELS) * (len(BLOCKS) + 2):
        failures.append("MUTATION_HISTORY_RUN_COUNT_MISMATCH")
    for name, hist in mutation.get("rows", {}).items():
        if hist.get("mutation_attempt_count", 0) < 1:
            failures.append(f"NO_MUTATION_ATTEMPTS:{name}")
        if hist.get("rejected_mutation_count", 0) < 1:
            failures.append(f"NO_REJECTED_MUTATIONS:{name}")
        if hist.get("rejected_mutation_count") != hist.get("rollback_count"):
            failures.append(f"ROLLBACK_MISMATCH:{name}")
        if len(hist.get("history", [])) != manifest.get("settings", {}).get("repair_generations"):
            failures.append(f"REPAIR_GENERATION_COUNT_MISMATCH:{name}")

    if set(qat.get("rows", {})) != set(LOW_BIT_LEVELS):
        failures.append("QAT_LEVELS_MISMATCH")
    for level in LOW_BIT_LEVELS:
        if set(qat.get("rows", {}).get(level, {})) != width_keys:
            failures.append(f"QAT_REPORT_WIDTHS_MISMATCH:{level}")

    if no_synth.get("generated_from_row_level_eval") is not True:
        failures.append("NO_SYNTH_ROW_EVAL_FLAG_BAD")
    if no_synth.get("row_level_samples_present") is not True:
        failures.append("ROW_LEVEL_SAMPLE_FLAG_BAD")
    if no_synth.get("mutation_repair_used_optimizer_or_backprop") is not False:
        failures.append("MUTATION_REPAIR_BACKPROP_FLAG_BAD")
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

    progress = progress_rows(out)
    events = {row.get("event") for row in progress}
    for event in ("startup", "task_generated", "repair_generation", "qat_epoch", "deterministic_replay_complete", "final_artifacts_written"):
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")
    if len(progress) < 10:
        failures.append("PROGRESS_TOO_SPARSE")

    report_text = (out / "report.md").read_text(encoding="utf-8").lower()
    for forbidden in ("consciousness", "sentience", "agi", "model-scale"):
        if forbidden in report_text:
            failures.append(f"FORBIDDEN_REPORT_CLAIM_TOKEN:{forbidden}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="target/pilot_wave/e7a7_low_bit_repair_operator_audit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = resolve_out(args.out)
    failures = ast_scan() + check_artifacts(out)
    report = {
        "schema_version": "e7a7_checker_report_v1",
        "out": out.relative_to(REPO_ROOT).as_posix(),
        "failure_count": len(failures),
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Checker for E7A11 binary minimal-scale falsification sweep artifacts."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = "scripts/probes/run_e7a11_binary_minimal_scale_falsification_sweep.py"
CHECKER = "scripts/probes/run_e7a11_binary_minimal_scale_falsification_sweep_check.py"
METHODS = (
    "float32_matrix_core",
    "int4_direct",
    "int3_direct",
    "ternary_block_scale_qat",
    "binary_direct_block_scale",
    "binary_minimal_scale_qat",
)
VALID_DECISIONS = {
    "e7a11_binary_minimal_survives_falsification",
    "e7a11_binary_minimal_partially_survives",
    "e7a11_binary_minimal_seed_or_task_artifact_detected",
    "e7a11_int4_restored_preference",
    "e7a11_task_family_redesign_required",
    "e7a11_invalid_artifact_detected",
}
BASE_REQUIRED = (
    "e7a11_backend_manifest.json",
    "e7a11_task_family_report.json",
    "e7a11_case_results.json",
    "e7a11_bit_budget_falsification_report.json",
    "e7a11_width_scaling_report.json",
    "e7a11_no_synthetic_metric_audit.json",
    "e7a11_runtime_report.json",
    "e7a11_deterministic_replay_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "progress.jsonl",
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
        "configure_task_family",
        "binary_minimal_scale_qat",
        "budget_comparisons",
        "reference_budget_bits",
        "ProcessPoolExecutor",
        "deterministic_replay",
        "input_dim",
        "class_count",
    ):
        if token not in text:
            failures.append(f"TOKEN_MISSING_IN_RUNNER:{token}")
    return failures


def check_row(failures: list[str], method: str, case_id: str, width: str, payload: dict[str, Any]) -> None:
    row = payload.get("row", {})
    for key in REQUIRED_ROW_METRICS:
        if key not in row:
            failures.append(f"ROW_METRIC_MISSING:{case_id}:{method}:{width}:{key}")
    bit = row.get("bit_cost", {})
    for key in ("total_bit_cost", "compression_vs_float32", "nominal_weight_bits"):
        if key not in bit:
            failures.append(f"BIT_COST_FIELD_MISSING:{case_id}:{method}:{width}:{key}")
    samples = payload.get("row_level_samples", {})
    for split in ("heldout", "ood", "counterfactual", "adversarial"):
        if not samples.get(split):
            failures.append(f"ROW_SAMPLE_MISSING:{case_id}:{method}:{width}:{split}")


def check_artifacts(out: Path) -> list[str]:
    failures: list[str] = []
    for name in BASE_REQUIRED:
        if not (out / name).exists():
            failures.append(f"MISSING_ARTIFACT:{name}")
    if failures:
        return failures

    manifest = load_json(out / "e7a11_backend_manifest.json")
    task = load_json(out / "e7a11_task_family_report.json")
    cases = load_json(out / "e7a11_case_results.json")
    budget = load_json(out / "e7a11_bit_budget_falsification_report.json")
    width = load_json(out / "e7a11_width_scaling_report.json")
    no_synth = load_json(out / "e7a11_no_synthetic_metric_audit.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    replay = load_json(out / "e7a11_deterministic_replay_report.json")
    decision = load_json(out / "decision.json")
    summary = load_json(out / "summary.json")

    if tuple(manifest.get("methods", [])) != METHODS:
        failures.append("MANIFEST_METHODS_MISMATCH")
    if manifest.get("parallel_replay_supported") is not True:
        failures.append("PARALLEL_REPLAY_FLAG_MISSING")
    if manifest.get("broad_claims_intentionally_deferred") is not True:
        failures.append("BROAD_CLAIMS_DEFER_FLAG_MISSING")
    for width_key in ("float_widths", "int4_widths", "ternary_widths", "binary_widths"):
        if not manifest.get(width_key):
            failures.append(f"MANIFEST_WIDTHS_EMPTY:{width_key}")

    if task.get("inherits_matrix_core_from") != "E7A10_BINARY_SCALE_OVERHEAD_AND_BIT_BUDGET_AUDIT":
        failures.append("TASK_INHERITANCE_BAD")
    if task.get("case_count", 0) < 1:
        failures.append("NO_CASES_REPORTED")
    for case in task.get("cases", []):
        if int(case.get("input_dim", 0)) < 10:
            failures.append(f"INPUT_DIM_TOO_SMALL:{case.get('case_id')}")
        if int(case.get("class_count", 0)) < 2:
            failures.append(f"CLASS_COUNT_TOO_SMALL:{case.get('case_id')}")

    case_rows = cases.get("cases", {})
    if len(case_rows) != manifest.get("case_count"):
        failures.append("CASE_COUNT_MISMATCH")
    float_widths = {str(width) for width in manifest.get("float_widths", [])}
    int4_widths = {str(width) for width in manifest.get("int4_widths", [])}
    ternary_widths = {str(width) for width in manifest.get("ternary_widths", [])}
    binary_widths = {str(width) for width in manifest.get("binary_widths", [])}
    expected_widths = {
        "float32_matrix_core": float_widths,
        "int4_direct": int4_widths,
        "int3_direct": int4_widths,
        "ternary_block_scale_qat": ternary_widths,
        "binary_direct_block_scale": binary_widths,
        "binary_minimal_scale_qat": binary_widths,
    }
    for case_id, case_result in case_rows.items():
        methods = case_result.get("methods", {})
        if set(methods) != set(METHODS):
            failures.append(f"CASE_METHOD_SET_MISMATCH:{case_id}")
        for method, widths in expected_widths.items():
            if set(methods.get(method, {})) != widths:
                failures.append(f"CASE_METHOD_WIDTHS_MISMATCH:{case_id}:{method}")
            for width_value, payload in methods.get(method, {}).items():
                check_row(failures, method, case_id, width_value, payload)
        comp = case_result.get("budget_comparisons", {}).get("by_int4_reference_width", {})
        if set(comp) != int4_widths:
            failures.append(f"BUDGET_REFERENCE_WIDTHS_MISMATCH:{case_id}")
        for ref_width, row in comp.items():
            if row.get("reference_budget_bits", 0) <= 0:
                failures.append(f"REFERENCE_BUDGET_MISSING:{case_id}:{ref_width}")
            best = row.get("best_binary_minimal_same_budget")
            if not best:
                failures.append(f"BEST_BINARY_MISSING:{case_id}:{ref_width}")
            elif best.get("bit_cost", {}).get("total_bit_cost", 10**18) > row.get("reference_budget_bits", -1):
                failures.append(f"BEST_BINARY_EXCEEDS_BUDGET:{case_id}:{ref_width}")

    if set(budget.get("case_comparisons", {})) != set(case_rows):
        failures.append("BUDGET_CASES_MISMATCH")
    if set(width.get("rows", {})) != set(case_rows):
        failures.append("WIDTH_CASES_MISMATCH")
    if aggregate.get("case_count") != len(case_rows):
        failures.append("AGGREGATE_CASE_COUNT_MISMATCH")
    for key in ("positive_case_count", "falsified_case_count", "median_reference32_margin", "by_case"):
        if key not in aggregate:
            failures.append(f"AGGREGATE_FIELD_MISSING:{key}")

    if no_synth.get("generated_from_row_level_eval") is not True:
        failures.append("NO_SYNTH_ROW_EVAL_FLAG_BAD")
    if no_synth.get("row_level_samples_present") is not True:
        failures.append("ROW_LEVEL_SAMPLE_FLAG_BAD")
    if no_synth.get("task_family_patch_used") is not True:
        failures.append("TASK_FAMILY_PATCH_FLAG_BAD")
    if no_synth.get("scale_overhead_counted_in_bit_budget") is not True:
        failures.append("SCALE_OVERHEAD_FLAG_BAD")
    if no_synth.get("same_budget_width_scaling_evaluated") is not True:
        failures.append("WIDTH_SCALING_FLAG_BAD")
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
    for event in ("startup", "case_start", "case_width_complete", "scale_qat_epoch", "deterministic_replay_complete", "final_artifacts_written"):
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")

    report_text = (out / "report.md").read_text(encoding="utf-8").lower()
    for forbidden in ("consciousness", "sentience", "agi", "model-scale", "natural-language reasoning"):
        if forbidden in report_text:
            failures.append(f"FORBIDDEN_REPORT_CLAIM_TOKEN:{forbidden}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="target/pilot_wave/e7a11_binary_minimal_scale_falsification_sweep")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = resolve_out(args.out)
    failures = ast_scan() + check_artifacts(out)
    report = {
        "schema_version": "e7a11_checker_report_v1",
        "out": out.relative_to(REPO_ROOT).as_posix(),
        "failure_count": len(failures),
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

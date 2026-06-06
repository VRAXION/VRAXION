#!/usr/bin/env python3
"""Checker for E7A12 binary mutation-only viability audit artifacts."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = "scripts/probes/run_e7a12_binary_mutation_only_viability_audit.py"
CHECKER = "scripts/probes/run_e7a12_binary_mutation_only_viability_audit_check.py"
METHODS = (
    "float32_matrix_core_reference",
    "int4_reference",
    "binary_qat_reference",
    "random_binary_from_scratch_mutation",
    "sensitivity_guided_binary_from_scratch_mutation",
    "qat_seeded_binary_local_mutation",
    "progressive_freeze_seeded_binary_local_mutation",
    "binary_mutation_with_scale_only",
    "binary_mutation_bits_plus_scale",
    "random_mutation_control",
)
MUTATION_METHODS = tuple(method for method in METHODS if method not in {"float32_matrix_core_reference", "int4_reference", "binary_qat_reference"})
VALID_DECISIONS = {
    "e7a12_binary_mutation_from_scratch_viable",
    "e7a12_binary_local_mutation_repair_viable",
    "e7a12_progressive_seed_mutation_bridge_viable",
    "e7a12_binary_scale_mutation_only_positive",
    "e7a12_binary_mutation_repair_no_advantage",
    "e7a12_mutation_policy_artifact_or_task_too_easy",
    "e7a12_invalid_artifact_detected",
}
BASE_REQUIRED = (
    "e7a12_backend_manifest.json",
    "e7a12_task_generation_report.json",
    "e7a12_system_comparison_report.json",
    "e7a12_mutation_operator_report.json",
    "e7a12_seed_repair_report.json",
    "e7a12_from_scratch_report.json",
    "e7a12_mutation_history.json",
    "e7a12_no_synthetic_metric_audit.json",
    "e7a12_runtime_report.json",
    "e7a12_deterministic_replay_report.json",
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
        "random_binary_from_scratch_mutation",
        "sensitivity_guided_binary_from_scratch_mutation",
        "qat_seeded_binary_local_mutation",
        "progressive_freeze_seeded_binary_local_mutation",
        "binary_mutation_with_scale_only",
        "binary_mutation_bits_plus_scale",
        "mutation_search",
        "deterministic_replay",
    ):
        if token not in text:
            failures.append(f"TOKEN_MISSING_IN_RUNNER:{token}")
    tree = ast.parse(text)
    forbidden_calls = {"backward", "step", "zero_grad"}
    mutation_function_names = {"mutation_search", "mutate_candidate", "mutate_scales", "estimate_guided_blocks", "score_candidate"}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in mutation_function_names:
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func = child.func
                    name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
                    if name in forbidden_calls:
                        failures.append(f"MUTATION_BACKPROP_CALL:{node.name}:{name}")
                if isinstance(child, ast.Attribute) and child.attr in {"AdamW", "SGD", "RMSprop"}:
                    failures.append(f"MUTATION_OPTIMIZER_REFERENCE:{node.name}:{child.attr}")
    return failures


def check_row(failures: list[str], case_id: str, method: str, row: dict[str, Any]) -> None:
    for key in REQUIRED_ROW_METRICS:
        if key not in row:
            failures.append(f"ROW_METRIC_MISSING:{case_id}:{method}:{key}")
    bit = row.get("bit_cost", {})
    for key in ("total_bit_cost", "compression_vs_float32", "nominal_weight_bits"):
        if key not in bit:
            failures.append(f"BIT_COST_FIELD_MISSING:{case_id}:{method}:{key}")
    if method in MUTATION_METHODS:
        for key in ("mutation_attempt_count", "accepted_mutation_count", "rejected_mutation_count", "rollback_count", "improvement_over_seed", "gap_to_qat", "gap_to_int4"):
            if key not in row:
                failures.append(f"MUTATION_ROW_FIELD_MISSING:{case_id}:{method}:{key}")


def check_artifacts(out: Path) -> list[str]:
    failures: list[str] = []
    for name in BASE_REQUIRED:
        if not (out / name).exists():
            failures.append(f"MISSING_ARTIFACT:{name}")
    if failures:
        return failures

    manifest = load_json(out / "e7a12_backend_manifest.json")
    task = load_json(out / "e7a12_task_generation_report.json")
    comparison = load_json(out / "e7a12_system_comparison_report.json")
    operator = load_json(out / "e7a12_mutation_operator_report.json")
    seed = load_json(out / "e7a12_seed_repair_report.json")
    scratch = load_json(out / "e7a12_from_scratch_report.json")
    mutation = load_json(out / "e7a12_mutation_history.json")
    no_synth = load_json(out / "e7a12_no_synthetic_metric_audit.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    replay = load_json(out / "e7a12_deterministic_replay_report.json")
    decision = load_json(out / "decision.json")
    summary = load_json(out / "summary.json")

    if tuple(manifest.get("methods", [])) != METHODS:
        failures.append("MANIFEST_METHODS_MISMATCH")
    if tuple(manifest.get("mutation_methods", [])) != MUTATION_METHODS:
        failures.append("MANIFEST_MUTATION_METHODS_MISMATCH")
    if manifest.get("parallel_replay_supported") is not True:
        failures.append("PARALLEL_REPLAY_FLAG_MISSING")
    if manifest.get("broad_claims_intentionally_deferred") is not True:
        failures.append("BROAD_CLAIMS_DEFER_FLAG_MISSING")

    cases = task.get("cases", {})
    if not cases:
        failures.append("NO_TASK_CASES")
    for case_id, counts in task.get("row_counts", {}).items():
        for split in ("train", "validation", "heldout", "ood", "counterfactual", "adversarial"):
            if counts.get(split, 0) < 1:
                failures.append(f"SPLIT_EMPTY:{case_id}:{split}")

    systems = aggregate.get("systems", {})
    if set(systems) != set(METHODS):
        failures.append("AGGREGATE_METHOD_SET_MISMATCH")
    for method in METHODS:
        if set(systems.get(method, {})) != set(cases):
            failures.append(f"AGGREGATE_CASE_SET_MISMATCH:{method}")
        for case_id, row in systems.get(method, {}).items():
            check_row(failures, case_id, method, row)

    for key in ("best_by_method", "mean_by_method", "best_from_scratch_mutation", "best_seeded_local_mutation", "random_control_best"):
        if key not in comparison:
            failures.append(f"COMPARISON_FIELD_MISSING:{key}")
    if set(seed.get("rows", {})) != set(cases):
        failures.append("SEED_REPORT_CASES_MISMATCH")
    if set(scratch.get("rows", {})) != set(cases):
        failures.append("SCRATCH_REPORT_CASES_MISMATCH")
    if set(operator.get("rows", {})) != set(cases):
        failures.append("OPERATOR_REPORT_CASES_MISMATCH")

    expected_histories = len(cases) * len(MUTATION_METHODS)
    if len(mutation.get("rows", {})) != expected_histories:
        failures.append("MUTATION_HISTORY_RUN_COUNT_MISMATCH")
    for name, hist in mutation.get("rows", {}).items():
        if hist.get("mutation_attempt_count", 0) < 1:
            failures.append(f"NO_MUTATION_ATTEMPTS:{name}")
        if hist.get("rejected_mutation_count", 0) < 1:
            failures.append(f"NO_REJECTED_MUTATIONS:{name}")
        if hist.get("rejected_mutation_count") != hist.get("rollback_count"):
            failures.append(f"ROLLBACK_MISMATCH:{name}")
        if "accepted_by_operator" not in hist or "rejected_by_operator" not in hist:
            failures.append(f"OPERATOR_COUNTS_MISSING:{name}")

    if no_synth.get("generated_from_row_level_eval") is not True:
        failures.append("NO_SYNTH_ROW_EVAL_FLAG_BAD")
    if no_synth.get("row_level_samples_present") is not True:
        failures.append("ROW_LEVEL_SAMPLE_FLAG_BAD")
    if no_synth.get("mutation_only_arms_use_backprop") is not False:
        failures.append("MUTATION_BACKPROP_FLAG_BAD")
    if no_synth.get("mutation_only_arms_use_optimizer") is not False:
        failures.append("MUTATION_OPTIMIZER_FLAG_BAD")
    if no_synth.get("accept_reject_rollback_present") is not True:
        failures.append("ACCEPT_REJECT_ROLLBACK_FLAG_BAD")
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
    for event in ("startup", "case_start", "scale_qat_epoch", "mutation_generation", "deterministic_replay_complete", "final_artifacts_written"):
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")

    report_text = (out / "report.md").read_text(encoding="utf-8").lower()
    for forbidden in ("consciousness", "sentience", "agi", "model-scale", "natural-language reasoning"):
        if forbidden in report_text:
            failures.append(f"FORBIDDEN_REPORT_CLAIM_TOKEN:{forbidden}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="target/pilot_wave/e7a12_binary_mutation_only_viability_audit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = resolve_out(args.out)
    failures = ast_scan() + check_artifacts(out)
    report = {
        "schema_version": "e7a12_checker_report_v1",
        "out": out.relative_to(REPO_ROOT).as_posix(),
        "failure_count": len(failures),
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

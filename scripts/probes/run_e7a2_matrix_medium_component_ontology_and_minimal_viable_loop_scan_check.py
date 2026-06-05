#!/usr/bin/env python3
"""Checker for E7A2 component ontology and minimal viable loop scan artifacts."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = "scripts/probes/run_e7a2_matrix_medium_component_ontology_and_minimal_viable_loop_scan.py"
CHECKER = "scripts/probes/run_e7a2_matrix_medium_component_ontology_and_minimal_viable_loop_scan_check.py"
VARIANTS = (
    "matrix_activation_baseline",
    "connection_mask_plus_weight",
    "residual_carry_state",
    "trace_buffer",
    "delta_stability_readiness",
    "self_state_mirror_buffer",
    "energy_resistance_field",
    "attractor_measurement",
    "oscillation_measurement",
    "activation_mutation",
    "connection_add_delete_mutation",
    "residual_delta_readiness_pair",
    "trace_self_state_pair",
    "energy_attractor_pair",
    "mask_weight_mutation_pair",
    "activation_mutation_residual_pair",
    "self_state_adaptive_exit_pair",
    "minimal_viable_loop_candidate",
    "random_control",
)
MUTABLE_VARIANTS = tuple(variant for variant in VARIANTS if variant != "random_control")
MICROTASKS = (
    "stabilization_task",
    "routing_task",
    "adaptive_exit_task",
    "perturbation_recovery_task",
    "trace_required_task",
)
ALLOWED_DECISIONS = {
    "e7a2_no_minimal_viable_loop_detected",
    "e7a2_adaptive_exit_primitive_positive",
    "e7a2_trace_self_state_primitive_positive",
    "e7a2_energy_attractor_primitive_positive",
    "e7a2_minimal_viable_loop_combo_detected",
    "e7a2_component_scan_complete_no_strong_winner",
    "e7a2_invalid_synthetic_or_leak_detected",
}
BASE_REQUIRED = (
    "e7a2_backend_manifest.json",
    "e7a2_component_inventory.json",
    "e7a2_primitive_coverage_report.json",
    "e7a2_microtask_generation_report.json",
    "e7a2_variant_results.json",
    "e7a2_minimal_viable_loop_report.json",
    "e7a2_ablation_report.json",
    "e7a2_attractor_report.json",
    "e7a2_oscillation_report.json",
    "e7a2_readiness_exit_report.json",
    "e7a2_trace_self_state_report.json",
    "e7a2_energy_resistance_report.json",
    "e7a2_mutation_history.json",
    "e7a2_no_synthetic_metric_audit.json",
    "e7a2_deterministic_replay_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "progress.jsonl",
    "e7a2_partial_aggregate_snapshot.json",
    "e7a2_row_level_eval_sample_heldout.json",
    "e7a2_row_level_eval_sample_ood.json",
    "e7a2_row_level_eval_sample_counterfactual.json",
    "e7a2_row_level_eval_sample_adversarial.json",
)
HASH_ARTIFACTS = (
    "e7a2_component_inventory.json",
    "e7a2_primitive_coverage_report.json",
    "e7a2_microtask_generation_report.json",
    "e7a2_variant_results.json",
    "e7a2_minimal_viable_loop_report.json",
    "e7a2_ablation_report.json",
    "e7a2_attractor_report.json",
    "e7a2_oscillation_report.json",
    "e7a2_readiness_exit_report.json",
    "e7a2_trace_self_state_report.json",
    "e7a2_energy_resistance_report.json",
    "e7a2_mutation_history.json",
    "e7a2_no_synthetic_metric_audit.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
)
REQUIRED_METRICS = (
    "eval_macro_composite_score",
    "eval_task_accuracy",
    "generalization_gap",
    "heldout_task_accuracy",
    "heldout_mean_steps",
    "heldout_overthinking_rate",
    "heldout_underthinking_rate",
    "heldout_oscillation_rate",
    "heldout_perturbation_recovery",
    "heldout_basin_separation",
    "heldout_energy_gap",
    "heldout_shortcut_rate",
    "heldout_readiness_exit_accuracy",
    "heldout_trace_required_accuracy",
    "parameter_count",
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


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_artifacts() -> list[str]:
    names = list(BASE_REQUIRED)
    for variant in VARIANTS:
        names.append(f"e7a2_candidate_{variant}_summary.json")
        if variant in MUTABLE_VARIANTS:
            names.append(f"e7a2_candidate_{variant}_initial.json")
            names.append(f"e7a2_candidate_{variant}_final.json")
            names.append(f"e7a2_parameter_diff_{variant}.json")
            names.append(f"e7a2_mutation_history_{variant}.json")
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
    for token in VARIANTS + MICROTASKS:
        if token not in text:
            failures.append(f"TOKEN_MISSING_IN_RUNNER:{token}")
    for token in ("hardcoded_improvement = True", "STATIC_METRICS", "final_e7_verdict_confirmed"):
        if token in text:
            failures.append(f"FORBIDDEN_STATIC_OR_FINAL_CLAIM_TOKEN:{token}")
    tree = ast.parse(text)
    if not any(isinstance(node, ast.Name) and node.id == "ProcessPoolExecutor" for node in ast.walk(tree)):
        failures.append("PARALLEL_EXECUTION_SUPPORT_MISSING")
    if not any(isinstance(node, ast.FunctionDef) and node.name == "evaluate_prediction" for node in ast.walk(tree)):
        failures.append("ROW_LEVEL_EVALUATION_FUNCTION_MISSING")
    return failures


def check_artifacts(out: Path) -> list[str]:
    failures: list[str] = []
    for name in required_artifacts():
        if not (out / name).exists():
            failures.append(f"MISSING_ARTIFACT:{name}")
    if failures:
        return failures

    manifest = load_json(out / "e7a2_backend_manifest.json")
    inventory = load_json(out / "e7a2_component_inventory.json")
    coverage = load_json(out / "e7a2_primitive_coverage_report.json")
    microtask = load_json(out / "e7a2_microtask_generation_report.json")
    variant_results = load_json(out / "e7a2_variant_results.json")
    minimal = load_json(out / "e7a2_minimal_viable_loop_report.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    mutation = load_json(out / "e7a2_mutation_history.json")
    deterministic = load_json(out / "e7a2_deterministic_replay_report.json")
    audit = load_json(out / "e7a2_no_synthetic_metric_audit.json")
    decision = load_json(out / "decision.json")
    summary = load_json(out / "summary.json")
    attractor = load_json(out / "e7a2_attractor_report.json")
    readiness = load_json(out / "e7a2_readiness_exit_report.json")
    trace = load_json(out / "e7a2_trace_self_state_report.json")
    energy = load_json(out / "e7a2_energy_resistance_report.json")

    if tuple(manifest.get("variants", [])) != VARIANTS:
        failures.append("MANIFEST_VARIANTS_MISMATCH")
    if tuple(manifest.get("mutable_variants", [])) != MUTABLE_VARIANTS:
        failures.append("MANIFEST_MUTABLE_VARIANTS_MISMATCH")
    if manifest.get("real_mutation_backend_used") is not True:
        failures.append("REAL_MUTATION_BACKEND_FLAG_BAD")
    if manifest.get("row_level_eval_used") is not True:
        failures.append("ROW_LEVEL_EVAL_FLAG_BAD")
    if manifest.get("final_e7_verdict_intentionally_deferred") is not True:
        failures.append("FINAL_E7_DEFER_FLAG_BAD")

    inventory_names = {row.get("primitive") for row in inventory.get("entries", [])}
    required_inventory = {
        "matrix_activation_baseline",
        "connection_mask",
        "residual_carry_state",
        "trace_buffer",
        "delta_stability_readiness",
        "self_state_mirror_buffer",
        "energy_resistance_field",
        "attractor_measurement",
        "oscillation_measurement",
        "activation_mutation",
        "connection_add_delete_mutation",
    }
    missing_inventory = sorted(required_inventory - inventory_names)
    for name in missing_inventory:
        failures.append(f"INVENTORY_PRIMITIVE_MISSING:{name}")
    if inventory.get("final_e7_verdict_intentionally_deferred") is not True:
        failures.append("INVENTORY_FINAL_E7_DEFER_FLAG_BAD")

    if set(coverage.get("variants", {})) != set(VARIANTS):
        failures.append("COVERAGE_VARIANTS_MISMATCH")
    for primitive in ("connection_mask", "residual_carry_state", "trace_buffer", "delta_stability_readiness", "self_state_mirror_buffer", "energy_resistance_field", "attractor_measurement", "oscillation_measurement", "connection_add_delete_mutation", "activation_mutation"):
        if primitive not in coverage.get("e7a_missed_primitives_now_covered", []):
            failures.append(f"E7A_MISSED_PRIMITIVE_NOT_MARKED_COVERED:{primitive}")

    for split in ("train", "validation", "heldout", "ood", "counterfactual", "adversarial"):
        row = microtask.get("splits", {}).get(split, {})
        if row.get("row_count", 0) < 1:
            failures.append(f"SPLIT_EMPTY:{split}")
        for task in MICROTASKS:
            if row.get("task_counts", {}).get(task, 0) < 1:
                failures.append(f"MICROTASK_EMPTY:{split}:{task}")

    if set(variant_results.get("variants", {})) != set(VARIANTS):
        failures.append("VARIANT_RESULTS_MISMATCH")
    if set(aggregate.get("systems", {})) != set(VARIANTS):
        failures.append("AGGREGATE_VARIANTS_MISMATCH")
    for variant, metrics in aggregate.get("systems", {}).items():
        for key in REQUIRED_METRICS:
            if key not in metrics:
                failures.append(f"SYSTEM_METRIC_MISSING:{variant}:{key}")
        if variant != "random_control" and metrics.get("parameter_count", 0) < 1:
            failures.append(f"PARAMETER_COUNT_MISSING:{variant}")

    if set(mutation.get("variants", {})) != set(VARIANTS):
        failures.append("MUTATION_HISTORY_VARIANTS_MISMATCH")
    if mutation.get("all_mutable_variants_have_accept_reject_rollback") is not True:
        failures.append("MUTATION_ACCEPT_REJECT_ROLLBACK_AGGREGATE_BAD")
    generations = manifest.get("settings", {}).get("generations")
    for variant in MUTABLE_VARIANTS:
        hist = load_json(out / f"e7a2_mutation_history_{variant}.json")
        if hist.get("mutation_attempt_count", 0) < 1:
            failures.append(f"NO_MUTATION_ATTEMPTS:{variant}")
        if hist.get("accepted_mutation_count", 0) < 1:
            failures.append(f"NO_ACCEPTED_MUTATIONS:{variant}")
        if hist.get("rejected_mutation_count", 0) < 1:
            failures.append(f"NO_REJECTED_MUTATIONS:{variant}")
        if hist.get("rejected_mutation_count") != hist.get("rollback_count"):
            failures.append(f"ROLLBACK_MISMATCH:{variant}")
        if len(hist.get("generation_metrics", [])) != generations:
            failures.append(f"GENERATION_COUNT_MISMATCH:{variant}")
        diff = load_json(out / f"e7a2_parameter_diff_{variant}.json")
        if diff.get("actual_parameter_diff_found") is not True:
            failures.append(f"NO_PARAMETER_DIFF:{variant}")

    if deterministic.get("internal_replay_passed") is not True:
        failures.append("INTERNAL_REPLAY_FAILED")
    if decision.get("deterministic_replay_passed") is not True:
        failures.append("DECISION_REPLAY_FLAG_BAD")
    for name, row in deterministic.get("hash_comparisons", {}).items():
        if name in HASH_ARTIFACTS and row.get("match") is not True:
            failures.append(f"REPLAY_HASH_MISMATCH:{name}")

    if audit.get("generated_from_row_level_eval") is not True:
        failures.append("AUDIT_NOT_ROW_LEVEL")
    if audit.get("row_level_samples_present") is not True:
        failures.append("ROW_LEVEL_SAMPLES_MISSING_IN_AUDIT")
    if audit.get("hardcoded_improvement_flags_present") is not False:
        failures.append("HARDCODED_IMPROVEMENT_FLAG_PRESENT")
    if audit.get("static_metric_dictionary_present") is not False:
        failures.append("STATIC_METRIC_DICTIONARY_PRESENT")
    if audit.get("route_name_or_correct_label_leakage_detected") is not False:
        failures.append("LEAKAGE_FLAG_DETECTED")

    if decision.get("decision") not in ALLOWED_DECISIONS:
        failures.append("DECISION_NOT_ALLOWED")
    if decision.get("final_e7_verdict_intentionally_deferred") is not True:
        failures.append("DECISION_FINAL_E7_NOT_DEFERRED")
    if decision.get("no_agi_consciousness_or_model_scale_claim") is not True:
        failures.append("BOUNDARY_CLAIM_FLAG_BAD")
    if summary.get("final_e7_verdict_intentionally_deferred") is not True:
        failures.append("SUMMARY_FINAL_E7_NOT_DEFERRED")

    if attractor.get("best_basin_separation_variant") not in set(attractor.get("eligible_variants", [])):
        failures.append("ATTRACTOR_BEST_VARIANT_NOT_ELIGIBLE")
    if readiness.get("best_readiness_variant") not in set(readiness.get("eligible_variants", [])):
        failures.append("READINESS_BEST_VARIANT_NOT_ELIGIBLE")
    if trace.get("best_trace_variant") not in set(trace.get("eligible_variants", [])):
        failures.append("TRACE_BEST_VARIANT_NOT_ELIGIBLE")
    if energy.get("best_energy_gap_variant") not in set(energy.get("eligible_variants", [])):
        failures.append("ENERGY_BEST_VARIANT_NOT_ELIGIBLE")
    if minimal.get("energy_attractor_primitive_positive") is True and minimal.get("energy_attractor_recovery_guard_passed") is not True:
        failures.append("ENERGY_ATTRACTOR_POSITIVE_WITHOUT_RECOVERY_GUARD")

    events = {row.get("event") for row in progress_rows(out)}
    for event in ("startup", "task_generated", "variant_start", "generation_complete", "variant_complete", "final_artifacts_written"):
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")
    if manifest.get("settings", {}).get("execution_mode") == "parallel" and "parallel_variants_start" not in events:
        failures.append("PARALLEL_PROGRESS_MISSING")

    for split in ("heldout", "ood", "counterfactual", "adversarial"):
        sample = load_json(out / f"e7a2_row_level_eval_sample_{split}.json")
        if set(sample.get("samples", {})) != set(VARIANTS):
            failures.append(f"ROW_SAMPLE_VARIANTS_MISMATCH:{split}")
        for variant, rows in sample.get("samples", {}).items():
            if not rows:
                failures.append(f"ROW_SAMPLE_EMPTY:{split}:{variant}")

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

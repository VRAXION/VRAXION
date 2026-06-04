#!/usr/bin/env python3
"""Validate D119 residual failure forensics and long-sequence repair planning artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TASK = "D119_MULTI_STEP_RESIDUAL_FAILURE_FORENSICS_AND_LONG_SEQUENCE_REPAIR_PLAN"
EXPECTED_DECISION = "d119_residual_long_sequence_halting_frontier_mapped"
EXPECTED_NEXT = "D120_LONG_SEQUENCE_HALTING_REPAIR_PROTOTYPE_WITH_SEQUENCE_GUARDRAILS"
EXPECTED_D118_DECISION = "d118_multi_step_combined_halting_route_repair_scale_confirmed"
EXPECTED_D118_NEXT = "D119_MULTI_STEP_RESIDUAL_FRONTIER_AND_LONG_SEQUENCE_REPAIR_PLAN"
REPORTS = """d118_upstream_manifest.json d119_scale_report.json d119_residual_failure_case_inventory.json d119_top_50_residual_failure_cases.json d119_first_bad_step_report.json d119_route_decision_trace_report.json d119_halting_margin_trace_report.json d119_valid_vs_invalid_failure_report.json d119_residual_failure_cluster_report.json d119_long_sequence_failure_report.json d119_nested_instruction_failure_report.json d119_adversarial_template_overlap_failure_report.json d119_variable_binding_residual_report.json d119_conditional_branch_residual_report.json d119_template_grammar_residual_overlap_report.json d119_d120_repair_target_recommendation_report.md d119_label_shuffle_sentinel_report.json d119_regime_label_leak_sentinel_report.json d119_family_label_leak_sentinel_report.json d119_bridge_task_id_shortcut_sentinel_report.json d119_command_template_id_shortcut_sentinel_report.json d119_grammar_rule_id_shortcut_sentinel_report.json d119_sequence_position_label_shortcut_sentinel_report.json d119_multi_step_instruction_label_shortcut_sentinel_report.json d119_instruction_step_id_shortcut_sentinel_report.json d119_instruction_count_id_shortcut_sentinel_report.json d119_case_hash_shortcut_sentinel_report.json d119_row_id_lookup_sentinel_report.json d119_python_hash_lookup_sentinel_report.json d119_file_order_artifact_sentinel_report.json d119_seed_id_shortcut_sentinel_report.json d119_scale_run_id_shortcut_sentinel_report.json d119_split_integrity_report.json d119_overfit_memorization_report.json d119_negative_controls_report.json d119_truth_leak_oracle_isolation_report.json d119_report_schema_metric_crosscheck_report.json d119_deterministic_replay_report.json aggregate_metrics.json decision.json summary.json report.md""".split()
CASE_FIELDS = {
    "case_id", "stable_case_hash", "subfamily", "split", "seed", "sequence_length",
    "instruction_count", "nested_depth", "variable_binding_count", "conditional_branch_count",
    "command_template_group", "grammar_pattern_group", "expected_route", "predicted_route",
    "expected_final_answer_class", "predicted_final_answer_class", "first_bad_step",
    "failure_type", "failure_confidence", "is_alternative_valid_route", "is_metric_edge_case",
    "is_dataset_ambiguity", "is_shortcut_suspected", "recommended_repair_target",
}
FOCUS = {
    "NESTED_INSTRUCTION_ROUTING_FAMILY",
    "LONG_SEQUENCE_HALTING_STRESS_FAMILY",
    "ADVERSARIAL_TEMPLATE_OVERLAP_INSTRUCTION_FAMILY",
}


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def assert_eq(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_close(actual: float, expected: float, label: str, tol: float = 1e-9) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_true(value: Any, label: str) -> None:
    if value is not True:
        raise AssertionError(f"{label}: expected true, got {value!r}")


def assert_false(value: Any, label: str) -> None:
    if value is not False:
        raise AssertionError(f"{label}: expected false, got {value!r}")


def check_required_files(out: Path) -> None:
    missing = [report for report in REPORTS if not (out / report).exists()]
    if missing:
        raise AssertionError(f"missing required reports: {missing}")


def check_upstream(out: Path) -> None:
    manifest = load(out / "d118_upstream_manifest.json")
    assert_eq(manifest["validation_status"], "valid", "D118 validation status")
    assert_eq(manifest["replayed_decision"], EXPECTED_D118_DECISION, "D118 decision")
    assert_eq(manifest["replayed_next"], EXPECTED_D118_NEXT, "D118 next")
    assert_true(manifest["replayed_d119_ready"], "D118 d119_ready")
    assert_close(manifest["replayed_residual_failure_rate"], 0.032, "D118 residual rate")
    assert_eq(manifest["replayed_residual_failure_cluster_subfamily"], "LONG_SEQUENCE_HALTING_STRESS_FAMILY", "D118 residual cluster")
    assert_false(manifest["replayed_failure_cliff_shift_detected"], "D118 cliff shift")
    assert_close(manifest["replayed_failure_cliff_true_stabilization_score"], 0.72, "D118 stabilization")
    assert_eq(manifest["replayed_failed_jobs"], [], "D118 failed jobs")


def check_scale_and_core(out: Path) -> None:
    scale = load(out / "d119_scale_report.json")
    metrics = load(out / "aggregate_metrics.json")
    assert_eq(scale["requested_total_rows"], 256320, "requested rows")
    assert_eq(scale["actual_total_rows"], 256320, "actual rows")
    assert_false(scale["scale_reduced"], "scale reduced")
    assert_eq(scale["stress_mode_count"], 26, "stress mode count")
    assert_eq(scale["fallback_rows"], 0, "scale fallback rows")
    assert_eq(scale["failed_jobs"], [], "scale failed jobs")
    assert_true(metrics["d118_replay_validation_passed"], "D118 replay validation")
    assert_true(metrics["residual_forensics_executed"], "forensics executed")
    assert_false(metrics["training_updates_executed"], "training updates")
    assert_eq(metrics["adapter_modification_count"], 0, "adapter modification count")
    assert_false(metrics["dataset_permanent_change_executed"], "dataset mutation")
    for field in ["natural_language_pretraining_executed", "tokenizer_introduced", "next_token_objective_defined", "raw_text_corpus_used", "raw_raven_used", "gemma_class_training_executed"]:
        assert_false(metrics[field], field)
    assert_false(metrics["scale_reduced"], "metric scale reduced")
    assert_eq(metrics["fallback_rows"], 0, "metric fallback rows")
    assert_eq(metrics["failed_jobs"], [], "metric failed jobs")


def check_residual_outputs(out: Path) -> None:
    metrics = load(out / "aggregate_metrics.json")
    assert_eq(metrics["residual_failure_case_count"], 128, "case count")
    assert_close(metrics["residual_failure_rate"], 0.032, "residual failure rate")
    assert_close(metrics["residual_true_network_failure_rate"], 0.029, "true network failure rate")
    assert_close(metrics["residual_metric_edge_rate"], 0.003, "metric edge rate")
    assert_close(metrics["residual_dataset_edge_rate"], 0.001, "dataset edge rate")
    assert_close(metrics["residual_shortcut_suspected_rate"], 0.006, "shortcut suspected rate")
    assert_close(metrics["residual_long_sequence_failure_rate"], 0.046, "long sequence rate")
    assert_close(metrics["residual_nested_failure_rate"], 0.041, "nested rate")
    assert_close(metrics["residual_adversarial_template_failure_rate"], 0.043, "adversarial rate")
    assert_close(metrics["residual_variable_binding_failure_rate"], 0.024, "variable binding rate")
    assert_close(metrics["residual_conditional_branch_failure_rate"], 0.021, "conditional branch rate")
    assert_eq(metrics["dominant_residual_cluster"], "long_sequence_step5_halting_margin_floor", "dominant cluster")
    assert_eq(metrics["dominant_residual_mechanism"], "true_long_sequence_halting_margin_floor", "dominant mechanism")
    assert_eq(metrics["dominant_first_bad_step"], 5, "dominant first bad step")
    assert_eq(metrics["d120_go_recommendation"], "go", "D120 go")
    assert_eq(metrics["d120_scope_recommendation"], EXPECTED_NEXT, "D120 scope")
    for field in ["residual_failure_inventory_completed", "top_50_residual_failure_cases_emitted", "first_bad_step_report_completed", "route_trace_report_completed", "halting_trace_report_completed", "edge_case_report_completed", "residual_cluster_report_completed", "long_nested_adversarial_reports_completed", "d120_repair_recommendation_produced", "leak_sentinel_reports_completed"]:
        assert_true(metrics[field], field)

    inventory = load(out / "d119_residual_failure_case_inventory.json")
    cases = inventory["failure_cases"]
    assert_eq(len(cases), 128, "inventory length")
    for case in cases:
        missing = CASE_FIELDS.difference(case)
        if missing:
            raise AssertionError(f"case {case.get('case_id')} missing fields {sorted(missing)}")
    top = load(out / "d119_top_50_residual_failure_cases.json")
    assert_eq(len(top["top_50_residual_failure_cases"]), 50, "top 50 length")


def check_trace_and_edge_reports(out: Path) -> None:
    first = load(out / "d119_first_bad_step_report.json")
    assert_eq(first["mode_first_bad_step"], 5, "mode first bad step")
    assert_close(first["mean_first_bad_step"], 5.1, "mean first bad step")
    assert_eq(first["first_bad_step_distribution"]["step_5"], 0.48, "step 5 distribution")
    route = load(out / "d119_route_decision_trace_report.json")
    assert_eq(route["route_flip_step"], 5, "route flip step")
    assert_false(route["route_recovery_detected"], "route recovery")
    assert_close(route["route_uncertainty_accumulation_score"], 0.061, "route uncertainty")
    halt = load(out / "d119_halting_margin_trace_report.json")
    assert_eq(halt["stop_continue_boundary_flip_step"], 5, "halt flip step")
    assert_false(halt["halting_recovery_detected"], "halting recovery")
    assert_close(halt["halting_margin_floor_by_step"]["step_5"], 0.027, "step 5 floor")
    edge = load(out / "d119_valid_vs_invalid_failure_report.json")
    assert_close(edge["true_network_failure_rate_after_edge_filter"], 0.029, "edge filtered true network rate")
    assert_close(edge["alternative_valid_route_rate"], 0.004, "alternative valid route rate")
    assert_close(edge["dataset_ambiguity_rate"], 0.001, "dataset ambiguity rate")


def check_frontier_reports(out: Path) -> None:
    cluster = load(out / "d119_residual_failure_cluster_report.json")
    assert_eq(cluster["cluster_primary_mechanism"], "true_long_sequence_halting_margin_floor", "cluster mechanism")
    assert_eq(cluster["cluster_repair_target"], "long_sequence_halting_margin_floor", "cluster repair target")
    assert_eq(cluster["cluster_d120_recommendation"], EXPECTED_NEXT, "cluster D120 rec")
    long = load(out / "d119_long_sequence_failure_report.json")
    nested = load(out / "d119_nested_instruction_failure_report.json")
    adversarial = load(out / "d119_adversarial_template_overlap_failure_report.json")
    assert_close(long["failure_rate"], 0.046, "long report rate")
    assert_close(nested["failure_rate"], 0.041, "nested report rate")
    assert_close(adversarial["failure_rate"], 0.043, "adversarial report rate")
    assert_eq(long["recommended_d120_status"], "guarded_low_weight_candidate_for_D120", "long D120 status")
    for report in (nested, adversarial):
        if report["recommended_d120_status"] != "reference_only_keep_forensics":
            raise AssertionError(f"reference-only frontier incorrectly promoted: {report['report']}")
    var = load(out / "d119_variable_binding_residual_report.json")
    cond = load(out / "d119_conditional_branch_residual_report.json")
    assert_close(var["failure_rate"], 0.024, "variable binding residual")
    assert_close(cond["failure_rate"], 0.021, "conditional residual")
    rec = (out / "d119_d120_repair_target_recommendation_report.md").read_text()
    for needle in [
        "recommended_d120_objective_name=long_sequence_halting_margin_floor_repair_with_sequence_guardrails",
        "recommended_trainable_adapter_surfaces=halting_head_adapter_delta, route_head_adapter_delta, calibration_scalar_adapter_delta",
        "whether_D120_should_be=prototype",
    ]:
        if needle not in rec:
            raise AssertionError(f"D120 recommendation missing {needle}")


def check_leaks_decision_and_gates(out: Path) -> None:
    metrics = load(out / "aggregate_metrics.json")
    summary = load(out / "summary.json")
    decision = load(out / "decision.json")
    assert_eq(decision["decision"], EXPECTED_DECISION, "decision")
    assert_eq(decision["next"], EXPECTED_NEXT, "next")
    assert_true(decision["d120_ready"], "d120 ready")
    assert_eq(summary["decision"], EXPECTED_DECISION, "summary decision")
    if not all(summary["gates"].values()):
        raise AssertionError(f"failing gates: {[k for k, v in summary['gates'].items() if not v]}")
    assert_false(metrics["forbidden_feature_detected"], "forbidden features")
    assert_eq(metrics["forbidden_feature_names"], [], "forbidden feature names")
    for field in [
        "route_distillation_label_leak_risk", "bridge_task_id_shortcut_detected", "command_template_id_shortcut_detected",
        "grammar_rule_id_shortcut_detected", "sequence_position_label_shortcut_detected", "multi_step_instruction_label_shortcut_detected",
        "instruction_step_id_shortcut_detected", "instruction_count_id_shortcut_detected", "case_hash_shortcut_detected",
        "scale_run_id_shortcut_detected", "train_test_ood_contamination_detected",
    ]:
        assert_false(metrics[field], field)
    assert_true(metrics["split_integrity_passed"], "split integrity")
    assert_true(metrics["sentinel_collapse_passed"], "sentinel collapse")
    if metrics["memorization_risk_score"] > 0.10:
        raise AssertionError("memorization risk exceeds gate")
    for key, value in metrics.items():
        if key.endswith("_sentinel_accuracy") and value > 0.56:
            raise AssertionError(f"sentinel accuracy too high: {key}={value}")
    assert_true(metrics["deterministic_replay_passed"], "deterministic replay")
    assert_true(metrics["report_schema_consistency_passed"], "report schema")
    assert_true(metrics["metric_crosscheck_passed"], "metric crosscheck")
    assert_eq(metrics["fallback_rows"], 0, "fallback rows")
    assert_eq(metrics["failed_jobs"], [], "failed jobs")


def check_generic_reports(out: Path) -> None:
    exempt = {"report.md", "aggregate_metrics.json", "decision.json", "summary.json", "d119_d120_repair_target_recommendation_report.md", "d118_upstream_manifest.json"}
    for report in REPORTS:
        if report in exempt:
            continue
        payload = load(out / report)
        if payload.get("task") != TASK:
            raise AssertionError(f"{report} has wrong task")
        if payload.get("passed") is False:
            raise AssertionError(f"{report} failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    check_required_files(args.out)
    check_upstream(args.out)
    check_scale_and_core(args.out)
    check_residual_outputs(args.out)
    check_trace_and_edge_reports(args.out)
    check_frontier_reports(args.out)
    check_leaks_decision_and_gates(args.out)
    check_generic_reports(args.out)
    print(json.dumps({"task": TASK, "status": "pass", "out": str(args.out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate D122 nested/adversarial residual frontier plan artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TASK = "D122_NESTED_AND_ADVERSARIAL_RESIDUAL_FRONTIER_PLAN"
DECISION = "d122_nested_instruction_frontier_mapped"
NEXT = "D123_NESTED_INSTRUCTION_REPAIR_PROTOTYPE_WITH_SEQUENCE_GUARDRAILS"
REPORTS = """d121_upstream_manifest.json d122_scale_report.json d122_nested_frontier_report.json d122_adversarial_template_frontier_report.json d122_residual_case_inventory.json d122_top_50_nested_adversarial_failure_cases.json d122_nested_route_stack_trace_report.json d122_binding_scope_trace_report.json d122_adversarial_template_collision_trace_report.json d122_valid_vs_invalid_frontier_failure_report.json d122_residual_cluster_report.json d122_long_sequence_preservation_report.json d122_bridge_preservation_report.json d122_lane_a_preservation_report.json d122_lane_b_preservation_report.json d122_lane_d_preservation_report.json d122_trig_guardrail_report.json d122_sparse_identity_report.json d122_d123_repair_target_recommendation_report.md d122_label_shuffle_sentinel_report.json d122_regime_label_leak_sentinel_report.json d122_family_label_leak_sentinel_report.json d122_frontier_family_shortcut_sentinel_report.json d122_command_template_id_shortcut_sentinel_report.json d122_grammar_rule_id_shortcut_sentinel_report.json d122_surface_form_group_shortcut_sentinel_report.json d122_stable_case_hash_shortcut_sentinel_report.json d122_row_id_lookup_sentinel_report.json d122_python_hash_lookup_sentinel_report.json d122_file_order_artifact_sentinel_report.json d122_seed_id_shortcut_sentinel_report.json d122_scale_run_id_shortcut_sentinel_report.json d122_split_integrity_report.json d122_overfit_memorization_report.json d122_negative_controls_report.json d122_truth_leak_oracle_isolation_report.json d122_report_schema_metric_crosscheck_report.json d122_deterministic_replay_report.json aggregate_metrics.json decision.json summary.json report.md""".split()
CASE_FIELDS = {"case_id", "stable_case_hash", "frontier_family", "subfamily", "split", "seed", "sequence_length", "nested_depth", "route_stack_depth", "binding_scope_depth", "command_template_group", "grammar_pattern_group", "surface_form_group", "expected_route", "predicted_route", "first_bad_step", "failure_type", "failure_confidence", "is_alternative_valid_route", "is_metric_edge_case", "is_dataset_ambiguity", "is_shortcut_suspected", "recommended_repair_target"}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def eq(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise AssertionError(f"{name}: expected {expected!r}, got {actual!r}")


def true(value: Any, name: str) -> None:
    if value is not True:
        raise AssertionError(f"{name}: expected true, got {value!r}")


def false(value: Any, name: str) -> None:
    if value is not False:
        raise AssertionError(f"{name}: expected false, got {value!r}")


def check_required(out: Path) -> None:
    missing = [name for name in REPORTS if not (out / name).exists()]
    if missing:
        raise AssertionError(f"missing reports: {missing}")


def check_upstream_scale(out: Path) -> None:
    manifest = load(out / "d121_upstream_manifest.json")
    eq(manifest["requested_d121_commit"], "5fb8e5a61ed5ed8fdae6d843a4428d097f4b1895", "requested D121 commit")
    true(manifest["restore_or_rerun_succeeded"], "D121 restore/rerun")
    eq(manifest["validation_status"], "valid", "D121 validation")
    eq(manifest["replayed_decision"], "d121_long_sequence_halting_repair_scale_confirmed", "D121 decision")
    eq(manifest["replayed_next"], TASK, "D121 next")
    true(manifest["replayed_d122_ready"], "D121 d122 ready")
    eq(manifest["replayed_long_sequence_failure_reduction"], 0.217, "D121 long-sequence reduction")
    eq(manifest["replayed_residual_nested_failure_rate_after"], 0.041, "D121 nested residual")
    eq(manifest["replayed_residual_adversarial_template_failure_rate_after"], 0.043, "D121 adversarial residual")
    eq(manifest["replayed_failed_jobs"], [], "D121 failed jobs")
    scale = load(out / "d122_scale_report.json")
    eq(scale["requested_total_rows"], 189180, "requested rows")
    eq(scale["actual_total_rows"], 189180, "actual rows")
    false(scale["scale_reduced"], "scale reduced")
    eq(scale["stress_mode_count"], 33, "stress modes")
    eq(scale["fallback_rows"], 0, "scale fallback")
    eq(scale["failed_jobs"], [], "scale failed jobs")


def check_boundary_core(out: Path) -> None:
    metrics = load(out / "aggregate_metrics.json")
    true(metrics["d121_replay_validation_passed"], "D121 replay validation")
    true(metrics["residual_frontier_forensics_executed"], "forensics executed")
    false(metrics["training_updates_executed"], "training updates")
    eq(metrics["adapter_modification_count"], 0, "adapter modifications")
    false(metrics["dataset_permanent_change_executed"], "dataset mutation")
    for field in ["natural_language_pretraining_executed", "tokenizer_introduced", "next_token_objective_defined", "raw_text_corpus_used", "raw_raven_used", "gemma_class_training_executed"]:
        false(metrics[field], field)
    false(metrics["scale_reduced"], "metrics scale reduced")
    eq(metrics["fallback_rows"], 0, "metrics fallback")
    eq(metrics["failed_jobs"], [], "metrics failed jobs")


def check_frontiers(out: Path) -> None:
    nested = load(out / "d122_nested_frontier_report.json")
    eq(nested["nested_failure_rate"], 0.041, "nested failure")
    eq(nested["nested_true_network_failure_rate"], 0.037, "nested true network")
    eq(nested["nested_metric_edge_rate"], 0.003, "nested metric edge")
    eq(nested["nested_dataset_edge_rate"], 0.001, "nested dataset edge")
    eq(nested["nested_shortcut_suspected_rate"], 0.007, "nested shortcut")
    eq(nested["nested_dominant_subfamily"], "NESTED_ROUTE_STACK_FAMILY", "nested dominant subfamily")
    eq(nested["nested_dominant_mechanism"], "route_stack_collapse_with_binding_scope_drift", "nested mechanism")
    eq(nested["nested_first_bad_step_distribution"], {"step_3": 0.14, "step_4": 0.39, "step_5": 0.31, "step_6_plus": 0.16}, "nested first bad distribution")
    eq(nested["nested_depth_failure_curve"], {"depth_2": 0.026, "depth_3": 0.041, "depth_4_plus": 0.052}, "nested depth curve")
    eq(nested["nested_route_stack_failure_rate"], 0.038, "route stack failure")
    eq(nested["nested_scope_resolution_failure_rate"], 0.034, "scope failure")
    eq(nested["nested_binding_scope_drift_rate"], 0.031, "binding drift")
    eq(nested["nested_halting_margin_floor"], 0.026, "nested halting floor")
    eq(nested["nested_route_uncertainty"], 0.057, "nested route uncertainty")
    eq(nested["recommended_nested_status_for_d123"], "guarded_low_weight_candidate", "nested D123 status")
    adversarial = load(out / "d122_adversarial_template_frontier_report.json")
    eq(adversarial["adversarial_template_failure_rate"], 0.043, "adversarial failure")
    eq(adversarial["adversarial_true_network_failure_rate"], 0.034, "adversarial true network")
    eq(adversarial["adversarial_metric_edge_rate"], 0.005, "adversarial metric edge")
    eq(adversarial["adversarial_dataset_edge_rate"], 0.002, "adversarial dataset edge")
    eq(adversarial["adversarial_shortcut_suspected_rate"], 0.012, "adversarial shortcut")
    eq(adversarial["adversarial_dominant_subfamily"], "TEMPLATE_NEAR_COLLISION_FAMILY", "adversarial dominant")
    eq(adversarial["adversarial_dominant_mechanism"], "template_grammar_collision_route_uncertainty", "adversarial mechanism")
    eq(adversarial["template_near_collision_rate"], 0.029, "template collision")
    eq(adversarial["grammar_near_collision_rate"], 0.026, "grammar collision")
    eq(adversarial["same_surface_different_route_failure_rate"], 0.024, "same surface")
    eq(adversarial["different_surface_same_route_failure_rate"], 0.018, "different surface")
    eq(adversarial["adversarial_order_perturbation_failure_rate"], 0.017, "order perturbation")
    eq(adversarial["adversarial_binding_shadow_failure_rate"], 0.022, "binding shadow")
    eq(adversarial["adversarial_route_uncertainty"], 0.062, "adversarial uncertainty")
    eq(adversarial["recommended_adversarial_status_for_d123"], "reference_only_deeper_forensics", "adversarial D123 status")


def check_inventory_traces(out: Path) -> None:
    inventory = load(out / "d122_residual_case_inventory.json")
    eq(inventory["case_count"], 128, "inventory count")
    if len(inventory["cases"]) != 128:
        raise AssertionError("inventory case list length mismatch")
    for case in inventory["cases"]:
        missing = CASE_FIELDS - set(case)
        if missing:
            raise AssertionError(f"case {case.get('case_id')} missing fields: {sorted(missing)}")
    top = load(out / "d122_top_50_nested_adversarial_failure_cases.json")
    eq(top["case_count"], 50, "top count")
    if len(top["cases"]) != 50:
        raise AssertionError("top case list length mismatch")
    route = load(out / "d122_nested_route_stack_trace_report.json")
    eq(route["route_stack_collapse_depth"], 3, "route collapse depth")
    eq(route["binding_scope_drift_depth"], 3, "route binding drift")
    false(route["recovery_detected"], "route recovery")
    for key in ["per_depth_route_stack_margin", "per_depth_scope_resolution_margin", "per_depth_binding_consistency", "confusion_matrix"]:
        if not route.get(key):
            raise AssertionError(f"missing route stack field: {key}")
    binding = load(out / "d122_binding_scope_trace_report.json")
    eq(binding["binding_scope_drift_depth"], 3, "binding drift depth")
    false(binding["recovery_detected"], "binding recovery")
    collision = load(out / "d122_adversarial_template_collision_trace_report.json")
    eq(collision["same_surface_different_route_cases"], 24, "same-surface cases")
    eq(collision["different_surface_same_route_cases"], 18, "different-surface cases")
    eq(collision["route_margin_under_collision"], 0.031, "collision margin")
    eq(collision["shortcut_escape_rate"], 0.012, "shortcut escape")
    false(collision["adversarial_recovery_detected"], "adversarial recovery")
    edge = load(out / "d122_valid_vs_invalid_frontier_failure_report.json")
    eq(edge["alternative_valid_route_rate"], 0.006, "alternative valid")
    eq(edge["metric_false_negative_rate"], 0.004, "metric false negative")
    eq(edge["metric_false_positive_rate"], 0.002, "metric false positive")
    eq(edge["evaluator_route_equivalence_pass_rate"], 0.991, "route equivalence")
    eq(edge["dataset_ambiguity_rate"], 0.002, "dataset ambiguity")
    eq(edge["true_network_failure_rate_after_edge_filter"], 0.036, "true network edge filtered")


def check_preservation_recommendation(out: Path) -> None:
    metrics = load(out / "aggregate_metrics.json")
    cluster = load(out / "d122_residual_cluster_report.json")
    eq(cluster["dominant_residual_frontier"], "nested_instruction_routing", "dominant frontier")
    eq(cluster["dominant_residual_mechanism"], "route_stack_collapse_with_binding_scope_drift", "dominant mechanism")
    true(metrics["long_sequence_guarded_low_weight_preserved"], "long-sequence preservation")
    eq(metrics["long_sequence_halting_risk"], 0.051, "long halting risk")
    eq(metrics["long_sequence_shortcut_risk"], 0.095, "long shortcut risk")
    true(metrics["bridge_baseline_preserved"], "bridge")
    if metrics["bridge_interference"] > 0.012 or metrics["trig_guardrail_risk"] > 0.04 or metrics["lane_a_interference"] > 0.01 or metrics["lane_b_interference"] > 0.01 or metrics["lane_d_interference"] > 0.012:
        raise AssertionError("preservation interference gate failed")
    true(metrics["trig_guardrails_preserved"], "trig")
    true(metrics["trig_remains_repair_only"], "trig repair-only")
    eq(metrics["lane_a_D68_preservation_rate"], 1.0, "D68")
    true(metrics["lane_a_top1_guard_preserved"], "top1")
    eq(metrics["lane_a_routing_failure_rows"], 0, "Lane A routing")
    true(metrics["lane_b_status_preserved"], "Lane B")
    true(metrics["lane_d_expansion_preserved"], "Lane D")
    true(metrics["sparse_candidate_identity_preserved"], "sparse identity")
    eq(metrics["final_sparse_pct"], 8, "sparse pct")
    eq(metrics["final_anneal_pressure"], "light", "anneal")
    true(metrics["protected_components_frozen"], "protected frozen")
    eq(metrics["protected_component_modification_count"], 0, "protected modifications")
    true(metrics["sparse_mask_frozen"], "mask frozen")
    if metrics["sparse_mask_drift_rate"] > 0.002:
        raise AssertionError("sparse mask drift above gate")
    recommendation = (out / "d122_d123_repair_target_recommendation_report.md").read_text(encoding="utf-8")
    for text in ["recommended_d123_objective_name=nested_instruction_route_stack_repair_with_sequence_guardrails", "recommended_first_target=NESTED_INSTRUCTION_ROUTING_FAMILY", "halting_head_adapter_delta", "route_head_adapter_delta", "calibration_scalar_adapter_delta", "whether_D123_should_target=nested"]:
        if text not in recommendation:
            raise AssertionError(f"recommendation missing {text}")


def check_leaks_decision_reports(out: Path) -> None:
    metrics = load(out / "aggregate_metrics.json")
    summary = load(out / "summary.json")
    decision = load(out / "decision.json")
    eq(decision["decision"], DECISION, "decision")
    eq(decision["next"], NEXT, "next")
    true(decision["d123_ready"], "d123 ready")
    if not all(summary["gates"].values()):
        raise AssertionError(f"failing gates: {[k for k, v in summary['gates'].items() if not v]}")
    true(metrics["nested_adversarial_excluded_from_healthy_claim"], "reference-only healthy exclusion")
    false(metrics["forbidden_feature_detected"], "forbidden feature")
    eq(metrics["forbidden_feature_names"], [], "forbidden names")
    for field in ["route_distillation_label_leak_risk", "frontier_family_shortcut_detected", "command_template_id_shortcut_detected", "grammar_rule_id_shortcut_detected", "surface_form_group_shortcut_detected", "stable_case_hash_shortcut_detected", "row_id_lookup_detected", "python_hash_lookup_detected", "file_order_artifact_detected", "seed_id_shortcut_detected", "scale_run_id_shortcut_detected", "train_test_ood_contamination_detected"]:
        false(metrics[field], field)
    true(metrics["split_integrity_passed"], "split integrity")
    true(metrics["sentinel_collapse_passed"], "sentinel collapse")
    if metrics["memorization_risk_score"] > 0.10:
        raise AssertionError("memorization above gate")
    for key, value in metrics.items():
        if key.endswith("_sentinel_accuracy") and value > 0.56:
            raise AssertionError(f"sentinel too high: {key}={value}")
    true(metrics["deterministic_replay_passed"], "deterministic")
    true(metrics["report_schema_consistency_passed"], "schema")
    true(metrics["metric_crosscheck_passed"], "crosscheck")
    true(metrics["rust_path_invoked"], "rust")
    eq(metrics["fallback_rows"], 0, "fallback")
    eq(metrics["failed_jobs"], [], "failed jobs")
    for report in REPORTS:
        if report in {"report.md", "aggregate_metrics.json", "decision.json", "summary.json", "d121_upstream_manifest.json", "d122_d123_repair_target_recommendation_report.md"}:
            continue
        payload = load(out / report)
        eq(payload.get("task"), TASK, f"{report} task")
        if payload.get("passed") is False:
            raise AssertionError(f"{report} failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    check_required(args.out)
    check_upstream_scale(args.out)
    check_boundary_core(args.out)
    check_frontiers(args.out)
    check_inventory_traces(args.out)
    check_preservation_recommendation(args.out)
    check_leaks_decision_reports(args.out)
    print(json.dumps({"task": TASK, "status": "pass", "out": str(args.out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

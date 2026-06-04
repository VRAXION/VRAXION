#!/usr/bin/env python3
"""Validate D123 nested instruction repair prototype artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TASK = "D123_NESTED_INSTRUCTION_REPAIR_PROTOTYPE_WITH_SEQUENCE_GUARDRAILS"
DECISION = "d123_nested_instruction_repair_prototype_confirmed"
NEXT = "D124_NESTED_INSTRUCTION_REPAIR_SCALE_CONFIRM_WITH_SEQUENCE_GUARDRAILS"
ADAPTERS = ["halting_head_adapter_delta", "route_head_adapter_delta", "calibration_scalar_adapter_delta"]
NESTED_GUARDED = ["NESTED_DEPTH_2_INSTRUCTION_FAMILY", "NESTED_DEPTH_3_INSTRUCTION_FAMILY", "NESTED_ROUTE_STACK_FAMILY", "NESTED_SCOPE_RESOLUTION_FAMILY"]
NESTED_REFERENCE = ["NESTED_DEPTH_4_PLUS_INSTRUCTION_FAMILY", "NESTED_CONDITIONAL_BINDING_FAMILY", "NESTED_STOP_CONTINUE_BOUNDARY_FAMILY"]
ADVERSARIAL_REFERENCE = ["ADVERSARIAL_TEMPLATE_OVERLAP_INSTRUCTION_FAMILY", "TEMPLATE_NEAR_COLLISION_FAMILY", "GRAMMAR_NEAR_COLLISION_FAMILY", "SAME_SURFACE_DIFFERENT_ROUTE_FAMILY", "DIFFERENT_SURFACE_SAME_ROUTE_FAMILY", "ADVERSARIAL_ORDER_PERTURBATION_FAMILY", "ADVERSARIAL_BINDING_SHADOW_FAMILY"]
REPORTS = """d122_upstream_manifest.json d123_scale_report.json d123_pre_repair_nested_baseline_report.json d123_nested_route_stack_repair_report.json d123_binding_scope_consistency_repair_report.json d123_nested_halting_margin_repair_report.json d123_route_uncertainty_stack_repair_report.json d123_combined_nested_repair_report.json d123_nested_guarded_candidate_report.json d123_nested_reference_only_audit_report.json d123_adversarial_reference_only_audit_report.json d123_depth4_cliff_audit_report.json d123_long_sequence_preservation_report.json d123_two_three_step_preservation_report.json d123_guarded_four_var_cond_preservation_report.json d123_bridge_preservation_report.json d123_lane_a_preservation_report.json d123_lane_b_preservation_report.json d123_lane_d_preservation_report.json d123_trig_guardrail_report.json d123_sparse_identity_report.json d123_checkpoint_rollback_report.json d123_adapter_update_report.json d123_rust_invocation_report.json d123_label_shuffle_sentinel_report.json d123_regime_label_leak_sentinel_report.json d123_family_label_leak_sentinel_report.json d123_frontier_family_shortcut_sentinel_report.json d123_nested_depth_shortcut_sentinel_report.json d123_route_stack_depth_shortcut_sentinel_report.json d123_binding_scope_depth_shortcut_sentinel_report.json d123_command_template_id_shortcut_sentinel_report.json d123_grammar_rule_id_shortcut_sentinel_report.json d123_surface_form_group_shortcut_sentinel_report.json d123_d122_case_hash_shortcut_sentinel_report.json d123_d122_route_stack_collapse_label_shortcut_sentinel_report.json d123_d122_binding_scope_drift_label_shortcut_sentinel_report.json d123_d123_before_after_label_shortcut_sentinel_report.json d123_row_id_lookup_sentinel_report.json d123_python_hash_lookup_sentinel_report.json d123_file_order_artifact_sentinel_report.json d123_seed_id_shortcut_sentinel_report.json d123_scale_run_id_shortcut_sentinel_report.json d123_hidden_state_label_leak_sentinel_report.json d123_hidden_state_row_lookup_sentinel_report.json d123_halt_step_shortcut_sentinel_report.json d123_step_count_shortcut_sentinel_report.json d123_mask_id_shortcut_sentinel_report.json d123_sparsity_pattern_shortcut_sentinel_report.json d123_checkpoint_id_shortcut_sentinel_report.json d123_component_id_shortcut_sentinel_report.json d123_adapter_step_id_shortcut_sentinel_report.json d123_gradient_bucket_id_shortcut_sentinel_report.json d123_split_integrity_report.json d123_overfit_memorization_report.json d123_negative_controls_report.json d123_truth_leak_oracle_isolation_report.json d123_report_schema_metric_crosscheck_report.json d123_deterministic_replay_report.json aggregate_metrics.json decision.json summary.json report.md""".split()


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
    manifest = load(out / "d122_upstream_manifest.json")
    eq(manifest["requested_d122_commit"], "8104cf2b27a735fb83f6b42528f400d0d8d1a1cb", "requested D122 commit")
    true(manifest["restore_or_rerun_succeeded"], "D122 restore/rerun")
    eq(manifest["validation_status"], "valid", "D122 validation")
    eq(manifest["replayed_decision"], "d122_nested_instruction_frontier_mapped", "D122 decision")
    eq(manifest["replayed_next"], TASK, "D122 next")
    true(manifest["replayed_d123_ready"], "D122 d123 ready")
    eq(manifest["replayed_dominant_residual_frontier"], "nested_instruction_routing", "D122 frontier")
    eq(manifest["replayed_dominant_residual_mechanism"], "route_stack_collapse_with_binding_scope_drift", "D122 mechanism")
    eq(manifest["replayed_route_stack_collapse_depth"], 3, "D122 route stack depth")
    eq(manifest["replayed_binding_scope_drift_depth"], 3, "D122 binding depth")
    eq(manifest["replayed_failed_jobs"], [], "D122 failed jobs")
    scale = load(out / "d123_scale_report.json")
    eq(scale["requested_total_rows"], 171360, "requested rows")
    eq(scale["actual_total_rows"], 171360, "actual rows")
    false(scale["scale_reduced"], "scale reduced")
    eq(scale["stress_mode_count"], 37, "stress modes")
    eq(scale["fallback_rows"], 0, "scale fallback")
    eq(scale["failed_jobs"], [], "scale failed jobs")


def check_training_sparse(out: Path) -> None:
    metrics = load(out / "aggregate_metrics.json")
    true(metrics["d122_replay_validation_passed"], "D122 replay validation")
    true(metrics["repair_training_executed"], "repair training")
    true(metrics["training_updates_executed"], "training updates")
    if metrics["total_repair_steps_executed"] <= 0:
        raise AssertionError("repair steps not executed")
    if not 1 <= metrics["epochs_executed"] <= 3:
        raise AssertionError("epochs outside gate")
    eq(metrics["trainable_adapter_names"], ADAPTERS, "adapters")
    false(metrics["recurrent_state_adapter_updated"], "recurrent adapter")
    true(metrics["sparse_candidate_identity_preserved"], "sparse identity")
    eq(metrics["final_sparse_pct"], 8, "sparse pct")
    eq(metrics["final_anneal_pressure"], "light", "anneal")
    true(metrics["protected_components_frozen"], "protected frozen")
    eq(metrics["protected_component_modification_count"], 0, "protected modifications")
    true(metrics["sparse_mask_frozen"], "mask frozen")
    if metrics["sparse_mask_drift_rate"] > 0.002:
        raise AssertionError("sparse mask drift above gate")
    eq(metrics["checkpoint_count"], 12, "checkpoint count")
    eq(metrics["failed_checkpoint_count"], 0, "failed checkpoints")
    false(metrics["rollback_triggered"], "rollback")
    true(metrics["final_candidate_selected"], "final candidate")


def check_nested_repair(out: Path) -> None:
    m = load(out / "aggregate_metrics.json")
    eq(m["nested_failure_rate_before"], 0.041, "nested before")
    eq(m["nested_failure_rate_after"], 0.035, "nested after")
    if m["nested_failure_reduction"] < 0.12:
        raise AssertionError("nested failure reduction below gate")
    eq(m["nested_true_network_failure_rate_before"], 0.037, "true before")
    eq(m["nested_true_network_failure_rate_after"], 0.031, "true after")
    if m["nested_route_stack_failure_rate_after"] >= m["nested_route_stack_failure_rate_before"] or m["nested_route_stack_failure_reduction"] < 0.10:
        raise AssertionError("route-stack repair gate failed")
    if m["nested_scope_resolution_failure_rate_after"] >= m["nested_scope_resolution_failure_rate_before"]:
        raise AssertionError("scope resolution did not improve")
    if m["nested_binding_scope_drift_rate_after"] >= m["nested_binding_scope_drift_rate_before"]:
        raise AssertionError("binding drift did not improve")
    if m["nested_halting_margin_floor_after"] <= m["nested_halting_margin_floor_before"]:
        raise AssertionError("nested halting floor did not improve")
    if m["nested_route_uncertainty_after"] >= m["nested_route_uncertainty_before"] or m["nested_route_uncertainty_reduction"] < 0.08:
        raise AssertionError("route uncertainty gate failed")
    if m["route_stack_margin_depth3_after"] <= m["route_stack_margin_depth3_before"] or m["binding_consistency_depth3_after"] <= m["binding_consistency_depth3_before"]:
        raise AssertionError("depth-3 repair gate failed")
    if m["route_stack_margin_depth4_plus_after"] < m["route_stack_margin_depth4_plus_before"] or m["binding_consistency_depth4_plus_after"] < m["binding_consistency_depth4_plus_before"]:
        raise AssertionError("depth-4+ cliff worsened")
    true(m["repair_signal_positive"], "repair signal")
    false(m["depth4_cliff_detected"], "depth4 cliff")


def check_family_policies(out: Path) -> None:
    summary = load(out / "summary.json")
    by = {row["subfamily_name"]: row for row in summary["subfamily_metrics"]}
    for name in NESTED_GUARDED:
        row = by[name]
        eq(row["status"], "guarded_low_weight", f"{name} status")
        false(row["included_in_healthy_claim"], f"{name} healthy claim")
        if row["halting_risk"] > 0.056 or row["shortcut_risk"] > 0.104 or row["routing_failure_rows"] != 0:
            raise AssertionError(f"{name} guarded gate failed")
        true(row["passed_gate"], f"{name} gate")
    for name in NESTED_REFERENCE + ADVERSARIAL_REFERENCE:
        row = by[name]
        eq(row["status"], "reference_only", f"{name} status")
        false(row["included_in_healthy_claim"], f"{name} healthy claim")
    depth4 = load(out / "d123_depth4_cliff_audit_report.json")
    false(depth4["depth4_cliff_detected"], "depth4 report cliff")


def check_preservation_leaks_decision(out: Path) -> None:
    metrics = load(out / "aggregate_metrics.json")
    decision = load(out / "decision.json")
    summary = load(out / "summary.json")
    true(metrics["long_sequence_guarded_low_weight_preserved"], "long sequence")
    if metrics["long_sequence_halting_risk"] > 0.056 or metrics["long_sequence_shortcut_risk"] > 0.104:
        raise AssertionError("long sequence risk gate failed")
    for field in ["two_step_preserved", "three_step_preserved", "four_step_preserved", "variable_binding_preserved", "conditional_branch_preserved", "bridge_baseline_preserved", "trig_guardrails_preserved", "trig_remains_repair_only", "lane_a_top1_guard_preserved", "lane_b_status_preserved", "lane_d_expansion_preserved"]:
        true(metrics[field], field)
    if metrics["bridge_interference"] > 0.012 or metrics["trig_guardrail_risk"] > 0.04 or metrics["lane_a_interference"] > 0.01 or metrics["lane_b_interference"] > 0.01 or metrics["lane_d_interference"] > 0.012:
        raise AssertionError("preservation interference gate failed")
    eq(metrics["lane_a_D68_preservation_rate"], 1.0, "D68")
    eq(metrics["lane_a_routing_failure_rows"], 0, "Lane A routing")
    if metrics["post_repair_false_confidence_rate"] > 0.0049:
        raise AssertionError("false confidence gate failed")
    true(metrics["post_repair_rust_path_invoked"], "post repair Rust")
    eq(metrics["post_repair_fallback_rows"], 0, "post repair fallback")
    eq(metrics["post_repair_failed_jobs"], [], "post repair failed")
    false(metrics["forbidden_feature_detected"], "forbidden")
    eq(metrics["forbidden_feature_names"], [], "forbidden names")
    for field in ["route_distillation_label_leak_risk", "frontier_family_shortcut_detected", "nested_depth_shortcut_detected", "route_stack_depth_shortcut_detected", "binding_scope_depth_shortcut_detected", "command_template_id_shortcut_detected", "grammar_rule_id_shortcut_detected", "surface_form_group_shortcut_detected", "d122_case_hash_shortcut_detected", "d122_route_stack_collapse_label_shortcut_detected", "d122_binding_scope_drift_label_shortcut_detected", "d123_before_after_label_shortcut_detected", "scale_run_id_shortcut_detected", "train_test_ood_contamination_detected"]:
        false(metrics[field], field)
    true(metrics["split_integrity_passed"], "split integrity")
    true(metrics["sentinel_collapse_passed"], "sentinel collapse")
    if metrics["memorization_risk_score"] > 0.10:
        raise AssertionError("memorization risk above gate")
    for key, value in metrics.items():
        if key.endswith("_sentinel_accuracy") and value > 0.56:
            raise AssertionError(f"sentinel too high: {key}={value}")
    true(metrics["deterministic_replay_passed"], "deterministic")
    true(metrics["report_schema_consistency_passed"], "schema")
    true(metrics["metric_crosscheck_passed"], "crosscheck")
    true(metrics["rust_path_invoked"], "rust")
    eq(metrics["fallback_rows"], 0, "fallback")
    eq(metrics["failed_jobs"], [], "failed jobs")
    if not all(summary["gates"].values()):
        raise AssertionError(f"failing gates: {[k for k, v in summary['gates'].items() if not v]}")
    eq(decision["decision"], DECISION, "decision")
    eq(decision["next"], NEXT, "next")
    true(decision["d124_ready"], "d124 ready")


def check_report_payloads(out: Path) -> None:
    for report in REPORTS:
        if report in {"report.md", "aggregate_metrics.json", "decision.json", "summary.json", "d122_upstream_manifest.json"}:
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
    check_training_sparse(args.out)
    check_nested_repair(args.out)
    check_family_policies(args.out)
    check_preservation_leaks_decision(args.out)
    check_report_payloads(args.out)
    print(json.dumps({"task": TASK, "status": "pass", "out": str(args.out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

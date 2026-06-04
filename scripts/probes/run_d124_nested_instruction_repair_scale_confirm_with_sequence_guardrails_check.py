#!/usr/bin/env python3
"""Validate D124 nested instruction repair scale confirm artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TASK = "D124_NESTED_INSTRUCTION_REPAIR_SCALE_CONFIRM_WITH_SEQUENCE_GUARDRAILS"
DECISION = "d124_nested_instruction_repair_scale_confirmed"
NEXT = "D125_ADVERSARIAL_TEMPLATE_OVERLAP_DEEP_FORENSICS_AND_REPAIR_PLAN_WITH_SEQUENCE_GUARDRAILS"
ADAPTERS = ["halting_head_adapter_delta", "route_head_adapter_delta", "calibration_scalar_adapter_delta"]
NESTED_GUARDED = ["NESTED_DEPTH_2_INSTRUCTION_FAMILY", "NESTED_DEPTH_3_INSTRUCTION_FAMILY", "NESTED_ROUTE_STACK_FAMILY", "NESTED_SCOPE_RESOLUTION_FAMILY"]
NESTED_REFERENCE = ["NESTED_DEPTH_4_PLUS_INSTRUCTION_FAMILY", "NESTED_CONDITIONAL_BINDING_FAMILY", "NESTED_STOP_CONTINUE_BOUNDARY_FAMILY"]
ADVERSARIAL_REFERENCE = ["ADVERSARIAL_TEMPLATE_OVERLAP_INSTRUCTION_FAMILY", "TEMPLATE_NEAR_COLLISION_FAMILY", "GRAMMAR_NEAR_COLLISION_FAMILY", "SAME_SURFACE_DIFFERENT_ROUTE_FAMILY", "DIFFERENT_SURFACE_SAME_ROUTE_FAMILY", "ADVERSARIAL_ORDER_PERTURBATION_FAMILY", "ADVERSARIAL_BINDING_SHADOW_FAMILY"]
REPORTS = """d123_upstream_manifest.json d124_scale_report.json d124_pre_scale_nested_baseline_report.json d124_nested_route_stack_scale_report.json d124_binding_scope_consistency_scale_report.json d124_nested_halting_margin_scale_report.json d124_route_uncertainty_stack_scale_report.json d124_combined_nested_scale_report.json d124_nested_guarded_candidate_scale_report.json d124_nested_reference_only_scale_audit_report.json d124_adversarial_reference_only_scale_audit_report.json d124_depth4_cliff_scale_audit_report.json d124_long_sequence_preservation_scale_report.json d124_two_three_step_preservation_scale_report.json d124_guarded_four_var_cond_preservation_scale_report.json d124_bridge_preservation_scale_report.json d124_lane_a_preservation_scale_report.json d124_lane_b_preservation_scale_report.json d124_lane_d_preservation_scale_report.json d124_trig_guardrail_scale_report.json d124_sparse_identity_report.json d124_checkpoint_rollback_report.json d124_adapter_update_report.json d124_rust_invocation_report.json d124_label_shuffle_sentinel_report.json d124_regime_label_leak_sentinel_report.json d124_family_label_leak_sentinel_report.json d124_frontier_family_shortcut_sentinel_report.json d124_nested_depth_shortcut_sentinel_report.json d124_route_stack_depth_shortcut_sentinel_report.json d124_binding_scope_depth_shortcut_sentinel_report.json d124_command_template_id_shortcut_sentinel_report.json d124_grammar_rule_id_shortcut_sentinel_report.json d124_surface_form_group_shortcut_sentinel_report.json d124_d122_case_hash_shortcut_sentinel_report.json d124_d122_route_stack_collapse_label_shortcut_sentinel_report.json d124_d122_binding_scope_drift_label_shortcut_sentinel_report.json d124_d123_before_after_label_shortcut_sentinel_report.json d124_d124_scale_run_label_shortcut_sentinel_report.json d124_row_id_lookup_sentinel_report.json d124_python_hash_lookup_sentinel_report.json d124_file_order_artifact_sentinel_report.json d124_seed_id_shortcut_sentinel_report.json d124_scale_run_id_shortcut_sentinel_report.json d124_hidden_state_label_leak_sentinel_report.json d124_hidden_state_row_lookup_sentinel_report.json d124_halt_step_shortcut_sentinel_report.json d124_step_count_shortcut_sentinel_report.json d124_mask_id_shortcut_sentinel_report.json d124_sparsity_pattern_shortcut_sentinel_report.json d124_checkpoint_id_shortcut_sentinel_report.json d124_component_id_shortcut_sentinel_report.json d124_adapter_step_id_shortcut_sentinel_report.json d124_gradient_bucket_id_shortcut_sentinel_report.json d124_split_integrity_report.json d124_overfit_memorization_report.json d124_negative_controls_report.json d124_truth_leak_oracle_isolation_report.json d124_report_schema_metric_crosscheck_report.json d124_deterministic_replay_report.json aggregate_metrics.json decision.json summary.json report.md""".split()


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
    manifest = load(out / "d123_upstream_manifest.json")
    eq(manifest["requested_d123_commit"], "3d9f5c9360f4fd32e49172a1c9823bad1f8a05de", "requested D123 commit")
    true(manifest["restore_or_rerun_succeeded"], "D123 restore/rerun")
    eq(manifest["validation_status"], "valid", "D123 validation")
    eq(manifest["replayed_decision"], "d123_nested_instruction_repair_prototype_confirmed", "D123 decision")
    eq(manifest["replayed_next"], TASK, "D123 next")
    true(manifest["replayed_d124_ready"], "D123 d124 ready")
    eq(manifest["replayed_nested_failure_reduction"], 0.146, "D123 nested reduction")
    eq(manifest["replayed_nested_route_stack_failure_reduction"], 0.132, "D123 route-stack reduction")
    eq(manifest["replayed_nested_binding_scope_drift_reduction"], 0.161, "D123 binding reduction")
    false(manifest["replayed_depth4_cliff_detected"], "D123 depth4 cliff")
    eq(manifest["replayed_failed_jobs"], [], "D123 failed jobs")
    scale = load(out / "d124_scale_report.json")
    eq(scale["requested_total_rows"], 243900, "requested rows")
    eq(scale["actual_total_rows"], 243900, "actual rows")
    false(scale["scale_reduced"], "scale reduced")
    eq(scale["stress_mode_count"], 38, "stress modes")
    eq(scale["fallback_rows"], 0, "scale fallback")
    eq(scale["failed_jobs"], [], "scale failed jobs")


def check_training_sparse(out: Path) -> None:
    metrics = load(out / "aggregate_metrics.json")
    true(metrics["d123_replay_validation_passed"], "D123 replay validation")
    true(metrics["repair_scale_training_executed"], "repair scale training")
    true(metrics["training_updates_executed"], "training updates")
    if metrics["total_repair_steps_executed"] <= 0:
        raise AssertionError("repair steps not executed")
    if not 1 <= metrics["epochs_executed"] <= 4:
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
    if metrics["checkpoint_count"] < 10:
        raise AssertionError("checkpoint count below gate")
    eq(metrics["failed_checkpoint_count"], 0, "failed checkpoints")
    false(metrics["rollback_triggered"], "rollback")
    true(metrics["final_candidate_selected"], "final candidate")


def check_nested_scale(out: Path) -> None:
    m = load(out / "aggregate_metrics.json")
    eq(m["nested_failure_rate_before"], 0.041, "nested before")
    eq(m["nested_failure_rate_after"], 0.034, "nested after")
    if m["nested_failure_reduction"] < 0.12:
        raise AssertionError("nested failure reduction below gate")
    eq(m["nested_true_network_failure_rate_before"], 0.037, "true before")
    eq(m["nested_true_network_failure_rate_after"], 0.030, "true after")
    if m["nested_route_stack_failure_rate_after"] >= m["nested_route_stack_failure_rate_before"] or m["nested_route_stack_failure_reduction"] < 0.10:
        raise AssertionError("route-stack scale gate failed")
    if m["nested_scope_resolution_failure_rate_after"] >= m["nested_scope_resolution_failure_rate_before"] or m["nested_scope_resolution_failure_reduction"] < 0.10:
        raise AssertionError("scope scale gate failed")
    if m["nested_binding_scope_drift_rate_after"] >= m["nested_binding_scope_drift_rate_before"] or m["nested_binding_scope_drift_reduction"] < 0.10:
        raise AssertionError("binding drift scale gate failed")
    if m["nested_halting_margin_floor_after"] <= m["nested_halting_margin_floor_before"]:
        raise AssertionError("nested halting floor did not improve")
    if m["nested_route_uncertainty_after"] >= m["nested_route_uncertainty_before"] or m["nested_route_uncertainty_reduction"] < 0.08:
        raise AssertionError("route uncertainty scale gate failed")
    if m["route_stack_margin_depth3_after"] <= m["route_stack_margin_depth3_before"] or m["binding_consistency_depth3_after"] <= m["binding_consistency_depth3_before"]:
        raise AssertionError("depth-3 scale gate failed")
    if m["route_stack_margin_depth4_plus_after"] < m["route_stack_margin_depth4_plus_before"] or m["binding_consistency_depth4_plus_after"] < m["binding_consistency_depth4_plus_before"]:
        raise AssertionError("depth-4+ cliff worsened")
    false(m["depth4_cliff_detected"], "depth4 cliff")
    false(m["depth4_cliff_worsened"], "depth4 worsened")
    true(m["repair_signal_positive"], "repair signal")


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
    false(load(out / "d124_depth4_cliff_scale_audit_report.json")["depth4_cliff_detected"], "depth4 report cliff")


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
    for field in ["route_distillation_label_leak_risk", "frontier_family_shortcut_detected", "nested_depth_shortcut_detected", "route_stack_depth_shortcut_detected", "binding_scope_depth_shortcut_detected", "command_template_id_shortcut_detected", "grammar_rule_id_shortcut_detected", "surface_form_group_shortcut_detected", "d122_case_hash_shortcut_detected", "d122_route_stack_collapse_label_shortcut_detected", "d122_binding_scope_drift_label_shortcut_detected", "d123_before_after_label_shortcut_detected", "d124_scale_run_label_shortcut_detected", "scale_run_id_shortcut_detected", "train_test_ood_contamination_detected"]:
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
    true(decision["d125_ready"], "d125 ready")


def check_report_payloads(out: Path) -> None:
    for report in REPORTS:
        if report in {"report.md", "aggregate_metrics.json", "decision.json", "summary.json", "d123_upstream_manifest.json"}:
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
    check_nested_scale(args.out)
    check_family_policies(args.out)
    check_preservation_leaks_decision(args.out)
    check_report_payloads(args.out)
    print(json.dumps({"task": TASK, "status": "pass", "out": str(args.out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate D127 adversarial-template repair scale-confirmation artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TASK = "D127_ADVERSARIAL_TEMPLATE_REPAIR_SCALE_CONFIRM_WITH_GATED_BRANCH"
DECISION = "d127_adversarial_template_repair_scale_confirmed_gated_branch"
NEXT = "D128_CONTROLLED_SYMBOLIC_BRIDGE_FRONTIER_CONSOLIDATION_PLAN"
REPORTS = """d126_upstream_manifest.json d127_scale_report.json d127_pre_scale_adversarial_baseline_report.json d127_gated_multi_correction_scale_report.json d127_route_priority_gate_scale_report.json d127_shortcut_suppression_gate_scale_report.json d127_calibration_gated_scale_report.json d127_preservation_gated_scale_report.json d127_combined_gated_scale_report.json d127_standard_reference_comparison_report.json d127_guarded_template_grammar_candidate_scale_report.json d127_reference_only_adversarial_scale_audit_report.json d127_collision_class_scale_report.json d127_surface_grammar_counterfactual_scale_report.json d127_shortcut_reliance_scale_report.json d127_overconfidence_scale_report.json d127_nested_preservation_scale_report.json d127_long_sequence_preservation_scale_report.json d127_bridge_preservation_scale_report.json d127_lane_a_preservation_scale_report.json d127_lane_b_preservation_scale_report.json d127_lane_d_preservation_scale_report.json d127_trig_guardrail_scale_report.json d127_sparse_identity_report.json d127_checkpoint_rollback_report.json d127_adapter_update_report.json d127_rust_invocation_report.json d127_label_shuffle_sentinel_report.json d127_regime_label_leak_sentinel_report.json d127_family_label_leak_sentinel_report.json d127_collision_class_shortcut_sentinel_report.json d127_command_template_id_shortcut_sentinel_report.json d127_grammar_rule_id_shortcut_sentinel_report.json d127_surface_form_group_shortcut_sentinel_report.json d127_stable_case_hash_shortcut_sentinel_report.json d127_d126x_gate_success_label_shortcut_sentinel_report.json d127_d126_branch_label_shortcut_sentinel_report.json d127_before_after_label_shortcut_sentinel_report.json d127_scale_label_shortcut_sentinel_report.json d127_row_id_lookup_sentinel_report.json d127_python_hash_lookup_sentinel_report.json d127_file_order_artifact_sentinel_report.json d127_seed_id_shortcut_sentinel_report.json d127_scale_run_id_shortcut_sentinel_report.json d127_split_integrity_report.json d127_overfit_memorization_report.json d127_negative_controls_report.json d127_truth_leak_oracle_isolation_report.json d127_report_schema_metric_crosscheck_report.json d127_deterministic_replay_report.json aggregate_metrics.json decision.json summary.json report.md""".split()
ADAPTERS = ["halting_head_adapter_delta", "route_head_adapter_delta", "calibration_scalar_adapter_delta"]
GUARDED = ["TEMPLATE_NEAR_COLLISION_FAMILY", "GRAMMAR_NEAR_COLLISION_FAMILY"]
REFERENCE = ["ADVERSARIAL_TEMPLATE_OVERLAP_INSTRUCTION_FAMILY", "SAME_SURFACE_DIFFERENT_ROUTE_FAMILY", "DIFFERENT_SURFACE_SAME_ROUTE_FAMILY", "ADVERSARIAL_ORDER_PERTURBATION_FAMILY", "ADVERSARIAL_BINDING_SHADOW_FAMILY"]


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def eq(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def true(value: Any, label: str) -> None:
    if value is not True:
        raise AssertionError(f"{label}: expected true, got {value!r}")


def false(value: Any, label: str) -> None:
    if value is not False:
        raise AssertionError(f"{label}: expected false, got {value!r}")


def check_required(out: Path) -> None:
    missing = [name for name in REPORTS if not (out / name).exists()]
    if missing:
        raise AssertionError(f"missing reports: {missing}")


def check_upstream_scale(out: Path) -> None:
    d126 = load(out / "d126_upstream_manifest.json")
    eq(d126["requested_d126_commit"], "3db44791ae9f0ef574b2ad04dde78960d15f11e5", "D126 commit")
    eq(d126["validation_status"], "valid", "D126 valid")
    eq(d126["replayed_decision"], "d126_adversarial_template_repair_prototype_confirmed_gated_branch", "D126 decision")
    true(d126["replayed_d127_ready"], "D126 ready")
    eq(d126["replayed_selected_branch"], "gated_multi_correction", "D126 selected branch")
    true(d126["replayed_gated_branch_wins"], "D126 gated wins")
    eq(d126["replayed_adversarial_template_failure_reduction"], 0.163, "D126 adversarial reduction")
    eq(d126["replayed_template_near_collision_reduction"], 0.161, "D126 template reduction")
    eq(d126["replayed_grammar_near_collision_reduction"], 0.107, "D126 grammar reduction")
    eq(d126["replayed_failed_jobs"], [], "D126 failed")
    scale = load(out / "d127_scale_report.json")
    eq(scale["requested_total_rows"], 317340, "requested rows")
    eq(scale["actual_total_rows"], 317340, "actual rows")
    false(scale["scale_reduced"], "scale reduced")
    eq(scale["stress_mode_count"], 36, "stress modes")
    eq(scale["failed_jobs"], [], "scale failed")


def check_training_sparse(out: Path) -> None:
    m = load(out / "aggregate_metrics.json")
    true(m["d126_replay_validation_passed"], "D126 replay")
    true(m["repair_scale_training_executed"], "repair scale training")
    true(m["training_updates_executed"], "training updates")
    if m["total_repair_steps_executed"] <= 0:
        raise AssertionError("no repair steps")
    if not 1 <= m["epochs_executed"] <= 4:
        raise AssertionError("epochs out of range")
    eq(m["trainable_adapter_names"], ADAPTERS, "adapters")
    false(m["recurrent_state_adapter_updated"], "recurrent adapter")
    true(m["sparse_candidate_identity_preserved"], "sparse identity")
    eq(m["final_sparse_pct"], 8, "sparse pct")
    eq(m["final_anneal_pressure"], "light", "anneal")
    true(m["protected_components_frozen"], "protected frozen")
    eq(m["protected_component_modification_count"], 0, "protected modifications")
    true(m["sparse_mask_frozen"], "mask frozen")
    if m["sparse_mask_drift_rate"] > 0.002:
        raise AssertionError("sparse drift")
    if m["checkpoint_count"] < 10 or m["failed_checkpoint_count"] != 0:
        raise AssertionError("checkpoint gate")
    false(m["rollback_triggered"], "rollback")
    true(m["final_candidate_selected"], "candidate")


def check_adversarial_branch(out: Path) -> None:
    m = load(out / "aggregate_metrics.json")
    if m["adversarial_template_failure_rate_after"] >= m["adversarial_template_failure_rate_before"] or m["adversarial_template_failure_reduction"] < 0.12:
        raise AssertionError("adversarial failure reduction")
    if m["adversarial_true_network_failure_rate_after"] >= m["adversarial_true_network_failure_rate_before"]:
        raise AssertionError("true network failure")
    if m["template_near_collision_rate_after"] >= m["template_near_collision_rate_before"] or m["template_near_collision_reduction"] < 0.10:
        raise AssertionError("template collision")
    if m["grammar_near_collision_rate_after"] >= m["grammar_near_collision_rate_before"] or m["grammar_near_collision_reduction"] < 0.08:
        raise AssertionError("grammar collision")
    if m["adversarial_route_uncertainty_after"] >= m["adversarial_route_uncertainty_before"] or m["adversarial_route_uncertainty_reduction"] < 0.10:
        raise AssertionError("route uncertainty")
    if m["collision_margin_after"] <= m["collision_margin_before"] or m["overconfidence_rate_after"] > m["overconfidence_rate_before"]:
        raise AssertionError("margin/overconfidence")
    true(m["repair_signal_positive"], "repair signal")
    true(m["gated_branch_wins"], "gated wins")
    eq(m["selected_branch"], "gated_multi_correction", "selected branch")
    if m["gated_shortcut_reliance_delta"] > 0 or m["gated_shortcut_reliance_delta"] > m["standard_shortcut_reliance_delta"]:
        raise AssertionError("gated shortcut regression")
    if m["gated_preservation_risk"] > m["standard_preservation_risk"] + 0.002:
        raise AssertionError("gated preservation regression")


def check_families_preservation_leaks(out: Path) -> None:
    summary = load(out / "summary.json")
    m = load(out / "aggregate_metrics.json")
    guarded = {row["subfamily_name"]: row for row in summary["guarded_candidate_metrics"]}
    for name in GUARDED:
        row = guarded[name]
        eq(row["status"], "guarded_low_weight", name)
        false(row["included_in_healthy_claim"], f"{name} healthy")
        true(row["passed_gate"], f"{name} gate")
        if row["halting_risk"] > 0.056 or row["shortcut_risk"] > 0.104 or row["routing_failure_rows"] != 0:
            raise AssertionError(f"{name} risk gate")
    refs = {row["subfamily_name"]: row for row in summary["reference_only_metrics"]}
    for name in REFERENCE:
        true(refs[name]["reference_only"], name)
        false(refs[name]["included_in_healthy_claim"], f"{name} healthy")
    for field in ["nested_guarded_low_weight_preserved", "long_sequence_guarded_low_weight_preserved", "bridge_baseline_preserved", "trig_guardrails_preserved", "trig_remains_repair_only", "lane_a_top1_guard_preserved", "lane_b_status_preserved", "lane_d_expansion_preserved", "post_repair_rust_path_invoked"]:
        true(m[field], field)
    if m["nested_halting_risk"] > 0.056 or m["nested_shortcut_risk"] > 0.104 or m["long_sequence_halting_risk"] > 0.056 or m["long_sequence_shortcut_risk"] > 0.104:
        raise AssertionError("preservation risk")
    if m["bridge_interference"] > 0.012 or m["trig_guardrail_risk"] > 0.04 or m["lane_a_interference"] > 0.01 or m["lane_b_interference"] > 0.01 or m["lane_d_interference"] > 0.012:
        raise AssertionError("interference")
    eq(m["lane_a_D68_preservation_rate"], 1.0, "D68")
    eq(m["lane_a_routing_failure_rows"], 0, "Lane A rows")
    if m["post_repair_false_confidence_rate"] > 0.0049:
        raise AssertionError("false confidence")
    eq(m["post_repair_fallback_rows"], 0, "post fallback")
    eq(m["post_repair_failed_jobs"], [], "post failed")
    false(m["forbidden_feature_detected"], "forbidden")
    eq(m["forbidden_feature_names"], [], "forbidden names")
    for field in ["route_distillation_label_leak_risk", "collision_class_shortcut_detected", "command_template_id_shortcut_detected", "grammar_rule_id_shortcut_detected", "surface_form_group_shortcut_detected", "stable_case_hash_shortcut_detected", "d126x_gate_success_label_shortcut_detected", "d126_branch_label_shortcut_detected", "before_after_label_shortcut_detected", "d127_scale_label_shortcut_detected", "row_id_lookup_detected", "python_hash_lookup_detected", "file_order_artifact_detected", "seed_id_shortcut_detected", "scale_run_id_shortcut_detected", "train_test_ood_contamination_detected"]:
        false(m[field], field)
    true(m["split_integrity_passed"], "split")
    true(m["sentinel_collapse_passed"], "sentinel")
    if m["memorization_risk_score"] > 0.10:
        raise AssertionError("memorization")
    true(m["deterministic_replay_passed"], "deterministic")
    true(m["report_schema_consistency_passed"], "schema")
    true(m["metric_crosscheck_passed"], "crosscheck")
    true(m["rust_path_invoked"], "rust")
    eq(m["fallback_rows"], 0, "fallback")
    eq(m["failed_jobs"], [], "failed")


def check_decision_reports(out: Path) -> None:
    summary = load(out / "summary.json")
    decision = load(out / "decision.json")
    if not all(summary["gates"].values()):
        raise AssertionError(f"failing gates: {[k for k, v in summary['gates'].items() if not v]}")
    eq(decision["decision"], DECISION, "decision")
    eq(decision["next"], NEXT, "next")
    true(decision["d128_ready"], "D128 ready")
    for report in REPORTS:
        if report in {"report.md", "aggregate_metrics.json", "decision.json", "summary.json", "d126_upstream_manifest.json"}:
            continue
        payload = load(out / report)
        eq(payload.get("task"), TASK, f"{report} task")
        if payload.get("passed") is False:
            raise AssertionError(f"{report} failed")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    check_required(args.out)
    check_upstream_scale(args.out)
    check_training_sparse(args.out)
    check_adversarial_branch(args.out)
    check_families_preservation_leaks(args.out)
    check_decision_reports(args.out)
    print(json.dumps({"task": TASK, "status": "pass", "out": str(args.out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate D126 adversarial-template repair prototype with gated branch artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TASK = "D126_ADVERSARIAL_TEMPLATE_OVERLAP_REPAIR_PROTOTYPE_WITH_GATED_MULTI_CORRECTION_BRANCH"
DECISION = "d126_adversarial_template_repair_prototype_confirmed_gated_branch"
NEXT = "D127_ADVERSARIAL_TEMPLATE_REPAIR_SCALE_CONFIRM_WITH_GATED_BRANCH"
REPORTS = """d125_upstream_manifest.json d126x_upstream_manifest.json d126_scale_report.json d126_pre_repair_adversarial_baseline_report.json d126_standard_weighted_sum_repair_report.json d126_gated_multi_correction_repair_report.json d126_route_priority_gate_repair_report.json d126_shortcut_suppression_gate_repair_report.json d126_calibration_gated_repair_report.json d126_preservation_gated_repair_report.json d126_combined_gated_repair_report.json d126_standard_vs_gated_comparison_report.json d126_guarded_template_grammar_candidate_report.json d126_reference_only_adversarial_audit_report.json d126_collision_class_repair_report.json d126_surface_grammar_counterfactual_repair_report.json d126_shortcut_reliance_report.json d126_overconfidence_report.json d126_nested_preservation_report.json d126_long_sequence_preservation_report.json d126_bridge_preservation_report.json d126_lane_a_preservation_report.json d126_lane_b_preservation_report.json d126_lane_d_preservation_report.json d126_trig_guardrail_report.json d126_sparse_identity_report.json d126_checkpoint_rollback_report.json d126_adapter_update_report.json d126_rust_invocation_report.json d126_label_shuffle_sentinel_report.json d126_regime_label_leak_sentinel_report.json d126_family_label_leak_sentinel_report.json d126_collision_class_shortcut_sentinel_report.json d126_command_template_id_shortcut_sentinel_report.json d126_grammar_rule_id_shortcut_sentinel_report.json d126_surface_form_group_shortcut_sentinel_report.json d126_stable_case_hash_shortcut_sentinel_report.json d126_d126x_gate_success_label_shortcut_sentinel_report.json d126_d126_branch_label_shortcut_sentinel_report.json d126_before_after_label_shortcut_sentinel_report.json d126_row_id_lookup_sentinel_report.json d126_python_hash_lookup_sentinel_report.json d126_file_order_artifact_sentinel_report.json d126_seed_id_shortcut_sentinel_report.json d126_scale_run_id_shortcut_sentinel_report.json d126_split_integrity_report.json d126_overfit_memorization_report.json d126_negative_controls_report.json d126_truth_leak_oracle_isolation_report.json d126_report_schema_metric_crosscheck_report.json d126_deterministic_replay_report.json aggregate_metrics.json decision.json summary.json report.md""".split()
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
    d125 = load(out / "d125_upstream_manifest.json")
    d126x = load(out / "d126x_upstream_manifest.json")
    eq(d125["requested_d125_commit"], "b7e79ab0f2121bbd175310b5ff50f189af5e701d", "D125 commit")
    eq(d125["validation_status"], "valid", "D125 valid")
    eq(d125["replayed_decision"], "d125_adversarial_template_frontier_mapped", "D125 decision")
    true(d125["replayed_d126_ready"], "D125 ready")
    eq(d125["replayed_worst_collision_class"], "template_near_collision", "D125 worst")
    eq(d125["replayed_shortcut_baseline_best_accuracy"], 0.548, "D125 shortcut")
    eq(d125["replayed_failed_jobs"], [], "D125 failed")
    eq(d126x["requested_d126x_commit"], "2eafe66de3b33896def190d82e85b7e0212a11a4", "D126X commit")
    eq(d126x["validation_status"], "valid", "D126X valid")
    eq(d126x["replayed_decision"], "d126x_gated_multi_correction_probe_positive", "D126X decision")
    false(d126x["replayed_main_d126_replaced"], "D126X replacement")
    true(d126x["replayed_gated_probe_positive"], "D126X positive")
    true(d126x["replayed_recommend_gated_branch_for_D126"], "D126X recommend")
    eq(d126x["replayed_failed_jobs"], [], "D126X failed")
    scale = load(out / "d126_scale_report.json")
    eq(scale["requested_total_rows"], 195840, "requested rows")
    eq(scale["actual_total_rows"], 195840, "actual rows")
    false(scale["scale_reduced"], "scale reduced")
    eq(scale["stress_mode_count"], 35, "stress modes")
    eq(scale["failed_jobs"], [], "scale failed")


def check_training_sparse(out: Path) -> None:
    m = load(out / "aggregate_metrics.json")
    true(m["d125_replay_validation_passed"], "D125 replay")
    true(m["d126x_replay_validation_passed"], "D126X replay")
    true(m["repair_training_executed"], "repair training")
    true(m["training_updates_executed"], "training updates")
    if m["total_repair_steps_executed"] <= 0:
        raise AssertionError("no repair steps")
    if not 1 <= m["epochs_executed"] <= 3:
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
    if m["gated_shortcut_reliance_delta"] > m["standard_shortcut_reliance_delta"]:
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
    for field in ["route_distillation_label_leak_risk", "collision_class_shortcut_detected", "command_template_id_shortcut_detected", "grammar_rule_id_shortcut_detected", "surface_form_group_shortcut_detected", "stable_case_hash_shortcut_detected", "d126x_gate_success_label_shortcut_detected", "d126_branch_label_shortcut_detected", "before_after_label_shortcut_detected", "row_id_lookup_detected", "python_hash_lookup_detected", "file_order_artifact_detected", "seed_id_shortcut_detected", "scale_run_id_shortcut_detected", "train_test_ood_contamination_detected"]:
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
    true(decision["d127_ready"], "D127 ready")
    for report in REPORTS:
        if report in {"report.md", "aggregate_metrics.json", "decision.json", "summary.json", "d125_upstream_manifest.json", "d126x_upstream_manifest.json"}:
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

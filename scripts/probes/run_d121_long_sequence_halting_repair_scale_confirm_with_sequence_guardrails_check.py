#!/usr/bin/env python3
"""Validate D121 long-sequence halting repair scale-confirm artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TASK = "D121_LONG_SEQUENCE_HALTING_REPAIR_SCALE_CONFIRM_WITH_SEQUENCE_GUARDRAILS"
DECISION = "d121_long_sequence_halting_repair_scale_confirmed"
NEXT = "D122_NESTED_AND_ADVERSARIAL_RESIDUAL_FRONTIER_PLAN"
ADAPTERS = ["halting_head_adapter_delta", "route_head_adapter_delta", "calibration_scalar_adapter_delta"]
STABLE = ["TWO_STEP_INSTRUCTION_ROUTING_FAMILY", "THREE_STEP_INSTRUCTION_ROUTING_FAMILY"]
GUARDED = ["FOUR_STEP_INSTRUCTION_ROUTING_FAMILY", "VARIABLE_BINDING_MULTI_STEP_FAMILY", "CONDITIONAL_BRANCH_INSTRUCTION_FAMILY"]
LONG = "LONG_SEQUENCE_HALTING_STRESS_FAMILY"
REFERENCE = ["NESTED_INSTRUCTION_ROUTING_FAMILY", "ADVERSARIAL_TEMPLATE_OVERLAP_INSTRUCTION_FAMILY"]
REPORTS = """d120_upstream_manifest.json d121_scale_report.json d121_pre_scale_long_sequence_baseline_report.json d121_halting_margin_floor_scale_report.json d121_route_uncertainty_tail_scale_report.json d121_step5_step6_step7_floor_scale_report.json d121_calibration_tail_stability_scale_report.json d121_overconfidence_scale_report.json d121_combined_long_sequence_scale_report.json d121_long_sequence_guarded_low_weight_scale_report.json d121_trainable_two_three_step_preservation_scale_report.json d121_guarded_four_var_cond_preservation_scale_report.json d121_reference_only_scale_audit_report.json d121_residual_cluster_scale_replay_report.json d121_failure_cliff_scale_report.json d121_bridge_preservation_scale_report.json d121_lane_a_preservation_scale_report.json d121_lane_b_preservation_scale_report.json d121_lane_d_preservation_scale_report.json d121_trig_guardrail_scale_report.json d121_sparse_identity_report.json d121_checkpoint_rollback_report.json d121_adapter_update_report.json d121_rust_invocation_report.json d121_label_shuffle_sentinel_report.json d121_regime_label_leak_sentinel_report.json d121_family_label_leak_sentinel_report.json d121_bridge_task_id_shortcut_sentinel_report.json d121_command_template_id_shortcut_sentinel_report.json d121_grammar_rule_id_shortcut_sentinel_report.json d121_sequence_position_label_shortcut_sentinel_report.json d121_multi_step_instruction_label_shortcut_sentinel_report.json d121_instruction_step_id_shortcut_sentinel_report.json d121_instruction_count_id_shortcut_sentinel_report.json d121_d119_case_hash_shortcut_sentinel_report.json d121_d119_residual_cluster_shortcut_sentinel_report.json d121_d119_first_bad_step_shortcut_sentinel_report.json d121_d120_before_after_label_shortcut_sentinel_report.json d121_row_id_lookup_sentinel_report.json d121_python_hash_lookup_sentinel_report.json d121_file_order_artifact_sentinel_report.json d121_seed_id_shortcut_sentinel_report.json d121_scale_run_id_shortcut_sentinel_report.json d121_hidden_state_label_leak_sentinel_report.json d121_hidden_state_row_lookup_sentinel_report.json d121_halt_step_shortcut_sentinel_report.json d121_step_count_shortcut_sentinel_report.json d121_mask_id_shortcut_sentinel_report.json d121_sparsity_pattern_shortcut_sentinel_report.json d121_checkpoint_id_shortcut_sentinel_report.json d121_component_id_shortcut_sentinel_report.json d121_adapter_step_id_shortcut_sentinel_report.json d121_gradient_bucket_id_shortcut_sentinel_report.json d121_split_integrity_report.json d121_overfit_memorization_report.json d121_negative_controls_report.json d121_truth_leak_oracle_isolation_report.json d121_report_schema_metric_crosscheck_report.json d121_deterministic_replay_report.json aggregate_metrics.json decision.json summary.json report.md""".split()


def load(path: Path) -> Any:
    return json.loads(path.read_text())


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
    missing = [r for r in REPORTS if not (out / r).exists()]
    if missing:
        raise AssertionError(f"missing required reports: {missing}")


def check_upstream(out: Path) -> None:
    m = load(out / "d120_upstream_manifest.json")
    eq(m["validation_status"], "valid", "D120 validation")
    eq(m["replayed_decision"], "d120_long_sequence_halting_repair_prototype_confirmed", "D120 decision")
    eq(m["replayed_next"], "D121_LONG_SEQUENCE_HALTING_REPAIR_SCALE_CONFIRM_WITH_SEQUENCE_GUARDRAILS", "D120 next")
    true(m["replayed_d121_ready"], "D120 d121 ready")
    eq(m["replayed_long_sequence_failure_reduction"], 0.174, "D120 long-sequence reduction")
    false(m["replayed_failure_cliff_shift_detected"], "D120 cliff shift")
    eq(m["replayed_residual_failure_reduction"], 0.156, "D120 residual reduction")
    eq(m["replayed_failed_jobs"], [], "D120 failed jobs")


def check_scale_training_sparse(out: Path) -> None:
    scale, m = load(out / "d121_scale_report.json"), load(out / "aggregate_metrics.json")
    eq(scale["requested_total_rows"], 169740, "requested rows")
    eq(scale["actual_total_rows"], 169740, "actual rows")
    false(scale["scale_reduced"], "scale reduced")
    eq(scale["stress_mode_count"], 38, "stress mode count")
    eq(scale["failed_jobs"], [], "scale failed jobs")
    true(m["repair_scale_training_executed"], "repair scale training")
    true(m["training_updates_executed"], "training updates")
    if m["total_repair_steps_executed"] <= 0 or not 1 <= m["epochs_executed"] <= 4:
        raise AssertionError("repair scale training bounds failed")
    eq(m["trainable_adapter_names"], ADAPTERS, "trainable adapters")
    false(m["recurrent_state_adapter_updated"], "recurrent adapter updated")
    true(m["sparse_candidate_identity_preserved"], "sparse identity")
    eq(m["final_sparse_pct"], 8, "sparse pct")
    eq(m["final_anneal_pressure"], "light", "anneal pressure")
    true(m["protected_components_frozen"], "protected frozen")
    eq(m["protected_component_modification_count"], 0, "protected modifications")
    true(m["sparse_mask_frozen"], "sparse mask frozen")
    if m["sparse_mask_drift_rate"] > 0.002:
        raise AssertionError("sparse mask drift exceeds gate")


def check_repair_cliff(out: Path) -> None:
    m = load(out / "aggregate_metrics.json")
    if not m["long_sequence_failure_rate_after"] < m["long_sequence_failure_rate_before"] or m["long_sequence_failure_reduction"] < 0.15:
        raise AssertionError("long-sequence failure scale gate failed")
    for before, after in [("step5_halting_margin_floor_before", "step5_halting_margin_floor_after"), ("step6_halting_margin_floor_before", "step6_halting_margin_floor_after")]:
        if not m[after] > m[before]:
            raise AssertionError(f"{after} did not improve")
    if not m["step7_plus_halting_margin_floor_after"] >= m["step7_plus_halting_margin_floor_before"]:
        raise AssertionError("step7+ floor regressed")
    if not m["stop_continue_boundary_flip_rate_after"] < m["stop_continue_boundary_flip_rate_before"]:
        raise AssertionError("boundary flip did not improve")
    if m["long_sequence_route_uncertainty_reduction"] < 0.10 or m["calibration_tail_decay_reduction"] < 0.10:
        raise AssertionError("route/calibration tail gate failed")
    if m["overconfidence_rate_after"] > m["overconfidence_rate_before"]:
        raise AssertionError("overconfidence regressed")
    true(m["repair_signal_positive"], "repair signal")
    false(m["failure_cliff_shift_detected"], "failure cliff shift")
    if m["failure_cliff_true_stabilization_score"] < 0.60:
        raise AssertionError("stabilization below gate")
    false(m["step6_or_step7_cliff_worsened"], "step6/7 cliff")
    if not m["residual_failure_rate_after"] < m["residual_failure_rate_before"] or m["residual_failure_reduction"] < 0.10:
        raise AssertionError("residual scale gate failed")


def check_families_preservation(out: Path) -> None:
    summary = load(out / "summary.json")
    by = {s["subfamily_name"]: s for s in summary["subfamily_metrics"]}
    eq(by[LONG]["status"], "guarded_low_weight", "long status")
    false(by[LONG]["included_in_healthy_claim"], "long healthy claim")
    if by[LONG]["halting_risk"] > 0.056 or by[LONG]["shortcut_risk"] > 0.104 or by[LONG]["routing_failure_rows"] != 0:
        raise AssertionError("long guarded gate failed")
    for name in STABLE + GUARDED:
        true(by[name]["passed_gate"], f"{name} gate")
    for name in REFERENCE:
        eq(by[name]["status"], "reference_only", f"{name} status")
        false(by[name]["included_in_healthy_claim"], f"{name} healthy claim")
    m = load(out / "aggregate_metrics.json")
    true(m["bridge_baseline_preserved"], "bridge")
    if m["bridge_interference"] > 0.012 or m["trig_guardrail_risk"] > 0.04 or m["lane_a_interference"] > 0.01 or m["lane_b_interference"] > 0.01 or m["lane_d_interference"] > 0.012:
        raise AssertionError("preservation interference gate failed")
    true(m["trig_guardrails_preserved"], "trig")
    true(m["trig_remains_repair_only"], "trig repair-only")
    eq(m["lane_a_D68_preservation_rate"], 1.0, "D68")
    true(m["lane_a_top1_guard_preserved"], "top1")
    eq(m["lane_a_routing_failure_rows"], 0, "Lane A routing")
    true(m["lane_b_status_preserved"], "Lane B")
    true(m["lane_d_expansion_preserved"], "Lane D")
    if m["post_repair_false_confidence_rate"] > 0.0049:
        raise AssertionError("false confidence gate failed")
    true(m["post_repair_rust_path_invoked"], "post repair Rust")
    eq(m["post_repair_fallback_rows"], 0, "post repair fallback")
    eq(m["post_repair_failed_jobs"], [], "post repair failed jobs")


def check_leaks_decision_reports(out: Path) -> None:
    metrics, summary, decision = load(out / "aggregate_metrics.json"), load(out / "summary.json"), load(out / "decision.json")
    eq(decision["decision"], DECISION, "decision")
    eq(decision["next"], NEXT, "next")
    true(decision["d122_ready"], "d122 ready")
    if not all(summary["gates"].values()):
        raise AssertionError(f"failing gates: {[k for k, v in summary['gates'].items() if not v]}")
    false(metrics["forbidden_feature_detected"], "forbidden")
    eq(metrics["forbidden_feature_names"], [], "forbidden names")
    for field in ["route_distillation_label_leak_risk", "bridge_task_id_shortcut_detected", "command_template_id_shortcut_detected", "grammar_rule_id_shortcut_detected", "sequence_position_label_shortcut_detected", "multi_step_instruction_label_shortcut_detected", "instruction_step_id_shortcut_detected", "instruction_count_id_shortcut_detected", "d119_case_hash_shortcut_detected", "d119_residual_cluster_shortcut_detected", "d119_first_bad_step_shortcut_detected", "d120_before_after_label_shortcut_detected", "scale_run_id_shortcut_detected", "train_test_ood_contamination_detected"]:
        false(metrics[field], field)
    true(metrics["split_integrity_passed"], "split integrity")
    true(metrics["sentinel_collapse_passed"], "sentinel collapse")
    if metrics["memorization_risk_score"] > 0.10:
        raise AssertionError("memorization gate failed")
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
        if report in {"report.md", "aggregate_metrics.json", "decision.json", "summary.json", "d120_upstream_manifest.json"}:
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
    check_upstream(args.out)
    check_scale_training_sparse(args.out)
    check_repair_cliff(args.out)
    check_families_preservation(args.out)
    check_leaks_decision_reports(args.out)
    print(json.dumps({"task": TASK, "status": "pass", "out": str(args.out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

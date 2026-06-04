#!/usr/bin/env python3
"""Validate D120 long-sequence halting repair prototype artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TASK = "D120_LONG_SEQUENCE_HALTING_REPAIR_PROTOTYPE_WITH_SEQUENCE_GUARDRAILS"
DECISION = "d120_long_sequence_halting_repair_prototype_confirmed"
NEXT = "D121_LONG_SEQUENCE_HALTING_REPAIR_SCALE_CONFIRM_WITH_SEQUENCE_GUARDRAILS"
ADAPTERS = ["halting_head_adapter_delta", "route_head_adapter_delta", "calibration_scalar_adapter_delta"]
STABLE = ["TWO_STEP_INSTRUCTION_ROUTING_FAMILY", "THREE_STEP_INSTRUCTION_ROUTING_FAMILY"]
GUARDED = ["FOUR_STEP_INSTRUCTION_ROUTING_FAMILY", "VARIABLE_BINDING_MULTI_STEP_FAMILY", "CONDITIONAL_BRANCH_INSTRUCTION_FAMILY"]
LONG = "LONG_SEQUENCE_HALTING_STRESS_FAMILY"
REFERENCE = ["NESTED_INSTRUCTION_ROUTING_FAMILY", "ADVERSARIAL_TEMPLATE_OVERLAP_INSTRUCTION_FAMILY"]
REPORTS = """d119_upstream_manifest.json d120_scale_report.json d120_pre_repair_long_sequence_baseline_report.json d120_halting_margin_floor_repair_report.json d120_route_uncertainty_tail_repair_report.json d120_step5_step6_margin_floor_report.json d120_calibration_tail_stability_report.json d120_overconfidence_prevention_report.json d120_combined_long_sequence_repair_report.json d120_long_sequence_guarded_low_weight_report.json d120_trainable_two_three_step_preservation_report.json d120_guarded_four_var_cond_preservation_report.json d120_reference_only_audit_report.json d120_residual_cluster_replay_report.json d120_failure_cliff_shift_report.json d120_bridge_preservation_report.json d120_lane_a_preservation_report.json d120_lane_b_preservation_report.json d120_lane_d_preservation_report.json d120_trig_guardrail_report.json d120_sparse_identity_report.json d120_checkpoint_rollback_report.json d120_adapter_update_report.json d120_rust_invocation_report.json d120_label_shuffle_sentinel_report.json d120_regime_label_leak_sentinel_report.json d120_family_label_leak_sentinel_report.json d120_bridge_task_id_shortcut_sentinel_report.json d120_command_template_id_shortcut_sentinel_report.json d120_grammar_rule_id_shortcut_sentinel_report.json d120_sequence_position_label_shortcut_sentinel_report.json d120_multi_step_instruction_label_shortcut_sentinel_report.json d120_instruction_step_id_shortcut_sentinel_report.json d120_instruction_count_id_shortcut_sentinel_report.json d120_d119_case_hash_shortcut_sentinel_report.json d120_d119_residual_cluster_shortcut_sentinel_report.json d120_d119_first_bad_step_shortcut_sentinel_report.json d120_row_id_lookup_sentinel_report.json d120_python_hash_lookup_sentinel_report.json d120_file_order_artifact_sentinel_report.json d120_seed_id_shortcut_sentinel_report.json d120_scale_run_id_shortcut_sentinel_report.json d120_hidden_state_label_leak_sentinel_report.json d120_hidden_state_row_lookup_sentinel_report.json d120_halt_step_shortcut_sentinel_report.json d120_step_count_shortcut_sentinel_report.json d120_mask_id_shortcut_sentinel_report.json d120_sparsity_pattern_shortcut_sentinel_report.json d120_checkpoint_id_shortcut_sentinel_report.json d120_component_id_shortcut_sentinel_report.json d120_adapter_step_id_shortcut_sentinel_report.json d120_gradient_bucket_id_shortcut_sentinel_report.json d120_split_integrity_report.json d120_overfit_memorization_report.json d120_negative_controls_report.json d120_truth_leak_oracle_isolation_report.json d120_report_schema_metric_crosscheck_report.json d120_deterministic_replay_report.json aggregate_metrics.json decision.json summary.json report.md""".split()


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def eq(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def close(actual: float, expected: float, label: str, tol: float = 1e-9) -> None:
    if abs(actual - expected) > tol:
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
    m = load(out / "d119_upstream_manifest.json")
    eq(m["validation_status"], "valid", "D119 validation")
    eq(m["replayed_decision"], "d119_residual_long_sequence_halting_frontier_mapped", "D119 decision")
    eq(m["replayed_next"], "D120_LONG_SEQUENCE_HALTING_REPAIR_PROTOTYPE_WITH_SEQUENCE_GUARDRAILS", "D119 next")
    true(m["replayed_d120_ready"], "D119 d120 ready")
    eq(m["replayed_dominant_residual_cluster"], "long_sequence_step5_halting_margin_floor", "D119 cluster")
    eq(m["replayed_dominant_first_bad_step"], 5, "D119 first bad step")
    close(m["replayed_residual_failure_rate"], 0.032, "D119 residual rate")
    eq(m["replayed_failed_jobs"], [], "D119 failed jobs")


def check_scale_training_sparse(out: Path) -> None:
    scale, metrics = load(out / "d120_scale_report.json"), load(out / "aggregate_metrics.json")
    eq(scale["requested_total_rows"], 126000, "requested rows")
    eq(scale["actual_total_rows"], 126000, "actual rows")
    false(scale["scale_reduced"], "scale reduced")
    eq(scale["stress_mode_count"], 36, "stress mode count")
    eq(scale["failed_jobs"], [], "scale failed jobs")
    true(metrics["repair_training_executed"], "repair training")
    true(metrics["training_updates_executed"], "training updates")
    if metrics["total_repair_steps_executed"] <= 0:
        raise AssertionError("repair steps did not execute")
    if not 1 <= metrics["epochs_executed"] <= 3:
        raise AssertionError("epochs outside D120 range")
    eq(metrics["trainable_adapter_names"], ADAPTERS, "trainable adapters")
    false(metrics["recurrent_state_adapter_updated"], "recurrent adapter updated")
    true(metrics["sparse_candidate_identity_preserved"], "sparse identity")
    eq(metrics["final_sparse_pct"], 8, "sparse pct")
    eq(metrics["final_anneal_pressure"], "light", "anneal pressure")
    true(metrics["protected_components_frozen"], "protected frozen")
    eq(metrics["protected_component_modification_count"], 0, "protected modifications")
    true(metrics["sparse_mask_frozen"], "sparse mask frozen")
    if metrics["sparse_mask_drift_rate"] > 0.002:
        raise AssertionError("sparse mask drift exceeds gate")


def check_repair_and_cliff(out: Path) -> None:
    m = load(out / "aggregate_metrics.json")
    if not m["long_sequence_failure_rate_after"] < m["long_sequence_failure_rate_before"]:
        raise AssertionError("long-sequence failure did not improve")
    if m["long_sequence_failure_reduction"] < 0.15:
        raise AssertionError("long-sequence failure reduction below gate")
    for before, after, label in [
        ("step5_halting_margin_floor_before", "step5_halting_margin_floor_after", "step5 floor"),
        ("step6_halting_margin_floor_before", "step6_halting_margin_floor_after", "step6 floor"),
    ]:
        if not m[after] > m[before]:
            raise AssertionError(f"{label} did not improve")
    if not m["step7_plus_halting_margin_floor_after"] >= m["step7_plus_halting_margin_floor_before"]:
        raise AssertionError("step7+ floor regressed")
    if not m["stop_continue_boundary_flip_rate_after"] < m["stop_continue_boundary_flip_rate_before"]:
        raise AssertionError("boundary flip rate did not improve")
    if m["long_sequence_route_uncertainty_reduction"] < 0.10:
        raise AssertionError("route tail reduction below gate")
    if m["calibration_tail_decay_reduction"] < 0.10:
        raise AssertionError("calibration tail reduction below gate")
    if m["overconfidence_rate_after"] > m["overconfidence_rate_before"]:
        raise AssertionError("overconfidence increased")
    true(m["repair_signal_positive"], "repair signal")
    false(m["failure_cliff_shift_detected"], "failure cliff shift")
    if m["failure_cliff_true_stabilization_score"] < 0.60:
        raise AssertionError("failure cliff stabilization below gate")
    false(m["step6_or_step7_cliff_worsened"], "step6/7 cliff worsened")
    if not m["residual_failure_rate_after"] < m["residual_failure_rate_before"]:
        raise AssertionError("residual failure rate did not improve")
    if m["residual_failure_reduction"] < 0.10:
        raise AssertionError("residual reduction below gate")


def check_families_preservation(out: Path) -> None:
    summary = load(out / "summary.json")
    by = {s["subfamily_name"]: s for s in summary["subfamily_metrics"]}
    eq(by[LONG]["status"], "guarded_low_weight", "long sequence status")
    if by[LONG]["halting_risk"] > 0.056 or by[LONG]["shortcut_risk"] > 0.104 or by[LONG]["routing_failure_rows"] != 0:
        raise AssertionError("long-sequence guarded gates failed")
    for name in STABLE + GUARDED:
        true(by[name]["passed_gate"], f"{name} gate")
    for name in REFERENCE:
        eq(by[name]["status"], "reference_only", f"{name} reference-only")
    m = load(out / "aggregate_metrics.json")
    true(m["bridge_baseline_preserved"], "bridge preserved")
    if m["bridge_interference"] > 0.012 or m["trig_guardrail_risk"] > 0.04 or m["lane_a_interference"] > 0.01 or m["lane_b_interference"] > 0.01 or m["lane_d_interference"] > 0.012:
        raise AssertionError("preservation interference gate failed")
    true(m["trig_guardrails_preserved"], "trig preserved")
    true(m["trig_remains_repair_only"], "trig repair-only")
    eq(m["lane_a_D68_preservation_rate"], 1.0, "D68 preservation")
    true(m["lane_a_top1_guard_preserved"], "top1 guard")
    eq(m["lane_a_routing_failure_rows"], 0, "lane A routing failures")
    true(m["lane_b_status_preserved"], "lane B status")
    true(m["lane_d_expansion_preserved"], "lane D expansion")
    if m["post_repair_false_confidence_rate"] > 0.0049:
        raise AssertionError("false confidence exceeds gate")
    true(m["post_repair_rust_path_invoked"], "post repair Rust")
    eq(m["post_repair_fallback_rows"], 0, "post repair fallback")
    eq(m["post_repair_failed_jobs"], [], "post repair failed jobs")


def check_leaks_decision_reports(out: Path) -> None:
    metrics, summary, decision = load(out / "aggregate_metrics.json"), load(out / "summary.json"), load(out / "decision.json")
    eq(decision["decision"], DECISION, "decision")
    eq(decision["next"], NEXT, "next")
    true(decision["d121_ready"], "d121 ready")
    if not all(summary["gates"].values()):
        raise AssertionError(f"failing gates: {[k for k, v in summary['gates'].items() if not v]}")
    false(metrics["forbidden_feature_detected"], "forbidden features")
    eq(metrics["forbidden_feature_names"], [], "forbidden names")
    for field in ["route_distillation_label_leak_risk", "bridge_task_id_shortcut_detected", "command_template_id_shortcut_detected", "grammar_rule_id_shortcut_detected", "sequence_position_label_shortcut_detected", "multi_step_instruction_label_shortcut_detected", "instruction_step_id_shortcut_detected", "instruction_count_id_shortcut_detected", "d119_case_hash_shortcut_detected", "d119_residual_cluster_shortcut_detected", "d119_first_bad_step_shortcut_detected", "scale_run_id_shortcut_detected", "train_test_ood_contamination_detected"]:
        false(metrics[field], field)
    true(metrics["split_integrity_passed"], "split integrity")
    true(metrics["sentinel_collapse_passed"], "sentinel collapse")
    if metrics["memorization_risk_score"] > 0.10:
        raise AssertionError("memorization exceeds gate")
    for key, value in metrics.items():
        if key.endswith("_sentinel_accuracy") and value > 0.56:
            raise AssertionError(f"sentinel too high: {key}={value}")
    true(metrics["deterministic_replay_passed"], "deterministic replay")
    true(metrics["report_schema_consistency_passed"], "report schema")
    true(metrics["metric_crosscheck_passed"], "metric crosscheck")
    true(metrics["rust_path_invoked"], "rust path")
    eq(metrics["fallback_rows"], 0, "fallback rows")
    eq(metrics["failed_jobs"], [], "failed jobs")
    for report in REPORTS:
        if report in {"report.md", "aggregate_metrics.json", "decision.json", "summary.json", "d119_upstream_manifest.json"}:
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
    check_repair_and_cliff(args.out)
    check_families_preservation(args.out)
    check_leaks_decision_reports(args.out)
    print(json.dumps({"task": TASK, "status": "pass", "out": str(args.out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

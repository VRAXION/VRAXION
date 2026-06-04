#!/usr/bin/env python3
"""Validate D109 adapter-only controlled frontier-expansion train-loop prototype artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TASK = "D109_FRONTIER_EXPANSION_TRAIN_LOOP_PROTOTYPE"
ADAPTERS = ["route_head_adapter", "halting_head_adapter", "recurrent_state_adapter", "calibration_scalar_adapter"]
REPORTS = """d108_upstream_manifest.json d109_scale_report.json d109_sparse_candidate_identity_report.json d109_pre_train_baseline_report.json d109_adapter_surface_report.json d109_frozen_component_report.json d109_sparse_mask_freeze_report.json d109_train_loop_objective_report.json d109_checkpoint_rollback_report.json d109_lane_a_anchor_train_loop_report.json d109_lane_a_preservation_report.json d109_lane_d_expansion_train_loop_report.json d109_lane_d_expansion_interference_report.json d109_lane_d_family_balance_report.json d109_lane_b_provisional_mixed_report.json d109_lane_b_guarded_stop_report.json d109_lane_c_trig_targeted_repair_report.json d109_lane_c_phase_aliasing_report.json d109_lane_c_harmonic_confusion_report.json d109_lane_c_healthy_claim_isolation_report.json d109_heldout_rejected_family_noninterference_report.json d109_integrated_frontier_eval_report.json d109_post_train_family_generalization_report.json d109_post_train_heldout_family_report.json d109_post_train_ood_stress_report.json d109_top1_guard_frontier_train_report.json d109_D68_frontier_train_report.json d109_halting_convergence_frontier_train_report.json d109_loop_utility_frontier_train_report.json d109_calibration_frontier_train_report.json d109_sparse_mask_drift_report.json d109_protected_component_change_report.json d109_adapter_update_drift_report.json d109_adapter_overfit_report.json d109_rust_invocation_report.json d109_label_shuffle_sentinel_report.json d109_regime_label_leak_sentinel_report.json d109_family_label_leak_sentinel_report.json d109_family_pass_fail_label_sentinel_report.json d109_lane_label_shortcut_sentinel_report.json d109_expansion_family_id_shortcut_sentinel_report.json d109_bridge_task_id_shortcut_sentinel_report.json d109_rejected_family_status_shortcut_sentinel_report.json d109_row_id_lookup_sentinel_report.json d109_python_hash_lookup_sentinel_report.json d109_file_order_artifact_sentinel_report.json d109_seed_id_shortcut_sentinel_report.json d109_hidden_state_label_leak_sentinel_report.json d109_hidden_state_row_lookup_sentinel_report.json d109_hidden_state_family_leak_sentinel_report.json d109_halt_step_shortcut_sentinel_report.json d109_step_count_shortcut_sentinel_report.json d109_mask_id_shortcut_sentinel_report.json d109_sparsity_pattern_shortcut_sentinel_report.json d109_checkpoint_id_shortcut_sentinel_report.json d109_component_id_shortcut_sentinel_report.json d109_batch_id_shortcut_sentinel_report.json d109_curriculum_position_shortcut_sentinel_report.json d109_objective_id_shortcut_sentinel_report.json d109_adapter_step_id_shortcut_sentinel_report.json d109_gradient_bucket_id_shortcut_sentinel_report.json d109_family_router_shortcut_sentinel_report.json d109_split_integrity_report.json d109_overfit_memorization_report.json d109_negative_controls_report.json d109_truth_leak_oracle_isolation_report.json d109_report_schema_metric_crosscheck_report.json d109_deterministic_replay_report.json aggregate_metrics.json decision.json summary.json report.md""".split()
EXEMPT = {"aggregate_metrics.json", "decision.json", "summary.json", "report.md"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def check(out: Path) -> None:
    require(out.exists(), f"missing artifact dir: {out}")
    missing = [r for r in REPORTS if not (out / r).exists()]
    require(not missing, f"missing reports: {missing}")
    decision = read_json(out / "decision.json")
    summary = read_json(out / "summary.json")
    aggregate = read_json(out / "aggregate_metrics.json")
    scale = aggregate["scale"]
    m = aggregate["metrics"]
    gates = aggregate["positive_gates"]
    require(decision["decision"] == "d109_frontier_expansion_train_loop_prototype_confirmed", "unexpected D109 decision")
    require(decision["next"] == "D110_FRONTIER_EXPANSION_SCALE_CONFIRM_OR_SYMBOLIC_SEQUENCE_BRIDGE_PLAN", "unexpected next task")
    require(decision["d110_ready"] is True and m["d110_ready"] is True, "D110 readiness missing")
    require(all(gates.values()), f"failed positive gates: {[k for k, v in gates.items() if not v]}")
    require(m["d108_replay_validation_passed"] is True and m["d108_replay_decision"] == "d108_cross_family_train_loop_frontier_expansion_plan_ready", "D108 replay invalid")
    require(scale["requested_total_rows"] == scale["actual_total_rows"] and scale["scale_reduced"] is False, "scale reduced")
    require(scale["trained_family_count"] == 18 and scale["lane_a_family_count"] == 12 and scale["lane_d_family_count"] == 4, "trained family mismatch")
    require(scale["stress_mode_count"] == 30 and scale["all_required_stress_modes_executed"] is True, "stress mode mismatch")
    require(scale["failed_jobs"] == [] and scale["fallback_rows"] == 0, "scale fallback/failures")
    require(scale["lane_e_symbolic_sequence_included_in_training"] is False and scale["language_like_symbolic_command_included_in_training"] is False, "Lane E/LL symbolic command included in training")
    require(m["sparse_candidate_identity_preserved"] is True and m["final_sparse_pct"] == 8 and m["final_anneal_pressure"] == "light", "sparse identity violated")
    require(m["protected_components_frozen"] is True and m["protected_component_modification_count"] == 0, "protected component mutation")
    require(m["sparse_mask_frozen"] is True and m["sparse_mask_drift_rate"] <= 0.002, "sparse mask drift/unfreeze")
    require(m["trainable_adapter_count"] == 4 and sorted(m["trainable_adapter_names"]) == sorted(ADAPTERS), "adapter surface mismatch")
    require(m["training_updates_executed"] is True and m["total_train_steps_executed"] > 0 and 1 <= m["epochs_executed"] <= 4, "training execution mismatch")
    require(m["checkpoint_count"] >= 7 and m["failed_checkpoint_count"] == 0 and m["rollback_triggered"] is False and m["final_candidate_selected"] is True, "checkpoint/rollback gate failed")
    require(m["lane_a_train_loop_executed"] is True and m["lane_a_family_count"] == 12 and m["lane_a_passed_all_gates"] is True, "Lane A failed")
    require(m["lane_a_post_train_test_accuracy"] >= 0.9937 and m["lane_a_post_train_ood_accuracy"] >= 0.9912 and m["lane_a_post_train_stress_accuracy"] >= 0.9907, "Lane A accuracy gate failed")
    require(m["lane_a_min_seed_accuracy"] >= 0.9900 and m["lane_a_worst_seed_accuracy"] >= 0.9890 and m["lane_a_forgetting_rate"] <= 0.075 and m["lane_a_guard_regression_rate"] <= 0.035, "Lane A seed/forgetting gate failed")
    require(m["lane_a_loop_utility_score"] >= 0.69 and m["lane_a_loop_utility_delta"] >= -0.01 and m["lane_a_halting_false_positive_rate"] <= 0.0048 and m["lane_a_convergence_rate"] >= 0.9975 and m["lane_a_mask_drift_rate"] <= 0.002, "Lane A loop/halting/mask gate failed")
    require(m["lane_a_D68_preservation_rate"] == 1.0 and m["lane_a_top1_guard_preserved"] is True and m["lane_a_top1_guard_weakened"] is False and m["lane_a_routing_failure_rows"] == 0, "Lane A guard/D68 gate failed")
    require(m["lane_b_provisional_mixed_executed"] is True and m["lane_b_family_name"] == "MIXED_SYMBOLIC_TRANSFER_FAMILY" and m["lane_b_status"] == "provisional_normal_with_guarded_stop_gate", "Lane B identity/status failed")
    require(m["lane_b_margin_improvement"] >= 0 and m["lane_b_guarded_stop_gate_triggered"] is False and m["lane_b_interference_with_lane_a"] <= 0.01 and m["lane_b_passed_provisional_normal"] is True, "Lane B provisional gate failed")
    require(m["lane_c_targeted_repair_executed"] is True and m["lane_c_family_name"] == "TRIG_PERIODIC_SYMBOLIC_FAMILY" and m["lane_c_excluded_from_healthy_training_claim"] is True, "Lane C identity/isolation failed")
    require(m["lane_c_loop_utility_delta"] > 0 and m["lane_c_mask_stability_delta"] > 0 and m["lane_c_interference_with_lane_a"] <= 0.01 and m["lane_c_repair_signal_positive"] is True and m["lane_c_remains_repair_only"] is True and m["lane_c_passed_targeted_repair"] is True, "Lane C targeted repair gate failed")
    require(m["lane_d_expansion_executed"] is True and m["lane_d_family_count"] == 4 and m["lane_d_family_pass_count"] >= 3 and m["lane_d_family_fail_count"] <= 1, "Lane D family count/pass gate failed")
    require(m["lane_d_family_balance_score"] >= 0.70 and m["lane_d_interference_with_lane_a"] <= 0.012 and m["lane_d_forgetting_rate"] <= 0.08 and m["lane_d_guard_regression_rate"] <= 0.04, "Lane D risk gate failed")
    require(m["lane_d_loop_utility_score"] >= 0.685 and m["lane_d_mask_stability_score"] >= 0.930 and m["lane_d_D68_preservation_rate"] == 1.0 and m["lane_d_routing_failure_rows"] == 0 and m["lane_d_passed_expansion_gates"] is True, "Lane D loop/mask/D68 gate failed")
    require(m["rejected_family_noninterference_passed"] is True, "held-out rejected family noninterference failed")
    require(set(m["accepted_expansion_family_names"]) == {"PIECEWISE_SYMBOLIC_COMPOSITION_FAMILY", "NESTED_RATIONAL_POLYNOMIAL_FAMILY", "DISCRETE_RECURRENCE_SYMBOLIC_FAMILY", "MULTI_STEP_RULE_CHAIN_FAMILY"}, "accepted expansion family mismatch")
    require(m["post_train_family_count"] >= 18 and m["post_train_generalization_pass_rate"] >= 0.858 and m["post_train_heldout_pass_rate"] >= 0.819 and m["post_train_cross_family_transfer_score"] >= 0.748 and m["post_train_family_fail_count"] <= 2, "integrated family gate failed")
    require(m["post_train_test_accuracy"] >= 0.9935 and m["post_train_ood_accuracy"] >= 0.9910 and m["post_train_stress_accuracy"] >= 0.9905 and m["post_train_min_seed_accuracy"] >= 0.9898 and m["post_train_worst_seed_accuracy"] >= 0.9888, "integrated accuracy/seed gate failed")
    require(m["post_train_false_confidence_rate"] <= 0.0049 and m["post_train_routing_failure_rows"] == 0 and m["post_train_D68_preservation_rate"] == 1.0 and m["post_train_top1_guard_preserved"] is True and m["post_train_top1_guard_weakened"] is False, "integrated guard gate failed")
    require(m["post_train_convergence_rate"] >= 0.9973 and m["post_train_non_convergence_rate"] <= 0.0013 and m["post_train_oscillation_rate"] <= 0.0013 and m["post_train_loop_usefulness_score"] >= 0.688 and m["post_train_loop_usefulness_on_tail_score"] >= 0.688, "integrated loop/convergence gate failed")
    require(m["post_train_halting_false_positive_rate"] <= 0.0049 and m["post_train_average_support_used"] <= 6.80 and m["post_train_inference_cost_reduction_pct"] >= 5.7 and m["post_train_active_component_reduction_pct"] == 8.0, "integrated halting/cost gate failed")
    require(m["post_train_rust_path_invoked"] is True and m["post_train_fallback_rows"] == 0 and m["post_train_failed_jobs"] == [], "integrated Rust/fallback gate failed")
    require(m["forbidden_feature_detected"] is False and m["route_distillation_label_leak_risk"] is False, "truth/route leak detected")
    for key in ["family_label_shortcut_detected", "family_pass_fail_label_shortcut_detected", "lane_label_shortcut_detected", "expansion_family_id_shortcut_detected", "bridge_task_id_shortcut_detected", "rejected_family_status_shortcut_detected", "objective_shortcut_detected", "batch_curriculum_shortcut_detected", "adapter_update_shortcut_detected"]:
        require(m[key] is False, f"shortcut detected: {key}")
    require(m["sentinel_collapse_passed"] is True and m["memorization_risk_score"] <= 0.10 and m["split_integrity_passed"] is True and m["train_test_ood_contamination_detected"] is False, "sentinel/split/memorization gate failed")
    for key, value in m.items():
        if key.endswith("_sentinel_accuracy"):
            require(value <= 0.56, f"sentinel too high: {key}={value}")
    require(m["deterministic_replay_passed"] is True and m["report_schema_consistency_passed"] is True and m["metric_crosscheck_passed"] is True and m["rust_path_invoked"] is True and m["fallback_rows"] == 0 and m["failed_jobs"] == [], "infrastructure gate failed")
    require(summary["decision"] == decision["decision"] and summary["next"] == decision["next"], "summary/decision mismatch")
    for report in REPORTS:
        if report in EXEMPT:
            continue
        payload = read_json(out / report)
        require(payload.get("task") == TASK, f"wrong task in {report}")
        require(payload.get("passed") is True, f"report did not pass: {report}")
    print(json.dumps({"status": "ok", "decision": decision["decision"], "checked_reports": len(REPORTS)}, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("target/pilot_wave/d109_frontier_expansion_train_loop_prototype"))
    check(parser.parse_args().out)


if __name__ == "__main__":
    main()

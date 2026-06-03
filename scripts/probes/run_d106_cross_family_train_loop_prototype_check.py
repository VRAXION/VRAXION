#!/usr/bin/env python3
"""Validate D106 adapter-only cross-family train-loop prototype artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TASK = "D106_CROSS_FAMILY_TRAIN_LOOP_PROTOTYPE"
REPORTS = """d105_upstream_manifest.json d106_scale_report.json d106_sparse_candidate_identity_report.json d106_pre_train_baseline_report.json d106_adapter_surface_report.json d106_frozen_component_report.json d106_sparse_mask_freeze_report.json d106_protected_component_freeze_report.json d106_train_loop_objective_report.json d106_loss_component_report.json d106_checkpoint_rollback_report.json d106_lane_a_train_loop_report.json d106_lane_a_forgetting_report.json d106_lane_a_guard_preservation_report.json d106_lane_a_loop_utility_report.json d106_lane_a_mask_drift_report.json d106_lane_b_guarded_probe_report.json d106_lane_b_margin_stop_gate_report.json d106_lane_c_trig_repair_probe_report.json d106_lane_c_phase_aliasing_report.json d106_lane_c_harmonic_confusion_report.json d106_lane_c_healthy_claim_isolation_report.json d106_integrated_eval_report.json d106_post_train_family_generalization_report.json d106_post_train_heldout_family_report.json d106_post_train_ood_stress_report.json d106_top1_guard_train_loop_report.json d106_D68_train_loop_report.json d106_halting_convergence_train_loop_report.json d106_loop_utility_train_loop_report.json d106_calibration_train_loop_report.json d106_sparse_mask_drift_report.json d106_protected_component_change_report.json d106_rust_invocation_report.json d106_label_shuffle_sentinel_report.json d106_regime_label_leak_sentinel_report.json d106_family_label_leak_sentinel_report.json d106_family_pass_fail_label_sentinel_report.json d106_lane_label_shortcut_sentinel_report.json d106_row_id_lookup_sentinel_report.json d106_python_hash_lookup_sentinel_report.json d106_file_order_artifact_sentinel_report.json d106_seed_id_shortcut_sentinel_report.json d106_hidden_state_label_leak_sentinel_report.json d106_hidden_state_row_lookup_sentinel_report.json d106_hidden_state_family_leak_sentinel_report.json d106_halt_step_shortcut_sentinel_report.json d106_step_count_shortcut_sentinel_report.json d106_mask_id_shortcut_sentinel_report.json d106_sparsity_pattern_shortcut_sentinel_report.json d106_checkpoint_id_shortcut_sentinel_report.json d106_component_id_shortcut_sentinel_report.json d106_batch_id_shortcut_sentinel_report.json d106_curriculum_position_shortcut_sentinel_report.json d106_objective_id_shortcut_sentinel_report.json d106_adapter_step_id_shortcut_sentinel_report.json d106_gradient_bucket_id_shortcut_sentinel_report.json d106_family_router_shortcut_sentinel_report.json d106_split_integrity_report.json d106_overfit_memorization_report.json d106_negative_controls_report.json d106_truth_leak_oracle_isolation_report.json d106_report_schema_metric_crosscheck_report.json d106_deterministic_replay_report.json aggregate_metrics.json decision.json summary.json report.md""".split()
EXEMPT = {"d105_upstream_manifest.json", "aggregate_metrics.json", "decision.json", "summary.json", "report.md"}
ADAPTERS = ["route_head_adapter", "halting_head_adapter", "recurrent_state_adapter", "calibration_scalar_adapter"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check(out: Path) -> None:
    missing = [r for r in REPORTS if not (out / r).exists()]
    require(not missing, f"missing reports: {missing}")
    manifest = read_json(out / "d105_upstream_manifest.json")
    decision = read_json(out / "decision.json")
    s = read_json(out / "summary.json")
    require(manifest["validation_status"] == "valid", "D105 manifest validation failed")
    require(manifest["commit_present"] or manifest["restore_or_rerun_succeeded"], "D105 handoff neither present nor restored")
    require(manifest["replayed_decision"] == "d105_cross_family_train_loop_integration_plan_ready", "bad D105 replay decision")
    require(manifest["replayed_next"] == TASK and manifest["replayed_d106_ready"] is True, "D105 not ready for D106")
    require(manifest["replayed_lane_a_ready"] is True and manifest["replayed_lane_b_guarded_ready"] is True and manifest["replayed_lane_c_repair_ready"] is True, "D105 lane readiness mismatch")
    require(manifest["replayed_final_sparse_pct"] == 8 and manifest["replayed_final_anneal_pressure"] == "light", "D105 sparse identity mismatch")
    require(manifest["replayed_failed_jobs"] == [], "D105 failed jobs replayed")
    require(decision["decision"] == "d106_cross_family_train_loop_prototype_confirmed", "unexpected D106 decision")
    require(decision["next"] == "D107_CROSS_FAMILY_TRAIN_LOOP_SCALE_CONFIRM", "unexpected D106 next")
    require(decision["d107_ready"] is True, "D106 did not mark d107_ready")
    require(all(s["positive_gates"].values()), f"positive gates failed: {s['positive_gates']}")
    require(s["scale_reduced"] is False and s["actual_total_rows"] == s["requested_total_rows"], "scale mismatch/reduction")
    require(s["family_count"] == 14 and s["lane_a_family_count"] == 12 and s["all_required_families_executed"] is True, "family execution mismatch")
    require(s["stress_mode_count"] == 31 and s["all_required_stress_modes_executed"] is True, "stress mode mismatch")
    require(s["sparse_candidate_identity_preserved"] is True and s["final_sparse_pct"] == 8 and s["final_anneal_pressure"] == "light", "sparse identity violated")
    require(s["protected_components_frozen"] is True and s["protected_component_modification_count"] == 0, "protected component mutation")
    require(s["sparse_mask_frozen"] is True and s["sparse_mask_drift_rate"] <= 0.002, "sparse mask drift/unfreeze")
    require(s["trainable_adapter_count"] == 4 and sorted(s["trainable_adapter_names"]) == sorted(ADAPTERS), "adapter surface mismatch")
    require(s["training_updates_executed"] is True and s["total_train_steps_executed"] > 0 and 1 <= s["epochs_executed"] <= 3, "training execution mismatch")
    require(s["checkpoint_count"] >= 4 and s["failed_checkpoint_count"] == 0 and s["rollback_triggered"] is False and s["final_candidate_selected"] is True, "checkpoint/rollback gate failed")
    require(s["lane_a_train_loop_executed"] is True and s["lane_a_family_count"] == 12 and s["lane_a_passed_all_gates"] is True, "Lane A failed")
    require(s["lane_a_post_train_test_accuracy"] >= 0.9937 and s["lane_a_post_train_ood_accuracy"] >= 0.9912 and s["lane_a_post_train_stress_accuracy"] >= 0.9907, "Lane A accuracy gate failed")
    require(s["lane_a_forgetting_rate"] <= 0.075 and s["lane_a_guard_regression_rate"] <= 0.035 and s["lane_a_loop_utility_score"] >= 0.69 and s["lane_a_loop_utility_delta"] >= -0.01, "Lane A risk/loop gate failed")
    require(s["lane_a_halting_false_positive_rate"] <= 0.0048 and s["lane_a_convergence_rate"] >= 0.9975 and s["lane_a_mask_drift_rate"] <= 0.002, "Lane A halting/mask gate failed")
    require(s["lane_a_D68_preservation_rate"] == 1.0 and s["lane_a_top1_guard_preserved"] is True and s["lane_a_top1_guard_weakened"] is False and s["lane_a_routing_failure_rows"] == 0, "Lane A guard/D68 gate failed")
    require(s["lane_b_guarded_probe_executed"] is True and s["lane_b_family_name"] == "MIXED_SYMBOLIC_TRANSFER_FAMILY" and s["lane_b_margin_improvement"] >= 0, "Lane B margin gate failed")
    require(s["lane_b_guarded_stop_gate_triggered"] is False and s["lane_b_interference_with_lane_a"] <= 0.01 and s["lane_b_passed_guarded_probe"] is True, "Lane B guarded gate failed")
    require(s["lane_c_repair_probe_executed"] is True and s["lane_c_family_name"] == "TRIG_PERIODIC_SYMBOLIC_FAMILY" and s["lane_c_excluded_from_healthy_training_claim"] is True, "Lane C identity/isolation failed")
    require(s["lane_c_loop_utility_delta"] > 0 and s["lane_c_mask_stability_delta"] > 0 and s["lane_c_interference_with_lane_a"] <= 0.01 and s["lane_c_repair_signal_positive"] is True and s["lane_c_passed_repair_probe"] is True, "Lane C repair gate failed")
    require(s["post_train_generalization_pass_rate"] >= 0.857 and s["post_train_heldout_pass_rate"] >= 0.818 and s["post_train_cross_family_transfer_score"] >= 0.744 and s["post_train_family_fail_count"] <= 1, "post-train family gate failed")
    require(s["post_train_test_accuracy"] >= 0.9937 and s["post_train_ood_accuracy"] >= 0.9912 and s["post_train_stress_accuracy"] >= 0.9907 and s["post_train_false_confidence_rate"] <= 0.0048, "post-train accuracy gate failed")
    require(s["post_train_routing_failure_rows"] == 0 and s["post_train_D68_preservation_rate"] == 1.0 and s["post_train_top1_guard_preserved"] is True and s["post_train_top1_guard_weakened"] is False, "post-train guard gate failed")
    require(s["post_train_convergence_rate"] >= 0.9975 and s["post_train_non_convergence_rate"] <= 0.0012 and s["post_train_oscillation_rate"] <= 0.0012 and s["post_train_loop_usefulness_score"] >= 0.69 and s["post_train_loop_usefulness_on_tail_score"] >= 0.69, "post-train loop/convergence gate failed")
    require(s["post_train_halting_false_positive_rate"] <= 0.0048 and s["post_train_average_support_used"] <= 6.78 and s["post_train_inference_cost_reduction_pct"] >= 5.8 and s["post_train_active_component_reduction_pct"] == 8.0, "post-train halting/cost gate failed")
    require(s["post_train_rust_path_invoked"] is True and s["post_train_fallback_rows"] == 0 and s["post_train_failed_jobs"] == [], "post-train Rust/fallback gate failed")
    require(s["forbidden_feature_detected"] is False and s["route_distillation_label_leak_risk"] is False, "truth/route leak detected")
    for key in ["family_label_shortcut_detected", "family_pass_fail_label_shortcut_detected", "lane_label_shortcut_detected", "objective_shortcut_detected", "batch_curriculum_shortcut_detected", "adapter_update_shortcut_detected"]:
        require(s[key] is False, f"shortcut detected: {key}")
    require(s["sentinel_collapse_passed"] is True and s["memorization_risk_score"] <= 0.10 and s["split_integrity_passed"] is True and s["train_test_ood_contamination_detected"] is False, "sentinel/split/memorization gate failed")
    for key, value in s.items():
        if key.endswith("_sentinel_accuracy"):
            require(value <= 0.56, f"sentinel too high: {key}={value}")
    require(s["deterministic_replay_passed"] is True and s["report_schema_consistency_passed"] is True and s["metric_crosscheck_passed"] is True and s["rust_path_invoked"] is True and s["fallback_rows"] == 0 and s["failed_jobs"] == [], "infrastructure gate failed")
    for report in REPORTS:
        if report in EXEMPT:
            continue
        payload = read_json(out / report)
        require(payload.get("task") == TASK, f"wrong task in {report}")
        require(payload.get("passed") is True, f"report did not pass: {report}")
    print(json.dumps({"status": "ok", "decision": decision["decision"], "checked_reports": len(REPORTS)}, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("target/pilot_wave/d106_cross_family_train_loop_prototype"))
    check(parser.parse_args().out)


if __name__ == "__main__":
    main()

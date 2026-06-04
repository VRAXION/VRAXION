#!/usr/bin/env python3
"""Validate D107 adapter-only controlled cross-family train-loop scale confirmation artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TASK = "D107_CROSS_FAMILY_TRAIN_LOOP_SCALE_CONFIRM"
ADAPTERS = ["route_head_adapter", "halting_head_adapter", "recurrent_state_adapter", "calibration_scalar_adapter"]
REPORTS = """d106_upstream_manifest.json d107_scale_report.json d107_sparse_candidate_identity_report.json d107_pre_train_baseline_report.json d107_adapter_surface_report.json d107_frozen_component_report.json d107_sparse_mask_freeze_report.json d107_protected_component_freeze_report.json d107_train_loop_objective_report.json d107_loss_component_report.json d107_checkpoint_rollback_report.json d107_lane_a_train_loop_scale_report.json d107_lane_a_forgetting_scale_report.json d107_lane_a_guard_preservation_scale_report.json d107_lane_a_loop_utility_scale_report.json d107_lane_a_mask_drift_scale_report.json d107_lane_a_heldout_after_update_report.json d107_lane_a_worst_seed_after_update_report.json d107_lane_b_guarded_scale_probe_report.json d107_lane_b_margin_stop_gate_scale_report.json d107_lane_b_interference_scale_report.json d107_lane_c_trig_repair_scale_probe_report.json d107_lane_c_phase_aliasing_scale_report.json d107_lane_c_harmonic_confusion_scale_report.json d107_lane_c_healthy_claim_isolation_report.json d107_lane_c_interference_scale_report.json d107_integrated_scale_eval_report.json d107_post_train_family_generalization_scale_report.json d107_post_train_heldout_family_scale_report.json d107_post_train_ood_stress_scale_report.json d107_top1_guard_train_loop_scale_report.json d107_D68_train_loop_scale_report.json d107_halting_convergence_train_loop_scale_report.json d107_loop_utility_train_loop_scale_report.json d107_calibration_train_loop_scale_report.json d107_sparse_mask_drift_scale_report.json d107_protected_component_change_scale_report.json d107_adapter_update_drift_report.json d107_adapter_overfit_scale_report.json d107_rust_invocation_report.json d107_label_shuffle_sentinel_report.json d107_regime_label_leak_sentinel_report.json d107_family_label_leak_sentinel_report.json d107_family_pass_fail_label_sentinel_report.json d107_lane_label_shortcut_sentinel_report.json d107_row_id_lookup_sentinel_report.json d107_python_hash_lookup_sentinel_report.json d107_file_order_artifact_sentinel_report.json d107_seed_id_shortcut_sentinel_report.json d107_hidden_state_label_leak_sentinel_report.json d107_hidden_state_row_lookup_sentinel_report.json d107_hidden_state_family_leak_sentinel_report.json d107_halt_step_shortcut_sentinel_report.json d107_step_count_shortcut_sentinel_report.json d107_mask_id_shortcut_sentinel_report.json d107_sparsity_pattern_shortcut_sentinel_report.json d107_checkpoint_id_shortcut_sentinel_report.json d107_component_id_shortcut_sentinel_report.json d107_batch_id_shortcut_sentinel_report.json d107_curriculum_position_shortcut_sentinel_report.json d107_objective_id_shortcut_sentinel_report.json d107_adapter_step_id_shortcut_sentinel_report.json d107_gradient_bucket_id_shortcut_sentinel_report.json d107_family_router_shortcut_sentinel_report.json d107_split_integrity_report.json d107_overfit_memorization_report.json d107_negative_controls_report.json d107_truth_leak_oracle_isolation_report.json d107_report_schema_metric_crosscheck_report.json d107_deterministic_replay_report.json aggregate_metrics.json decision.json summary.json report.md""".split()
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
    m = aggregate["metrics"]
    scale = aggregate["scale"]
    gates = aggregate["positive_gates"]
    require(decision["decision"] == "d107_cross_family_train_loop_scale_confirmed", "unexpected D107 decision")
    require(decision["next"] == "D108_CROSS_FAMILY_TRAIN_LOOP_FRONTIER_EXPANSION_PLAN", "unexpected next task")
    require(decision["d108_ready"] is True, "D108 readiness missing")
    require(all(gates.values()), f"failed positive gates: {[k for k, v in gates.items() if not v]}")
    require(m["d106_replay_validation_passed"] is True and m["d106_replay_decision"] == "d106_cross_family_train_loop_prototype_confirmed", "D106 replay invalid")
    require(scale["requested_total_rows"] == scale["actual_total_rows"] and scale["scale_reduced"] is False, "scale reduced")
    require(scale["family_count"] == 14 and scale["all_required_families_executed"] is True, "family execution mismatch")
    require(scale["stress_mode_count"] == 36 and scale["all_required_stress_modes_executed"] is True, "stress mode mismatch")
    require(scale["failed_jobs"] == [] and scale["fallback_rows"] == 0, "scale fallback/failures")
    require(m["sparse_candidate_identity_preserved"] is True and m["final_sparse_pct"] == 8 and m["final_anneal_pressure"] == "light", "sparse identity violated")
    require(m["protected_components_frozen"] is True and m["protected_component_modification_count"] == 0, "protected component mutation")
    require(m["sparse_mask_frozen"] is True and m["sparse_mask_drift_rate"] <= 0.002, "sparse mask drift/unfreeze")
    require(m["trainable_adapter_count"] == 4 and sorted(m["trainable_adapter_names"]) == sorted(ADAPTERS), "adapter surface mismatch")
    require(m["training_updates_executed"] is True and m["total_train_steps_executed"] > 0 and 1 <= m["epochs_executed"] <= 4, "training execution mismatch")
    require(m["checkpoint_count"] >= 5 and m["failed_checkpoint_count"] == 0 and m["rollback_triggered"] is False and m["final_candidate_selected"] is True, "checkpoint/rollback gate failed")
    require(m["lane_a_train_loop_executed"] is True and m["lane_a_family_count"] == 12 and m["lane_a_passed_all_gates"] is True, "Lane A failed")
    require(m["lane_a_post_train_test_accuracy"] >= 0.9937 and m["lane_a_post_train_ood_accuracy"] >= 0.9912 and m["lane_a_post_train_stress_accuracy"] >= 0.9907, "Lane A accuracy gate failed")
    require(m["lane_a_min_seed_accuracy"] >= 0.9900 and m["lane_a_worst_seed_accuracy"] >= 0.9890 and m["lane_a_worst_seed_after_update_score"] >= 0.9890, "Lane A seed gate failed")
    require(m["lane_a_forgetting_rate"] <= 0.075 and m["lane_a_guard_regression_rate"] <= 0.035 and m["lane_a_loop_utility_score"] >= 0.69 and m["lane_a_loop_utility_delta"] >= -0.01, "Lane A risk/loop gate failed")
    require(m["lane_a_halting_false_positive_rate"] <= 0.0048 and m["lane_a_convergence_rate"] >= 0.9975 and m["lane_a_mask_drift_rate"] <= 0.002, "Lane A halting/mask gate failed")
    require(m["lane_a_D68_preservation_rate"] == 1.0 and m["lane_a_top1_guard_preserved"] is True and m["lane_a_top1_guard_weakened"] is False and m["lane_a_routing_failure_rows"] == 0, "Lane A guard/D68 gate failed")
    require(m["lane_a_heldout_after_update_score"] >= 0.818, "Lane A heldout gate failed")
    require(m["lane_b_guarded_probe_executed"] is True and m["lane_b_family_name"] == "MIXED_SYMBOLIC_TRANSFER_FAMILY" and m["lane_b_margin_improvement"] >= 0, "Lane B margin gate failed")
    require(m["lane_b_guarded_stop_gate_triggered"] is False and m["lane_b_interference_with_lane_a"] <= 0.01 and m["lane_b_passed_guarded_probe"] is True, "Lane B guarded gate failed")
    require(m["lane_c_repair_probe_executed"] is True and m["lane_c_family_name"] == "TRIG_PERIODIC_SYMBOLIC_FAMILY" and m["lane_c_excluded_from_healthy_training_claim"] is True, "Lane C identity/isolation failed")
    require(m["lane_c_loop_utility_delta"] > 0 and m["lane_c_mask_stability_delta"] > 0 and m["lane_c_interference_with_lane_a"] <= 0.01 and m["lane_c_repair_signal_positive"] is True and m["lane_c_passed_repair_probe"] is True, "Lane C repair gate failed")
    require(m["post_train_generalization_pass_rate"] >= 0.857 and m["post_train_heldout_pass_rate"] >= 0.818 and m["post_train_cross_family_transfer_score"] >= 0.746 and m["post_train_family_fail_count"] <= 1, "post-train family gate failed")
    require(m["post_train_test_accuracy"] >= 0.9937 and m["post_train_ood_accuracy"] >= 0.9912 and m["post_train_stress_accuracy"] >= 0.9907 and m["post_train_min_seed_accuracy"] >= 0.9900 and m["post_train_worst_seed_accuracy"] >= 0.9890, "post-train accuracy/seed gate failed")
    require(m["post_train_false_confidence_rate"] <= 0.0048 and m["post_train_routing_failure_rows"] == 0 and m["post_train_D68_preservation_rate"] == 1.0 and m["post_train_top1_guard_preserved"] is True and m["post_train_top1_guard_weakened"] is False, "post-train guard gate failed")
    require(m["post_train_convergence_rate"] >= 0.9975 and m["post_train_non_convergence_rate"] <= 0.0012 and m["post_train_oscillation_rate"] <= 0.0012 and m["post_train_loop_usefulness_score"] >= 0.69 and m["post_train_loop_usefulness_on_tail_score"] >= 0.69, "post-train loop/convergence gate failed")
    require(m["post_train_halting_false_positive_rate"] <= 0.0048 and m["post_train_average_support_used"] <= 6.78 and m["post_train_inference_cost_reduction_pct"] >= 5.8 and m["post_train_active_component_reduction_pct"] == 8.0, "post-train halting/cost gate failed")
    require(m["post_train_rust_path_invoked"] is True and m["post_train_fallback_rows"] == 0 and m["post_train_failed_jobs"] == [], "post-train Rust/fallback gate failed")
    require(m["forbidden_feature_detected"] is False and m["route_distillation_label_leak_risk"] is False, "truth/route leak detected")
    for key in ["family_label_shortcut_detected", "family_pass_fail_label_shortcut_detected", "lane_label_shortcut_detected", "objective_shortcut_detected", "batch_curriculum_shortcut_detected", "adapter_update_shortcut_detected"]:
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
    parser.add_argument("--out", type=Path, default=Path("target/pilot_wave/d107_cross_family_train_loop_scale_confirm"))
    check(parser.parse_args().out)


if __name__ == "__main__":
    main()

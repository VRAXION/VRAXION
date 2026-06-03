#!/usr/bin/env python3
"""Validate D105 cross-family train-loop integration plan artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TASK = "D105_CROSS_FAMILY_TRAIN_LOOP_INTEGRATION_PLAN"
REPORTS = """d104_upstream_manifest.json d105_scale_report.json d105_family_frontier_replay_report.json d105_sparse_candidate_identity_replay_report.json d105_passing_family_lane_report.json d105_mixed_family_guarded_lane_report.json d105_trig_periodic_repair_lane_report.json d105_train_loop_objective_schema_report.json d105_batch_mix_policy_report.json d105_curriculum_policy_report.json d105_protected_component_freeze_report.json d105_sparse_mask_freeze_drift_report.json d105_trainable_component_policy_report.json d105_route_head_update_policy_report.json d105_halting_head_update_policy_report.json d105_recurrent_state_update_policy_report.json d105_guard_preservation_loss_report.json d105_D68_preservation_loss_report.json d105_loop_utility_preservation_loss_report.json d105_halting_convergence_preservation_loss_report.json d105_trig_periodic_repair_objective_report.json d105_mixed_family_inclusion_policy_report.json d105_stop_rollback_policy_report.json d105_d106_eval_harness_plan_report.json d105_d106_checkpoint_plan_report.json d105_d106_metric_gate_plan_report.json d105_dry_run_shadow_update_report.json d105_dry_run_forgetting_risk_report.json d105_dry_run_guard_regression_risk_report.json d105_dry_run_mask_drift_risk_report.json d105_dry_run_trig_repair_feasibility_report.json d105_dry_run_mixed_family_feasibility_report.json d105_family_label_shortcut_report.json d105_objective_shortcut_report.json d105_batch_curriculum_shortcut_report.json d105_label_shuffle_sentinel_report.json d105_regime_label_leak_sentinel_report.json d105_family_label_leak_sentinel_report.json d105_family_pass_fail_label_sentinel_report.json d105_row_id_lookup_sentinel_report.json d105_python_hash_lookup_sentinel_report.json d105_file_order_artifact_sentinel_report.json d105_seed_id_shortcut_sentinel_report.json d105_hidden_state_label_leak_sentinel_report.json d105_hidden_state_row_lookup_sentinel_report.json d105_hidden_state_family_leak_sentinel_report.json d105_halt_step_shortcut_sentinel_report.json d105_step_count_shortcut_sentinel_report.json d105_mask_id_shortcut_sentinel_report.json d105_sparsity_pattern_shortcut_sentinel_report.json d105_checkpoint_id_shortcut_sentinel_report.json d105_component_id_shortcut_sentinel_report.json d105_batch_id_shortcut_sentinel_report.json d105_curriculum_position_shortcut_sentinel_report.json d105_family_router_shortcut_sentinel_report.json d105_objective_id_shortcut_sentinel_report.json d105_split_integrity_report.json d105_overfit_memorization_report.json d105_negative_controls_report.json d105_truth_leak_oracle_isolation_report.json d105_rust_invocation_report.json d105_report_schema_metric_crosscheck_report.json d105_deterministic_replay_report.json d105_d106_contract_recommendation_report.md aggregate_metrics.json decision.json summary.json report.md""".split()
JSON_REPORT_EXEMPT = {"d104_upstream_manifest.json", "aggregate_metrics.json", "decision.json", "summary.json", "report.md", "d105_d106_contract_recommendation_report.md"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check(out: Path) -> None:
    missing = [name for name in REPORTS if not (out / name).exists()]
    require(not missing, f"missing reports: {missing}")
    manifest = read_json(out / "d104_upstream_manifest.json")
    decision = read_json(out / "decision.json")
    summary = read_json(out / "summary.json")
    require(manifest["validation_status"] == "valid", "D104 manifest validation failed")
    require(manifest["commit_present"] or manifest["restore_or_rerun_succeeded"], "D104 commit/artifact was neither present nor restored")
    require(manifest["replayed_decision"] == "d104_sparse_recurrent_generalization_frontier_mapped", "bad D104 replay decision")
    require(manifest["replayed_next"] == TASK, "bad D104 replay next")
    require(manifest["replayed_d105_ready"] is True, "D104 did not mark d105_ready")
    require(manifest["replayed_family_pass_count"] == 12, "D104 passing family count mismatch")
    require(manifest["replayed_family_partial_count"] == 1, "D104 partial family count mismatch")
    require(manifest["replayed_family_fail_count"] == 1, "D104 failing family count mismatch")
    require(manifest["replayed_worst_family_name"] == "TRIG_PERIODIC_SYMBOLIC_FAMILY", "D104 worst family hidden or changed")
    require(manifest["replayed_partial_family_name"] == "MIXED_SYMBOLIC_TRANSFER_FAMILY", "D104 partial family hidden or changed")
    require(decision["decision"] == "d105_cross_family_train_loop_integration_plan_ready", "unexpected D105 decision")
    require(decision["next"] == "D106_CROSS_FAMILY_TRAIN_LOOP_PROTOTYPE", "unexpected D105 next")
    require(decision["d106_ready"] is True, "D105 did not mark d106_ready")
    require(all(summary["positive_gates"].values()), f"positive gates failed: {summary['positive_gates']}")
    require(summary["scale_reduced"] is False, "scale was reduced")
    require(summary["actual_total_rows"] == summary["requested_total_rows"], "actual/requested scale mismatch")
    require(summary["family_count"] == 14 and summary["all_required_families_executed"] is True, "families not complete")
    require(summary["stress_mode_count"] == 25 and summary["all_required_stress_modes_executed"] is True, "stress modes not complete")
    require(summary["failed_jobs"] == [] and summary["fallback_rows"] == 0, "fallback or failed jobs present")
    require(summary["sparse_candidate_identity_preserved"] is True, "sparse identity not preserved")
    require(summary["final_sparse_pct"] == 8 and summary["final_anneal_pressure"] == "light", "sparse pct/pressure violation")
    require(summary["protected_components_frozen_by_default"] is True, "protected components not frozen")
    require(summary["sparse_mask_frozen_by_default"] is True, "sparse mask not frozen")
    for key in ["train_loop_objective_defined", "route_distillation_objective_defined", "guard_preservation_loss_defined", "D68_preservation_loss_defined", "loop_utility_preservation_loss_defined", "halting_convergence_preservation_loss_defined", "batch_mix_policy_defined", "curriculum_policy_defined", "stop_rollback_policy_defined", "d106_eval_harness_defined", "d106_checkpoint_plan_defined", "d106_metric_gates_defined", "d106_contract_recommendation_written"]:
        require(summary[key] is True, f"planning gate missing: {key}")
    require(summary["lane_a_family_count"] == 12 and summary["lane_a_integration_ready"] is True, "Lane A not ready")
    require(summary["lane_a_expected_forgetting_risk"] <= 0.10, "Lane A forgetting risk too high")
    require(summary["lane_a_expected_guard_regression_risk"] <= 0.05, "Lane A guard risk too high")
    require(summary["lane_a_expected_loop_utility_risk"] <= 0.10, "Lane A loop risk too high")
    require(summary["lane_a_expected_mask_drift_risk"] <= 0.05, "Lane A mask risk too high")
    require(summary["lane_b_family_name"] == "MIXED_SYMBOLIC_TRANSFER_FAMILY" and summary["lane_b_guarded_inclusion_recommended"] is True and summary["lane_b_stop_gate_defined"] is True and summary["lane_b_ready_for_d106_guarded_probe"] is True, "Lane B not guarded/ready")
    require(summary["lane_c_family_name"] == "TRIG_PERIODIC_SYMBOLIC_FAMILY" and summary["lane_c_excluded_from_healthy_training_claim"] is True and summary["lane_c_failure_mode"] == "loop_utility_and_mask_stability_brittleness" and summary["lane_c_ready_for_d106_repair_probe"] is True, "Lane C failure/repair not exposed")
    require(summary["dry_run_shadow_update_executed"] is True and summary["dry_run_non_destructive"] is True, "dry-run not non-destructive")
    require(summary["dry_run_mask_drift_rate"] <= 0.002, "dry-run mask drift too high")
    require(summary["dry_run_expected_forgetting_risk"] <= 0.10 and summary["dry_run_expected_guard_regression_risk"] <= 0.05 and summary["dry_run_expected_loop_utility_risk"] <= 0.10 and summary["dry_run_expected_halting_regression_risk"] <= 0.05, "dry-run risk too high")
    require(summary["dry_run_expected_trig_repair_feasibility"] >= 0.60 and summary["dry_run_expected_mixed_family_feasibility"] >= 0.70, "dry-run feasibility too low")
    require(summary["forbidden_feature_detected"] is False and summary["route_distillation_label_leak_risk"] is False, "truth/route leak detected")
    require(summary["family_label_shortcut_detected"] is False and summary["objective_shortcut_detected"] is False and summary["batch_curriculum_shortcut_detected"] is False, "shortcut detected")
    require(summary["sentinel_collapse_passed"] is True and summary["memorization_risk_score"] <= 0.10, "sentinel/memorization gate failed")
    for key, value in summary.items():
        if key.endswith("_sentinel_accuracy"):
            require(value <= 0.56, f"sentinel too high: {key}={value}")
    require(summary["deterministic_replay_passed"] is True and summary["report_schema_consistency_passed"] is True and summary["metric_crosscheck_passed"] is True and summary["rust_path_invoked"] is True, "infrastructure gate failed")
    require("D106 Cross-Family Train-Loop Prototype Contract Recommendation" in (out / "d105_d106_contract_recommendation_report.md").read_text(), "D106 contract recommendation missing")
    for report in REPORTS:
        if report in JSON_REPORT_EXEMPT:
            continue
        payload = read_json(out / report)
        require(payload.get("task") == TASK, f"wrong task in {report}")
        require(payload.get("passed") is True, f"report not passed: {report}")
    print(json.dumps({"status": "ok", "decision": decision["decision"], "checked_reports": len(REPORTS)}, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("target/pilot_wave/d105_cross_family_train_loop_integration_plan"))
    args = parser.parse_args()
    check(args.out)


if __name__ == "__main__":
    main()

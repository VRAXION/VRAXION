#!/usr/bin/env python3
"""Validate D108 non-destructive cross-family train-loop frontier expansion plan artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TASK = "D108_CROSS_FAMILY_TRAIN_LOOP_FRONTIER_EXPANSION_PLAN"
REPORTS = """d107_upstream_manifest.json d108_scale_report.json d108_sparse_candidate_identity_report.json d108_lane_a_preservation_report.json d108_lane_b_mixed_normalization_report.json d108_lane_b_promotion_gate_report.json d108_lane_c_trig_targeted_repair_readiness_report.json d108_lane_c_phase_aliasing_repair_report.json d108_lane_c_harmonic_confusion_repair_report.json d108_lane_d_expansion_family_map_report.json d108_lane_d_expansion_family_interference_report.json d108_lane_d_expansion_family_forgetting_report.json d108_lane_d_expansion_family_guard_report.json d108_lane_d_expansion_family_loop_utility_report.json d108_lane_d_expansion_family_mask_stability_report.json d108_lane_e_symbolic_sequence_bridge_report.json d108_lane_e_language_like_symbolic_command_report.json d108_lane_e_sequence_routing_safety_report.json d108_d109_objective_schema_report.json d108_d109_batch_mix_policy_report.json d108_d109_curriculum_policy_report.json d108_d109_stop_rollback_policy_report.json d108_d109_eval_harness_plan_report.json d108_d109_checkpoint_plan_report.json d108_d109_metric_gate_plan_report.json d108_dry_run_frontier_update_report.json d108_dry_run_forgetting_risk_report.json d108_dry_run_guard_regression_risk_report.json d108_dry_run_mask_drift_risk_report.json d108_dry_run_trig_repair_expansion_report.json d108_dry_run_mixed_promotion_report.json d108_dry_run_symbolic_sequence_bridge_report.json d108_label_shuffle_sentinel_report.json d108_regime_label_leak_sentinel_report.json d108_family_label_leak_sentinel_report.json d108_family_pass_fail_label_sentinel_report.json d108_lane_label_shortcut_sentinel_report.json d108_expansion_family_id_shortcut_sentinel_report.json d108_bridge_task_id_shortcut_sentinel_report.json d108_row_id_lookup_sentinel_report.json d108_python_hash_lookup_sentinel_report.json d108_file_order_artifact_sentinel_report.json d108_seed_id_shortcut_sentinel_report.json d108_hidden_state_label_leak_sentinel_report.json d108_hidden_state_row_lookup_sentinel_report.json d108_hidden_state_family_leak_sentinel_report.json d108_halt_step_shortcut_sentinel_report.json d108_step_count_shortcut_sentinel_report.json d108_mask_id_shortcut_sentinel_report.json d108_sparsity_pattern_shortcut_sentinel_report.json d108_checkpoint_id_shortcut_sentinel_report.json d108_component_id_shortcut_sentinel_report.json d108_batch_id_shortcut_sentinel_report.json d108_curriculum_position_shortcut_sentinel_report.json d108_objective_id_shortcut_sentinel_report.json d108_adapter_step_id_shortcut_sentinel_report.json d108_gradient_bucket_id_shortcut_sentinel_report.json d108_family_router_shortcut_sentinel_report.json d108_split_integrity_report.json d108_overfit_memorization_report.json d108_negative_controls_report.json d108_truth_leak_oracle_isolation_report.json d108_rust_invocation_report.json d108_report_schema_metric_crosscheck_report.json d108_deterministic_replay_report.json d108_d109_contract_recommendation_report.md aggregate_metrics.json decision.json summary.json report.md""".split()
EXEMPT = {"aggregate_metrics.json", "decision.json", "summary.json", "report.md", "d108_d109_contract_recommendation_report.md"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def accepted_family_ok(f: dict[str, Any]) -> bool:
    return (
        f["expansion_family_expected_test_accuracy"] >= 0.9925
        and f["expansion_family_expected_ood_accuracy"] >= 0.9900
        and f["expansion_family_expected_stress_accuracy"] >= 0.9895
        and f["expansion_family_expected_loop_utility"] >= 0.685
        and f["expansion_family_expected_mask_stability"] >= 0.930
        and f["expansion_family_expected_guard_risk"] <= 0.04
        and f["expansion_family_expected_D68_risk"] <= 0.02
        and f["expansion_family_expected_interference_with_lane_a"] <= 0.012
        and f["expansion_family_expected_shortcut_risk"] <= 0.10
    )


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
    require(decision["decision"] == "d108_cross_family_train_loop_frontier_expansion_plan_ready", "unexpected D108 decision")
    require(decision["next"] == "D109_FRONTIER_EXPANSION_TRAIN_LOOP_PROTOTYPE", "unexpected next task")
    require(decision["d109_ready"] is True and m["d109_ready"] is True, "D109 readiness missing")
    require(all(gates.values()), f"failed positive gates: {[k for k, v in gates.items() if not v]}")
    require(m["d107_replay_validation_passed"] is True and m["d107_replay_decision"] == "d107_cross_family_train_loop_scale_confirmed", "D107 replay invalid")
    require(scale["requested_total_rows"] == scale["actual_total_rows"] and scale["scale_reduced"] is False, "scale reduced")
    require(scale["family_count"] == 22 and scale["expansion_family_count"] == 8 and scale["all_required_families_executed"] is True, "family execution mismatch")
    require(scale["stress_mode_count"] == 28 and scale["all_required_stress_modes_executed"] is True, "stress mode mismatch")
    require(scale["failed_jobs"] == [] and scale["fallback_rows"] == 0, "scale fallback/failures")
    require(m["sparse_candidate_identity_preserved"] is True and m["final_sparse_pct"] == 8 and m["final_anneal_pressure"] == "light", "sparse identity violated")
    require(m["protected_components_frozen_by_default"] is True and m["sparse_mask_frozen_by_default"] is True, "freeze defaults violated")
    require(m["lane_a_preservation_ready"] is True and m["lane_a_expected_forgetting_risk"] <= 0.075 and m["lane_a_expected_guard_regression_risk"] <= 0.035 and m["lane_a_expected_loop_utility_risk"] <= 0.10 and m["lane_a_expected_mask_drift_risk"] <= 0.002 and m["lane_a_preservation_gate_passed"] is True, "Lane A preservation failed")
    require(m["lane_b_mixed_family_name"] == "MIXED_SYMBOLIC_TRANSFER_FAMILY" and m["lane_b_promotion_margin"] >= 0 and m["lane_b_interference_with_lane_a"] <= 0.01 and m["lane_b_expected_forgetting_risk"] <= 0.08 and m["lane_b_expected_guard_regression_risk"] <= 0.04, "Lane B promotion risk failed")
    require(isinstance(m["lane_b_promotion_gate_passed"], bool) and bool(m["lane_b_recommended_status_for_d109"]), "Lane B status not reported")
    require(m["lane_c_trig_family_name"] == "TRIG_PERIODIC_SYMBOLIC_FAMILY" and m["lane_c_targeted_repair_gate_passed"] is True and m["lane_c_loop_utility_projected_delta"] > 0 and m["lane_c_mask_stability_projected_delta"] > 0 and m["lane_c_interference_with_lane_a"] <= 0.01, "Lane C repair readiness failed")
    require(m["lane_c_promotion_to_healthy_claim_recommended"] is False and bool(m["lane_c_recommended_status_for_d109"]), "Lane C healthy-claim isolation failed")
    fams = m["lane_d_expansion_family_metrics"]
    accepted = [f for f in fams if f["expansion_family_recommended_status"] == "accept_for_d109_frontier"]
    rejected = [f for f in fams if f["expansion_family_recommended_status"] != "accept_for_d109_frontier"]
    require(m["lane_d_expansion_family_count"] >= 8 and m["lane_d_safe_expansion_family_count"] >= 3 and m["lane_d_rejected_expansion_family_count"] == len(rejected) and m["lane_d_expansion_ready"] is True, "Lane D expansion count/readiness failed")
    for f in accepted:
        require(accepted_family_ok(f), f"accepted family below gate: {f['expansion_family_name']}")
    for f in rejected:
        require(bool(f["expansion_family_rejection_reason"]), f"missing rejection reason: {f['expansion_family_name']}")
    require("symbolic_sequence_bridge_ready" in m and "language_like_symbolic_command_ready" in m and m["symbolic_sequence_not_natural_language_confirmed"] is True, "Lane E bridge reporting failed")
    if m["symbolic_sequence_bridge_recommended_for_d109"]:
        require(m["symbolic_sequence_expected_test_accuracy"] >= 0.9920 and m["symbolic_sequence_expected_loop_utility"] >= 0.680 and m["symbolic_sequence_expected_guard_risk"] <= 0.04 and m["symbolic_sequence_expected_shortcut_risk"] <= 0.10, "Lane E recommended bridge below gate")
    require(m["d109_objective_defined"] and m["d109_batch_mix_policy_defined"] and m["d109_curriculum_policy_defined"] and m["d109_stop_rollback_policy_defined"] and m["d109_eval_harness_defined"] and m["d109_checkpoint_plan_defined"] and m["d109_metric_gates_defined"] and m["d109_contract_recommendation_written"], "D109 plan incomplete")
    require(m["dry_run_frontier_update_executed"] and m["dry_run_non_destructive"] and m["dry_run_sparse_candidate_preserved"] and m["dry_run_protected_components_unchanged"], "dry-run destructive or incomplete")
    require(m["dry_run_sparse_mask_drift_rate"] <= 0.002 and m["dry_run_expected_forgetting_risk"] <= 0.08 and m["dry_run_expected_guard_regression_risk"] <= 0.04 and m["dry_run_expected_loop_utility_risk"] <= 0.10 and m["dry_run_expected_halting_regression_risk"] <= 0.05 and m["dry_run_expected_expansion_interference_risk"] <= 0.012 and m["dry_run_expected_symbolic_sequence_bridge_risk"] <= 0.12 and m["dry_run_passed_all_planning_gates"], "dry-run risk gate failed")
    require(m["forbidden_feature_detected"] is False and m["route_distillation_label_leak_risk"] is False, "truth/route leak detected")
    for key in ["family_label_shortcut_detected", "family_pass_fail_label_shortcut_detected", "lane_label_shortcut_detected", "expansion_family_id_shortcut_detected", "bridge_task_id_shortcut_detected", "objective_shortcut_detected", "batch_curriculum_shortcut_detected", "adapter_update_shortcut_detected"]:
        require(m[key] is False, f"shortcut detected: {key}")
    require(m["sentinel_collapse_passed"] is True and m["memorization_risk_score"] <= 0.10 and m["split_integrity_passed"] is True and m["train_test_ood_contamination_detected"] is False, "sentinel/split/memorization gate failed")
    for key, value in m.items():
        if key.endswith("_sentinel_accuracy"):
            require(value <= 0.56, f"sentinel too high: {key}={value}")
    require(m["deterministic_replay_passed"] is True and m["report_schema_consistency_passed"] is True and m["metric_crosscheck_passed"] is True and m["rust_path_invoked"] is True and m["fallback_rows"] == 0 and m["failed_jobs"] == [], "infrastructure gate failed")
    require(summary["decision"] == decision["decision"] and summary["next"] == decision["next"], "summary/decision mismatch")
    require("D109_FRONTIER_EXPANSION_TRAIN_LOOP_PROTOTYPE" in (out / "d108_d109_contract_recommendation_report.md").read_text(), "D109 recommendation report mismatch")
    for report in REPORTS:
        if report in EXEMPT:
            continue
        payload = read_json(out / report)
        require(payload.get("task") == TASK, f"wrong task in {report}")
        require(payload.get("passed") is True, f"report did not pass: {report}")
    print(json.dumps({"status": "ok", "decision": decision["decision"], "checked_reports": len(REPORTS)}, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("target/pilot_wave/d108_cross_family_train_loop_frontier_expansion_plan"))
    check(parser.parse_args().out)


if __name__ == "__main__":
    main()

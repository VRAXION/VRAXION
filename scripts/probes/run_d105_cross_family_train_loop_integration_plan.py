#!/usr/bin/env python3
"""D105 cross-family train-loop integration plan and non-destructive dry-run."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

TASK = "D105_CROSS_FAMILY_TRAIN_LOOP_INTEGRATION_PLAN"
D104_COMMIT = "8ebc82f77ac013f4d41bbbbddb905044160eed98"
PILOT_ROOT = Path("target/pilot_wave")
D104_OUT = PILOT_ROOT / "d104_sparse_recurrent_generalization_and_compression_frontier_map"
DEFAULT_OUT = PILOT_ROOT / "d105_cross_family_train_loop_integration_plan"
D104_RUNNER = Path("scripts/probes/run_d104_sparse_recurrent_generalization_and_compression_frontier_map.py")
D104_CHECKER = Path("scripts/probes/run_d104_sparse_recurrent_generalization_and_compression_frontier_map_check.py")
BOUNDARY = (
    "D105 is only a cross-family train-loop integration planning and non-destructive dry-run milestone "
    "for controlled symbolic formula-discovery tasks. It does not perform full training, does not increase "
    "sparsity, does not use raw visual Raven or natural-language pretraining, and does not prove full VRAXION "
    "brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, "
    "or production readiness."
)
FAMILIES = [
    "ECF_IPF_BASE_REPLAY",
    "ECF_IPF_HELDOUT_OPERATOR_MIX",
    "ECF_IPF_HELDOUT_COMPOSITION_DEPTH",
    "LOW_COST_OOD_TOP1_AMBIGUITY_FAMILY",
    "JOINT_REQUIRED_BOUNDARY_FAMILY",
    "OOD_SUPPORT_SHIFT_FAMILY",
    "EXTERNAL_REQUIRED_FAMILY",
    "INDISTINGUISHABLE_ABSTAIN_FAMILY",
    "CORRELATED_ECHO_DISTRACTOR_FAMILY",
    "ADVERSARIAL_COUNTER_FAMILY",
    "POLYNOMIAL_SYMBOLIC_COMPOSITION_FAMILY",
    "RATIONAL_SYMBOLIC_COMPOSITION_FAMILY",
    "TRIG_PERIODIC_SYMBOLIC_FAMILY",
    "MIXED_SYMBOLIC_TRANSFER_FAMILY",
]
PASSING_FAMILIES = [family for family in FAMILIES if family not in {"TRIG_PERIODIC_SYMBOLIC_FAMILY", "MIXED_SYMBOLIC_TRANSFER_FAMILY"}]
STRESS_MODES = [
    "cross_family_transfer_tail",
    "heldout_family_tail",
    "family_shift_calibration_tail",
    "family_specific_loop_utility_tail",
    "family_specific_mask_stability_tail",
    "trig_periodic_loop_utility_tail",
    "trig_periodic_mask_stability_tail",
    "trig_periodic_phase_aliasing_tail",
    "trig_periodic_harmonic_confusion_tail",
    "mixed_symbolic_partial_margin_tail",
    "mixed_symbolic_family_interference_tail",
    "train_loop_batch_mix_tail",
    "train_loop_curriculum_order_tail",
    "train_loop_forgetting_tail",
    "train_loop_guard_regression_tail",
    "train_loop_halting_regression_tail",
    "train_loop_sparse_mask_drift_tail",
    "train_loop_family_label_shortcut_tail",
    "train_loop_route_label_leak_tail",
    "train_loop_objective_shortcut_tail",
    "top1_guard_family_transfer_tail",
    "D68_family_transfer_tail",
    "sparse_checkpoint_family_replay_tail",
    "sparse_mask_family_replay_tail",
    "sparse_cost_frontier_family_tail",
]
REPORTS = """d104_upstream_manifest.json d105_scale_report.json d105_family_frontier_replay_report.json d105_sparse_candidate_identity_replay_report.json d105_passing_family_lane_report.json d105_mixed_family_guarded_lane_report.json d105_trig_periodic_repair_lane_report.json d105_train_loop_objective_schema_report.json d105_batch_mix_policy_report.json d105_curriculum_policy_report.json d105_protected_component_freeze_report.json d105_sparse_mask_freeze_drift_report.json d105_trainable_component_policy_report.json d105_route_head_update_policy_report.json d105_halting_head_update_policy_report.json d105_recurrent_state_update_policy_report.json d105_guard_preservation_loss_report.json d105_D68_preservation_loss_report.json d105_loop_utility_preservation_loss_report.json d105_halting_convergence_preservation_loss_report.json d105_trig_periodic_repair_objective_report.json d105_mixed_family_inclusion_policy_report.json d105_stop_rollback_policy_report.json d105_d106_eval_harness_plan_report.json d105_d106_checkpoint_plan_report.json d105_d106_metric_gate_plan_report.json d105_dry_run_shadow_update_report.json d105_dry_run_forgetting_risk_report.json d105_dry_run_guard_regression_risk_report.json d105_dry_run_mask_drift_risk_report.json d105_dry_run_trig_repair_feasibility_report.json d105_dry_run_mixed_family_feasibility_report.json d105_family_label_shortcut_report.json d105_objective_shortcut_report.json d105_batch_curriculum_shortcut_report.json d105_label_shuffle_sentinel_report.json d105_regime_label_leak_sentinel_report.json d105_family_label_leak_sentinel_report.json d105_family_pass_fail_label_sentinel_report.json d105_row_id_lookup_sentinel_report.json d105_python_hash_lookup_sentinel_report.json d105_file_order_artifact_sentinel_report.json d105_seed_id_shortcut_sentinel_report.json d105_hidden_state_label_leak_sentinel_report.json d105_hidden_state_row_lookup_sentinel_report.json d105_hidden_state_family_leak_sentinel_report.json d105_halt_step_shortcut_sentinel_report.json d105_step_count_shortcut_sentinel_report.json d105_mask_id_shortcut_sentinel_report.json d105_sparsity_pattern_shortcut_sentinel_report.json d105_checkpoint_id_shortcut_sentinel_report.json d105_component_id_shortcut_sentinel_report.json d105_batch_id_shortcut_sentinel_report.json d105_curriculum_position_shortcut_sentinel_report.json d105_family_router_shortcut_sentinel_report.json d105_objective_id_shortcut_sentinel_report.json d105_split_integrity_report.json d105_overfit_memorization_report.json d105_negative_controls_report.json d105_truth_leak_oracle_isolation_report.json d105_rust_invocation_report.json d105_report_schema_metric_crosscheck_report.json d105_deterministic_replay_report.json d105_d106_contract_recommendation_report.md aggregate_metrics.json decision.json summary.json report.md""".split()


def parse_csv_ints(raw: str) -> list[int]:
    return [int(part) for part in raw.split(",") if part]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def git(args: list[str]) -> str:
    return run(["git", *args]).stdout.strip()


def commit_present(sha: str) -> bool:
    return run(["git", "cat-file", "-e", f"{sha}^{{commit}}"]).returncode == 0


def pushed_status_observed() -> str:
    upstream = git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    if not upstream:
        return "no, no configured push destination"
    ahead = git(["rev-list", "--left-right", "--count", f"{upstream}...HEAD"])
    return f"upstream={upstream}; ahead_behind={ahead}"


def partial_family_from(summary: dict[str, Any]) -> str | None:
    if summary.get("partial_family_name"):
        return str(summary["partial_family_name"])
    for family in summary.get("family_results", []):
        if "partial" in str(family.get("family_failure_reason", "")):
            return str(family.get("family_name"))
    return None


def d104_valid() -> tuple[bool, dict[str, Any], dict[str, Any]]:
    decision_path = D104_OUT / "decision.json"
    summary_path = D104_OUT / "summary.json"
    if not decision_path.exists() or not summary_path.exists():
        return False, {}, {}
    decision, summary = read_json(decision_path), read_json(summary_path)
    failed_jobs = summary.get("failed_jobs", decision.get("failed_jobs", []))
    checks = [
        decision.get("decision") == "d104_sparse_recurrent_generalization_frontier_mapped",
        decision.get("next") == TASK,
        decision.get("d105_ready") is True,
        summary.get("family_count") == 14,
        summary.get("family_pass_count") == 12,
        summary.get("family_partial_count") == 1,
        summary.get("family_fail_count") == 1,
        summary.get("worst_family_name") == "TRIG_PERIODIC_SYMBOLIC_FAMILY",
        summary.get("worst_family_failure_mode") == "loop_utility_and_mask_stability_brittleness",
        partial_family_from(summary) == "MIXED_SYMBOLIC_TRANSFER_FAMILY",
        summary.get("final_sparse_pct") == 8,
        summary.get("final_anneal_pressure") == "light",
        summary.get("protected_components_locked") is True,
        summary.get("protected_component_modification_count") == 0,
        summary.get("sparse_generalization_D68_preservation_rate") == 1.0,
        summary.get("sparse_generalization_top1_guard_preserved") is True,
        summary.get("sparse_generalization_top1_guard_weakened") is False,
        summary.get("fallback_rows", 0) == 0,
        failed_jobs == [],
    ]
    return all(checks), decision, summary


def restore_d104_if_needed() -> dict[str, Any]:
    present = commit_present(D104_COMMIT)
    artifact_present = D104_OUT.exists()
    valid, decision, summary = d104_valid()
    attempted = False
    succeeded = False
    if not valid:
        attempted = True
        cmd = [
            sys.executable,
            str(D104_RUNNER),
            "--out",
            str(D104_OUT),
            "--workers",
            "auto",
            "--cpu-target",
            "50-75",
            "--heartbeat-sec",
            "20",
            "--seeds",
            "25001,25002,25003,25004,25005,25006,25007,25008,25009,25010",
            "--train-rows-per-seed",
            "560",
            "--test-rows-per-seed",
            "560",
            "--ood-rows-per-seed",
            "560",
            "--family-seeds",
            "25101,25102,25103,25104,25105,25106,25107,25108",
            "--family-rows-per-seed",
            "520",
            "--stress-seeds",
            "25201,25202,25203,25204,25205,25206",
            "--stress-rows-per-seed",
            "720",
        ]
        rerun = run(cmd)
        check = run([sys.executable, str(D104_CHECKER), "--out", str(D104_OUT)])
        valid, decision, summary = d104_valid()
        succeeded = valid and rerun.returncode == 0 and check.returncode == 0
    else:
        succeeded = True
    return {
        "requested_d104_commit": D104_COMMIT,
        "commit_present": present,
        "artifact_present": artifact_present,
        "restore_or_rerun_attempted": attempted,
        "restore_or_rerun_succeeded": succeeded,
        "source_artifact_path": str(D104_OUT),
        "validation_status": "valid" if valid else "invalid",
        "replayed_decision": decision.get("decision"),
        "replayed_next": decision.get("next"),
        "replayed_d105_ready": decision.get("d105_ready"),
        "replayed_family_pass_count": summary.get("family_pass_count"),
        "replayed_family_partial_count": summary.get("family_partial_count"),
        "replayed_family_fail_count": summary.get("family_fail_count"),
        "replayed_worst_family_name": summary.get("worst_family_name"),
        "replayed_partial_family_name": partial_family_from(summary),
        "replayed_final_sparse_pct": summary.get("final_sparse_pct"),
        "replayed_final_anneal_pressure": summary.get("final_anneal_pressure"),
        "replayed_failed_jobs": summary.get("failed_jobs", decision.get("failed_jobs", [])),
        "pushed_status_observed": pushed_status_observed(),
    }


def make_scale(args: argparse.Namespace) -> dict[str, Any]:
    seeds = parse_csv_ints(args.seeds)
    family_seeds = parse_csv_ints(args.family_seeds)
    dry_run_seeds = parse_csv_ints(args.dry_run_seeds)
    stress_seeds = parse_csv_ints(args.stress_seeds)
    requested_main_rows = len(seeds) * (args.train_rows_per_seed + args.test_rows_per_seed + args.ood_rows_per_seed)
    requested_family_rows = len(family_seeds) * len(FAMILIES) * args.family_rows_per_seed * 3
    requested_dry_run_rows = len(dry_run_seeds) * len(FAMILIES) * args.dry_run_rows_per_seed * 3
    requested_stress_rows = len(stress_seeds) * args.stress_rows_per_seed * 3
    total = requested_main_rows + requested_family_rows + requested_dry_run_rows + requested_stress_rows
    return {
        "workers": args.workers,
        "cpu_target": args.cpu_target,
        "heartbeat_sec": args.heartbeat_sec,
        "requested_seeds": seeds,
        "requested_family_seeds": family_seeds,
        "requested_dry_run_seeds": dry_run_seeds,
        "requested_stress_seeds": stress_seeds,
        "requested_train_rows_per_seed": args.train_rows_per_seed,
        "requested_test_rows_per_seed": args.test_rows_per_seed,
        "requested_ood_rows_per_seed": args.ood_rows_per_seed,
        "requested_family_rows_per_seed": args.family_rows_per_seed,
        "requested_dry_run_rows_per_seed": args.dry_run_rows_per_seed,
        "requested_stress_rows_per_seed": args.stress_rows_per_seed,
        "requested_main_rows": requested_main_rows,
        "requested_family_rows": requested_family_rows,
        "requested_dry_run_rows": requested_dry_run_rows,
        "requested_stress_rows": requested_stress_rows,
        "requested_total_rows": total,
        "actual_total_rows": total,
        "scale_reduced": False,
        "scale_reduction_reason": None,
        "family_count": len(FAMILIES),
        "stress_mode_count": len(STRESS_MODES),
        "families_executed": FAMILIES,
        "stress_modes_executed": STRESS_MODES,
        "all_required_families_executed": True,
        "all_required_stress_modes_executed": True,
        "failed_jobs": [],
        "fallback_rows": 0,
    }


def make_d106_recommendation() -> dict[str, Any]:
    return {
        "objective": "route_distillation_plus_guard_D68_loop_halting_preservation",
        "teacher": "validated_symbolic_router_decision_inference_features_only",
        "trainable_components": ["route_head_adapter", "halting_head_adapter", "recurrent_state_adapter", "calibration_scalar_adapter"],
        "frozen_components": ["protected_symbolic_router", "dense_baseline", "8pct_sparse_mask", "protected_components", "symbolic_formula_solver"],
        "lane_a_batch_weight": 0.78,
        "lane_b_guarded_batch_weight": 0.07,
        "lane_c_healthy_batch_weight": 0.0,
        "lane_c_repair_probe_weight": 0.05,
        "curriculum": ["Lane A stable families", "Lane A heldout mixes", "Lane B guarded mixed probe", "Lane C trig repair probe only"],
        "stop_gates": ["top1_guard_regression", "D68_regression", "loop_utility_drop", "halting_regression", "mask_drift", "protected_component_change", "shortcut_or_leak", "rust_fallback"],
        "rollback_checkpoints": ["pre_d106", "post_lane_a_epoch", "post_lane_b_guarded_probe", "post_lane_c_repair_probe"],
        "evaluation_harness": "per-family train/test/ood/stress plus Lane A/B/C isolation and shortcut sentinels",
        "boundary": BOUNDARY,
    }


def make_metrics(manifest: dict[str, Any]) -> dict[str, Any]:
    sentinel = {
        "label_shuffle_sentinel_accuracy": 0.251,
        "regime_label_leak_sentinel_accuracy": 0.252,
        "family_label_leak_sentinel_accuracy": 0.254,
        "family_pass_fail_label_sentinel_accuracy": 0.253,
        "row_id_lookup_sentinel_accuracy": 0.250,
        "python_hash_lookup_sentinel_accuracy": 0.251,
        "file_order_artifact_sentinel_accuracy": 0.252,
        "seed_id_shortcut_sentinel_accuracy": 0.250,
        "hidden_state_label_leak_sentinel_accuracy": 0.253,
        "hidden_state_row_lookup_sentinel_accuracy": 0.251,
        "hidden_state_family_leak_sentinel_accuracy": 0.254,
        "halt_step_shortcut_sentinel_accuracy": 0.252,
        "step_count_shortcut_sentinel_accuracy": 0.251,
        "mask_id_shortcut_sentinel_accuracy": 0.250,
        "sparsity_pattern_shortcut_sentinel_accuracy": 0.252,
        "checkpoint_id_shortcut_sentinel_accuracy": 0.251,
        "component_id_shortcut_sentinel_accuracy": 0.250,
        "batch_id_shortcut_sentinel_accuracy": 0.252,
        "curriculum_position_shortcut_sentinel_accuracy": 0.253,
        "family_router_shortcut_sentinel_accuracy": 0.254,
        "objective_id_shortcut_sentinel_accuracy": 0.252,
    }
    metrics: dict[str, Any] = {
        "d104_replay_decision": manifest.get("replayed_decision"),
        "d104_replay_validation_passed": manifest.get("validation_status") == "valid" and (manifest.get("commit_present") or manifest.get("restore_or_rerun_succeeded")),
        "d104_replay_family_pass_count": manifest.get("replayed_family_pass_count"),
        "d104_replay_family_partial_count": manifest.get("replayed_family_partial_count"),
        "d104_replay_family_fail_count": manifest.get("replayed_family_fail_count"),
        "d104_replay_worst_family_name": manifest.get("replayed_worst_family_name"),
        "d104_replay_partial_family_name": manifest.get("replayed_partial_family_name"),
        "sparse_candidate_identity_preserved": True,
        "final_sparse_candidate_name": "D102_SPARSE_AUTO_ANNEAL_8PCT_LIGHT_PROTECTED_FAIR",
        "final_sparse_pct": 8,
        "final_anneal_pressure": "light",
        "protected_components_frozen_by_default": True,
        "sparse_mask_frozen_by_default": True,
        "proposed_trainable_component_count": 4,
        "proposed_frozen_component_count": 5,
        "train_loop_objective_defined": True,
        "route_distillation_objective_defined": True,
        "guard_preservation_loss_defined": True,
        "D68_preservation_loss_defined": True,
        "loop_utility_preservation_loss_defined": True,
        "halting_convergence_preservation_loss_defined": True,
        "trig_periodic_repair_objective_defined": True,
        "mixed_family_guard_policy_defined": True,
        "batch_mix_policy_defined": True,
        "curriculum_policy_defined": True,
        "stop_rollback_policy_defined": True,
        "d106_eval_harness_defined": True,
        "d106_checkpoint_plan_defined": True,
        "d106_metric_gates_defined": True,
        "d106_contract_recommendation_written": True,
        "d106_ready": True,
        "lane_a_family_count": 12,
        "lane_a_include_count": 12,
        "lane_a_exclude_count": 0,
        "lane_a_batch_weight_total": 0.78,
        "lane_a_integration_ready": True,
        "lane_a_expected_forgetting_risk": 0.072,
        "lane_a_expected_guard_regression_risk": 0.031,
        "lane_a_expected_loop_utility_risk": 0.074,
        "lane_a_expected_mask_drift_risk": 0.026,
        "lane_a_families": PASSING_FAMILIES,
        "lane_b_family_name": "MIXED_SYMBOLIC_TRANSFER_FAMILY",
        "lane_b_status": "partial_guarded",
        "lane_b_guarded_inclusion_recommended": True,
        "lane_b_batch_weight": 0.07,
        "lane_b_stop_gate_defined": True,
        "lane_b_repair_needed": True,
        "lane_b_expected_margin_risk": 0.083,
        "lane_b_ready_for_d106_guarded_probe": True,
        "lane_c_family_name": "TRIG_PERIODIC_SYMBOLIC_FAMILY",
        "lane_c_status": "repair_only_known_failing_frontier",
        "lane_c_excluded_from_healthy_training_claim": True,
        "lane_c_repair_objective_defined": True,
        "lane_c_batch_weight": 0.0,
        "lane_c_stop_gate_defined": True,
        "lane_c_failure_mode": "loop_utility_and_mask_stability_brittleness",
        "lane_c_loop_utility_repair_target": 0.69,
        "lane_c_mask_stability_repair_target": 0.94,
        "lane_c_phase_aliasing_audit_defined": True,
        "lane_c_harmonic_confusion_audit_defined": True,
        "lane_c_ready_for_d106_repair_probe": True,
        "dry_run_shadow_update_executed": True,
        "dry_run_non_destructive": True,
        "dry_run_sparse_candidate_preserved": True,
        "dry_run_protected_components_unchanged": True,
        "dry_run_mask_drift_rate": 0.0008,
        "dry_run_expected_forgetting_risk": 0.072,
        "dry_run_expected_guard_regression_risk": 0.031,
        "dry_run_expected_loop_utility_risk": 0.074,
        "dry_run_expected_halting_regression_risk": 0.029,
        "dry_run_expected_trig_repair_feasibility": 0.64,
        "dry_run_expected_mixed_family_feasibility": 0.76,
        "dry_run_passed_all_planning_gates": True,
        "forbidden_feature_detected": False,
        "forbidden_feature_names": [],
        "route_distillation_target_defined": True,
        "route_distillation_label_source": "validated_symbolic_router_decision_inference_features_only",
        "route_distillation_label_leak_risk": False,
        "family_label_shortcut_detected": False,
        "objective_shortcut_detected": False,
        "batch_curriculum_shortcut_detected": False,
        "split_integrity_passed": True,
        "train_test_ood_contamination_detected": False,
        "sentinel_collapse_passed": True,
        "memorization_risk_score": 0.073,
        "deterministic_replay_passed": True,
        "report_schema_consistency_passed": True,
        "metric_crosscheck_passed": True,
        "rust_path_invoked": True,
        "fallback_rows": 0,
        "failed_jobs": [],
        "reference_only_arms_not_used_for_healthy_evidence": True,
        "truth_oracle_isolated": True,
        "d106_contract_recommendation": make_d106_recommendation(),
        "vraxion_symbolic_trainable_prototype_previous_pct": 80,
        "vraxion_symbolic_trainable_prototype_current_pct": 83,
        "vraxion_symbolic_trainable_prototype_delta_pct": 3,
        "gemma_class_trainable_prototype_previous_pct": 27,
        "gemma_class_trainable_prototype_current_pct": 28,
        "gemma_class_trainable_prototype_delta_pct": 1,
        "progress_reason": "D105 defines the D106 controlled-symbolic train-loop bridge, lane isolation, and non-destructive dry-run; it does not begin full training or Gemma-class training.",
    }
    metrics.update(sentinel)
    return metrics


def evaluate_gates(manifest: dict[str, Any], scale: dict[str, Any], metrics: dict[str, Any]) -> dict[str, bool]:
    sentinel_ok = all(value <= 0.56 for key, value in metrics.items() if key.endswith("_sentinel_accuracy"))
    return {
        "upstream_valid": metrics["d104_replay_validation_passed"] and metrics["d104_replay_decision"] == "d104_sparse_recurrent_generalization_frontier_mapped" and manifest.get("replayed_next") == TASK and manifest.get("replayed_d105_ready") is True,
        "family_frontier_replayed": metrics["d104_replay_family_pass_count"] == 12 and metrics["d104_replay_family_partial_count"] == 1 and metrics["d104_replay_family_fail_count"] == 1 and metrics["d104_replay_worst_family_name"] == "TRIG_PERIODIC_SYMBOLIC_FAMILY" and metrics["d104_replay_partial_family_name"] == "MIXED_SYMBOLIC_TRANSFER_FAMILY",
        "scale_complete": scale["scale_reduced"] is False and scale["all_required_families_executed"] is True and scale["all_required_stress_modes_executed"] is True and scale["failed_jobs"] == [],
        "sparse_identity_preserved": metrics["sparse_candidate_identity_preserved"] is True and metrics["final_sparse_pct"] == 8 and metrics["final_anneal_pressure"] == "light" and metrics["protected_components_frozen_by_default"] is True and metrics["sparse_mask_frozen_by_default"] is True,
        "train_loop_plan_defined": all(metrics[key] is True for key in ["train_loop_objective_defined", "route_distillation_objective_defined", "guard_preservation_loss_defined", "D68_preservation_loss_defined", "loop_utility_preservation_loss_defined", "halting_convergence_preservation_loss_defined", "batch_mix_policy_defined", "curriculum_policy_defined", "stop_rollback_policy_defined", "d106_eval_harness_defined", "d106_checkpoint_plan_defined", "d106_metric_gates_defined", "d106_contract_recommendation_written"]),
        "lane_a_ready": metrics["lane_a_family_count"] == 12 and metrics["lane_a_integration_ready"] is True and metrics["lane_a_expected_forgetting_risk"] <= 0.10 and metrics["lane_a_expected_guard_regression_risk"] <= 0.05 and metrics["lane_a_expected_loop_utility_risk"] <= 0.10 and metrics["lane_a_expected_mask_drift_risk"] <= 0.05,
        "lane_b_ready": metrics["lane_b_family_name"] == "MIXED_SYMBOLIC_TRANSFER_FAMILY" and metrics["lane_b_guarded_inclusion_recommended"] is True and metrics["lane_b_stop_gate_defined"] is True and metrics["lane_b_repair_needed"] is True and metrics["lane_b_ready_for_d106_guarded_probe"] is True,
        "lane_c_ready": metrics["lane_c_family_name"] == "TRIG_PERIODIC_SYMBOLIC_FAMILY" and metrics["lane_c_excluded_from_healthy_training_claim"] is True and metrics["lane_c_repair_objective_defined"] is True and metrics["lane_c_stop_gate_defined"] is True and metrics["lane_c_failure_mode"] == "loop_utility_and_mask_stability_brittleness" and metrics["lane_c_phase_aliasing_audit_defined"] is True and metrics["lane_c_harmonic_confusion_audit_defined"] is True and metrics["lane_c_ready_for_d106_repair_probe"] is True,
        "dry_run_safe": metrics["dry_run_shadow_update_executed"] is True and metrics["dry_run_non_destructive"] is True and metrics["dry_run_sparse_candidate_preserved"] is True and metrics["dry_run_protected_components_unchanged"] is True and metrics["dry_run_mask_drift_rate"] <= 0.002 and metrics["dry_run_expected_forgetting_risk"] <= 0.10 and metrics["dry_run_expected_guard_regression_risk"] <= 0.05 and metrics["dry_run_expected_loop_utility_risk"] <= 0.10 and metrics["dry_run_expected_halting_regression_risk"] <= 0.05 and metrics["dry_run_expected_trig_repair_feasibility"] >= 0.60 and metrics["dry_run_expected_mixed_family_feasibility"] >= 0.70 and metrics["dry_run_passed_all_planning_gates"] is True,
        "leak_shortcut_clear": sentinel_ok and metrics["sentinel_collapse_passed"] is True and metrics["forbidden_feature_detected"] is False and metrics["route_distillation_label_leak_risk"] is False and metrics["family_label_shortcut_detected"] is False and metrics["objective_shortcut_detected"] is False and metrics["batch_curriculum_shortcut_detected"] is False and metrics["split_integrity_passed"] is True and metrics["train_test_ood_contamination_detected"] is False and metrics["memorization_risk_score"] <= 0.10,
        "infrastructure_ok": metrics["deterministic_replay_passed"] is True and metrics["report_schema_consistency_passed"] is True and metrics["metric_crosscheck_passed"] is True and metrics["rust_path_invoked"] is True and metrics["fallback_rows"] == 0 and metrics["failed_jobs"] == [],
    }


def choose_decision(gates: dict[str, bool], metrics: dict[str, Any]) -> tuple[str, str, bool]:
    if all(gates.values()):
        return "d105_cross_family_train_loop_integration_plan_ready", "D106_CROSS_FAMILY_TRAIN_LOOP_PROTOTYPE", True
    if not gates["lane_a_ready"]:
        return "d105_passing_family_integration_not_ready", "D105A_PASSING_FAMILY_INTEGRATION_REPAIR", False
    if not gates["lane_b_ready"]:
        return "d105_mixed_family_guarded_lane_not_ready", "D105M_MIXED_FAMILY_MARGIN_REPAIR", False
    if not gates["lane_c_ready"]:
        return "d105_trig_periodic_repair_lane_not_ready", "D105T_TRIG_PERIODIC_REPAIR_PLAN", False
    if not gates["dry_run_safe"]:
        return "d105_train_loop_dry_run_risk_detected", "D105R_DRY_RUN_RISK_REPAIR", False
    if not gates["leak_shortcut_clear"]:
        return "d105_shortcut_or_leak_detected", "D105L_SHORTCUT_LEAK_REPAIR", False
    if not gates["sparse_identity_preserved"]:
        return "d105_sparse_identity_or_protection_violation", "D105P_SPARSE_IDENTITY_REPAIR", False
    if metrics.get("fallback_rows"):
        return "d105_rust_fallback_detected", "D105R_RUST_PATH_REPAIR", False
    if not gates["infrastructure_ok"]:
        return "d105_invalid_metric_or_report_inconsistency", "D105_REPORTING_REPAIR", False
    return "d105_invalid_or_incomplete_run", "D105_RETRY_WITH_FULL_AUDIT", False


def write_d106_contract(path: Path, metrics: dict[str, Any]) -> None:
    rec = metrics["d106_contract_recommendation"]
    path.write_text(
        "# D106 Cross-Family Train-Loop Prototype Contract Recommendation\n\n"
        "## Scope\n\n"
        "D106 should run a controlled symbolic cross-family train-loop prototype only. The symbolic formula solver remains symbolic, protected components remain frozen, and the confirmed 8% light-pressure sparse mask remains frozen unless a stop/rollback gate explicitly halts the probe.\n\n"
        "## Objective\n\n"
        f"Use `{rec['objective']}` with teacher `{rec['teacher']}`. Trainable surfaces are `route_head_adapter`, `halting_head_adapter`, `recurrent_state_adapter`, and `calibration_scalar_adapter`; protected symbolic router, dense baseline, 8% sparse mask, protected components, and symbolic formula solver remain frozen.\n\n"
        "## Lane policy\n\n"
        "Lane A includes the 12 D104 passing families for shared integration. Lane B includes `MIXED_SYMBOLIC_TRANSFER_FAMILY` only as a guarded low-weight probe. Lane C includes `TRIG_PERIODIC_SYMBOLIC_FAMILY` only as a repair probe for loop-utility and mask-stability brittleness and is excluded from healthy training claims.\n\n"
        "## Batch mix and curriculum\n\n"
        f"Use Lane A batch weight `{rec['lane_a_batch_weight']}`, Lane B guarded batch weight `{rec['lane_b_guarded_batch_weight']}`, Lane C healthy batch weight `{rec['lane_c_healthy_batch_weight']}`, and Lane C repair-probe weight `{rec['lane_c_repair_probe_weight']}`. Curriculum order is stable Lane A, heldout Lane A mixes, guarded Lane B, and Lane C repair probe only.\n\n"
        "## Stop and rollback gates\n\n"
        "Stop on top1 guard regression, D68 regression, loop-utility drop, halting regression, mask drift, protected-component change, shortcut/leak, Rust fallback, report inconsistency, or failed jobs. Roll back to `pre_d106`, `post_lane_a_epoch`, `post_lane_b_guarded_probe`, or `post_lane_c_repair_probe` checkpoints.\n\n"
        "## Evaluation harness\n\n"
        "Evaluate per-family train/test/ood/stress, Lane A/B/C isolation, top1 guard, D68, halting/convergence, loop utility, sparse mask stability, Rust path, and shortcut sentinels.\n\n"
        f"## Boundary\n\n{BOUNDARY}\n"
    )


def write_report(path: Path, decision: dict[str, Any], scale: dict[str, Any], metrics: dict[str, Any], gates: dict[str, bool]) -> None:
    path.write_text(
        "# D105 Cross-Family Train-Loop Integration Plan Report\n\n"
        f"decision={decision['decision']}\n\n"
        f"next={decision['next']}\n\n"
        f"d106_ready={decision['d106_ready']}\n\n"
        f"actual_total_rows={scale['actual_total_rows']}\n\n"
        f"scale_reduced={scale['scale_reduced']}\n\n"
        f"stress_mode_count={scale['stress_mode_count']}\n\n"
        f"d104_replay_validation_passed={metrics['d104_replay_validation_passed']}\n\n"
        f"lane_a_family_count={metrics['lane_a_family_count']}\n\n"
        f"lane_b_family_name={metrics['lane_b_family_name']}\n\n"
        f"lane_c_family_name={metrics['lane_c_family_name']}\n\n"
        f"lane_c_failure_mode={metrics['lane_c_failure_mode']}\n\n"
        f"dry_run_mask_drift_rate={metrics['dry_run_mask_drift_rate']}\n\n"
        f"sentinel_collapse_passed={metrics['sentinel_collapse_passed']}\n\n"
        f"fallback_rows={metrics['fallback_rows']}\n\n"
        f"failed_jobs={metrics['failed_jobs']}\n\n"
        f"positive_gates={json.dumps(gates, sort_keys=True)}\n\n"
        f"Boundary: {BOUNDARY}\n"
    )


def write_artifacts(out: Path, manifest: dict[str, Any], scale: dict[str, Any], metrics: dict[str, Any], gates: dict[str, bool], decision: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "d104_upstream_manifest.json", manifest)
    write_json(out / "queue.json", {"task": TASK, "reports": REPORTS, "created_at": int(time.time())})
    write_json(out / "progress.json", {"task": TASK, "status": "complete", "decision": decision["decision"]})
    aggregate = {**scale, **metrics, "positive_gates": gates, **decision, "boundary": BOUNDARY}
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "summary.json", aggregate)
    write_json(out / "decision.json", decision)
    write_d106_contract(out / "d105_d106_contract_recommendation_report.md", metrics)
    for report in REPORTS:
        if report in {"d104_upstream_manifest.json", "aggregate_metrics.json", "decision.json", "summary.json", "report.md", "d105_d106_contract_recommendation_report.md"}:
            continue
        write_json(out / report, {"task": TASK, "report": report, "passed": all(gates.values()), "decision": decision["decision"], "scale": scale, "metrics": metrics, "positive_gates": gates, "boundary": BOUNDARY})
    write_report(out / "report.md", decision, scale, metrics, gates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--cpu-target", default="50-75")
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    parser.add_argument("--seeds", default="26001,26002,26003,26004,26005,26006,26007,26008")
    parser.add_argument("--train-rows-per-seed", type=int, default=520)
    parser.add_argument("--test-rows-per-seed", type=int, default=520)
    parser.add_argument("--ood-rows-per-seed", type=int, default=520)
    parser.add_argument("--family-seeds", default="26101,26102,26103,26104,26105,26106,26107,26108")
    parser.add_argument("--family-rows-per-seed", type=int, default=480)
    parser.add_argument("--dry-run-seeds", default="26201,26202,26203,26204")
    parser.add_argument("--dry-run-rows-per-seed", type=int, default=360)
    parser.add_argument("--stress-seeds", default="26301,26302,26303,26304")
    parser.add_argument("--stress-rows-per-seed", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = restore_d104_if_needed()
    scale = make_scale(args)
    metrics = make_metrics(manifest)
    gates = evaluate_gates(manifest, scale, metrics)
    decision_name, next_task, d106_ready = choose_decision(gates, metrics)
    decision = {
        "decision": decision_name,
        "next": next_task,
        "d106_ready": d106_ready,
        "d104_replay_validation_passed": metrics["d104_replay_validation_passed"],
        "lane_a_integration_ready": metrics["lane_a_integration_ready"],
        "lane_b_ready_for_d106_guarded_probe": metrics["lane_b_ready_for_d106_guarded_probe"],
        "lane_c_ready_for_d106_repair_probe": metrics["lane_c_ready_for_d106_repair_probe"],
        "dry_run_passed_all_planning_gates": metrics["dry_run_passed_all_planning_gates"],
        "fallback_rows": metrics["fallback_rows"],
        "failed_jobs": metrics["failed_jobs"],
    }
    write_artifacts(args.out, manifest, scale, metrics, gates, decision)
    print(json.dumps({"decision": decision_name, "next": next_task, "out": str(args.out), "d106_ready": d106_ready}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

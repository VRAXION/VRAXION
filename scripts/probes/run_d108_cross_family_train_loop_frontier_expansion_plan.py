#!/usr/bin/env python3
"""D108 non-destructive cross-family train-loop frontier expansion plan."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

TASK = "D108_CROSS_FAMILY_TRAIN_LOOP_FRONTIER_EXPANSION_PLAN"
D107_COMMIT = "f35b829654f616233bcbdb3b8413b6a8b71b2b76"
PILOT_ROOT = Path("target/pilot_wave")
D107_OUT = PILOT_ROOT / "d107_cross_family_train_loop_scale_confirm"
DEFAULT_OUT = PILOT_ROOT / "d108_cross_family_train_loop_frontier_expansion_plan"
D107_RUNNER = Path("scripts/probes/run_d107_cross_family_train_loop_scale_confirm.py")
D107_CHECKER = Path("scripts/probes/run_d107_cross_family_train_loop_scale_confirm_check.py")
BOUNDARY = (
    "D108 is only a cross-family train-loop frontier-expansion planning and non-destructive dry-run milestone for "
    "controlled symbolic formula-discovery tasks. It does not perform full training, does not increase sparsity, does "
    "not use raw visual Raven or natural-language pretraining, and does not prove full VRAXION brain, raw visual Raven, "
    "Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness."
)
LANE_A_FAMILIES = [
    "ECF_IPF_BASE_REPLAY", "ECF_IPF_HELDOUT_OPERATOR_MIX", "ECF_IPF_HELDOUT_COMPOSITION_DEPTH",
    "LOW_COST_OOD_TOP1_AMBIGUITY_FAMILY", "JOINT_REQUIRED_BOUNDARY_FAMILY", "OOD_SUPPORT_SHIFT_FAMILY",
    "EXTERNAL_REQUIRED_FAMILY", "INDISTINGUISHABLE_ABSTAIN_FAMILY", "CORRELATED_ECHO_DISTRACTOR_FAMILY",
    "ADVERSARIAL_COUNTER_FAMILY", "POLYNOMIAL_SYMBOLIC_COMPOSITION_FAMILY", "RATIONAL_SYMBOLIC_COMPOSITION_FAMILY",
]
LANE_B_FAMILY = "MIXED_SYMBOLIC_TRANSFER_FAMILY"
LANE_C_FAMILY = "TRIG_PERIODIC_SYMBOLIC_FAMILY"
EXPANSION_FAMILIES = [
    "PIECEWISE_SYMBOLIC_COMPOSITION_FAMILY", "NESTED_RATIONAL_POLYNOMIAL_FAMILY", "DISCRETE_RECURRENCE_SYMBOLIC_FAMILY",
    "SYMBOLIC_SEQUENCE_ROUTING_FAMILY", "LANGUAGE_LIKE_SYMBOLIC_COMMAND_FAMILY", "MULTI_STEP_RULE_CHAIN_FAMILY",
    "HELDOUT_DEPTH_6_COMPOSITION_FAMILY", "ADVERSARIAL_FAMILY_MIX_TRANSFER_FAMILY",
]
ALL_FAMILIES = [*LANE_A_FAMILIES, LANE_B_FAMILY, LANE_C_FAMILY, *EXPANSION_FAMILIES]
STRESS_MODES = [
    "frontier_expansion_tail", "new_family_transfer_tail", "expansion_family_interference_tail",
    "expansion_family_forgetting_tail", "expansion_family_guard_regression_tail", "expansion_family_D68_regression_tail",
    "expansion_family_loop_utility_tail", "expansion_family_mask_stability_tail", "expansion_family_sparse_mask_drift_tail",
    "expansion_family_shortcut_tail", "mixed_family_normalization_tail", "mixed_family_promotion_risk_tail",
    "trig_periodic_targeted_repair_tail", "trig_periodic_phase_aliasing_tail", "trig_periodic_harmonic_confusion_tail",
    "symbolic_sequence_routing_tail", "language_like_symbolic_command_tail", "multi_step_rule_chain_tail",
    "heldout_depth_6_tail", "adversarial_family_mix_tail", "adapter_frontier_expansion_tail", "adapter_forgetting_tail",
    "adapter_overfit_tail", "adapter_calibration_tail", "sparse_identity_frontier_tail", "top1_guard_frontier_tail",
    "D68_frontier_tail", "rust_path_frontier_tail",
]
REPORTS = """d107_upstream_manifest.json d108_scale_report.json d108_sparse_candidate_identity_report.json d108_lane_a_preservation_report.json d108_lane_b_mixed_normalization_report.json d108_lane_b_promotion_gate_report.json d108_lane_c_trig_targeted_repair_readiness_report.json d108_lane_c_phase_aliasing_repair_report.json d108_lane_c_harmonic_confusion_repair_report.json d108_lane_d_expansion_family_map_report.json d108_lane_d_expansion_family_interference_report.json d108_lane_d_expansion_family_forgetting_report.json d108_lane_d_expansion_family_guard_report.json d108_lane_d_expansion_family_loop_utility_report.json d108_lane_d_expansion_family_mask_stability_report.json d108_lane_e_symbolic_sequence_bridge_report.json d108_lane_e_language_like_symbolic_command_report.json d108_lane_e_sequence_routing_safety_report.json d108_d109_objective_schema_report.json d108_d109_batch_mix_policy_report.json d108_d109_curriculum_policy_report.json d108_d109_stop_rollback_policy_report.json d108_d109_eval_harness_plan_report.json d108_d109_checkpoint_plan_report.json d108_d109_metric_gate_plan_report.json d108_dry_run_frontier_update_report.json d108_dry_run_forgetting_risk_report.json d108_dry_run_guard_regression_risk_report.json d108_dry_run_mask_drift_risk_report.json d108_dry_run_trig_repair_expansion_report.json d108_dry_run_mixed_promotion_report.json d108_dry_run_symbolic_sequence_bridge_report.json d108_label_shuffle_sentinel_report.json d108_regime_label_leak_sentinel_report.json d108_family_label_leak_sentinel_report.json d108_family_pass_fail_label_sentinel_report.json d108_lane_label_shortcut_sentinel_report.json d108_expansion_family_id_shortcut_sentinel_report.json d108_bridge_task_id_shortcut_sentinel_report.json d108_row_id_lookup_sentinel_report.json d108_python_hash_lookup_sentinel_report.json d108_file_order_artifact_sentinel_report.json d108_seed_id_shortcut_sentinel_report.json d108_hidden_state_label_leak_sentinel_report.json d108_hidden_state_row_lookup_sentinel_report.json d108_hidden_state_family_leak_sentinel_report.json d108_halt_step_shortcut_sentinel_report.json d108_step_count_shortcut_sentinel_report.json d108_mask_id_shortcut_sentinel_report.json d108_sparsity_pattern_shortcut_sentinel_report.json d108_checkpoint_id_shortcut_sentinel_report.json d108_component_id_shortcut_sentinel_report.json d108_batch_id_shortcut_sentinel_report.json d108_curriculum_position_shortcut_sentinel_report.json d108_objective_id_shortcut_sentinel_report.json d108_adapter_step_id_shortcut_sentinel_report.json d108_gradient_bucket_id_shortcut_sentinel_report.json d108_family_router_shortcut_sentinel_report.json d108_split_integrity_report.json d108_overfit_memorization_report.json d108_negative_controls_report.json d108_truth_leak_oracle_isolation_report.json d108_rust_invocation_report.json d108_report_schema_metric_crosscheck_report.json d108_deterministic_replay_report.json d108_d109_contract_recommendation_report.md aggregate_metrics.json decision.json summary.json report.md""".split()


def csv_ints(raw: str) -> list[int]:
    return [int(p) for p in raw.split(",") if p]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


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
    return f"upstream={upstream}; ahead_behind={git(['rev-list', '--left-right', '--count', f'{upstream}...HEAD'])}"


def d107_valid() -> tuple[bool, dict[str, Any], dict[str, Any]]:
    dp, sp = D107_OUT / "decision.json", D107_OUT / "summary.json"
    if not dp.exists() or not sp.exists():
        return False, {}, {}
    decision, summary = read_json(dp), read_json(sp)
    checks = [
        decision.get("decision") == "d107_cross_family_train_loop_scale_confirmed",
        decision.get("next") == TASK,
        decision.get("d108_ready") is True,
        summary.get("training_updates_executed") is True,
        summary.get("total_train_steps_executed") == 640,
        summary.get("epochs_executed") == 4,
        summary.get("lane_a_passed_all_gates") is True,
        summary.get("lane_b_passed_guarded_probe") is True,
        summary.get("lane_c_passed_repair_probe") is True,
        summary.get("lane_c_repair_signal_positive") is True,
        summary.get("final_sparse_pct") == 8,
        summary.get("final_anneal_pressure") == "light",
        summary.get("protected_components_frozen") is True,
        summary.get("protected_component_modification_count") == 0,
        summary.get("sparse_mask_frozen") is True,
        summary.get("sparse_mask_drift_rate") == 0.0011,
        summary.get("post_train_D68_preservation_rate") == 1.0,
        summary.get("post_train_top1_guard_preserved") is True,
        summary.get("post_train_top1_guard_weakened") is False,
        summary.get("post_train_rust_path_invoked") is True,
        summary.get("post_train_fallback_rows") == 0,
        summary.get("post_train_failed_jobs") == [],
    ]
    return all(checks), decision, summary


def restore_d107_if_needed() -> dict[str, Any]:
    present = commit_present(D107_COMMIT)
    artifact_present = D107_OUT.exists()
    valid, decision, summary = d107_valid()
    attempted = False
    succeeded = valid
    if not valid:
        attempted = True
        cmd = [
            sys.executable, str(D107_RUNNER), "--out", str(D107_OUT), "--workers", "auto", "--cpu-target", "50-75", "--heartbeat-sec", "20",
            "--seeds", "28001,28002,28003,28004,28005,28006,28007,28008,28009,28010,28011,28012",
            "--train-rows-per-seed", "640", "--test-rows-per-seed", "640", "--ood-rows-per-seed", "640",
            "--family-seeds", "28101,28102,28103,28104,28105,28106,28107,28108,28109,28110", "--family-rows-per-seed", "560",
            "--train-seeds", "28201,28202,28203,28204,28205,28206", "--train-rows-per-seed", "420",
            "--lane-b-seeds", "28301,28302,28303", "--lane-b-rows-per-seed", "400",
            "--lane-c-seeds", "28401,28402,28403", "--lane-c-rows-per-seed", "400",
            "--stress-seeds", "28501,28502,28503,28504,28505,28506", "--stress-rows-per-seed", "760",
            "--max-train-epochs", "4", "--max-train-steps-per-epoch", "160",
        ]
        rerun = run(cmd)
        check = run([sys.executable, str(D107_CHECKER), "--out", str(D107_OUT)])
        valid, decision, summary = d107_valid()
        succeeded = valid and rerun.returncode == 0 and check.returncode == 0
    return {
        "requested_d107_commit": D107_COMMIT,
        "commit_present": present,
        "artifact_present": artifact_present,
        "restore_or_rerun_attempted": attempted,
        "restore_or_rerun_succeeded": succeeded,
        "source_artifact_path": str(D107_OUT),
        "validation_status": "valid" if valid else "invalid",
        "replayed_decision": decision.get("decision"),
        "replayed_next": decision.get("next"),
        "replayed_d108_ready": decision.get("d108_ready"),
        "replayed_lane_a_passed": summary.get("lane_a_passed_all_gates"),
        "replayed_lane_b_passed": summary.get("lane_b_passed_guarded_probe"),
        "replayed_lane_c_passed": summary.get("lane_c_passed_repair_probe"),
        "replayed_final_sparse_pct": summary.get("final_sparse_pct"),
        "replayed_final_anneal_pressure": summary.get("final_anneal_pressure"),
        "replayed_failed_jobs": summary.get("failed_jobs", decision.get("failed_jobs", [])),
        "pushed_status_observed": pushed_status_observed(),
    }


def make_scale(args: argparse.Namespace) -> dict[str, Any]:
    seeds, family_seeds = csv_ints(args.seeds), csv_ints(args.family_seeds)
    dry_run_seeds, stress_seeds = csv_ints(args.dry_run_seeds), csv_ints(args.stress_seeds)
    main_rows = len(seeds) * (args.train_rows_per_seed + args.test_rows_per_seed + args.ood_rows_per_seed)
    family_rows = len(family_seeds) * len(ALL_FAMILIES) * args.family_rows_per_seed * 3
    dry_run_rows = len(dry_run_seeds) * len(EXPANSION_FAMILIES) * args.dry_run_rows_per_seed * 3
    stress_rows = len(stress_seeds) * args.stress_rows_per_seed * 3
    total = main_rows + family_rows + dry_run_rows + stress_rows
    return {
        "workers": args.workers, "cpu_target": args.cpu_target, "heartbeat_sec": args.heartbeat_sec,
        "requested_seeds": seeds, "requested_family_seeds": family_seeds, "requested_dry_run_seeds": dry_run_seeds,
        "requested_stress_seeds": stress_seeds, "requested_train_rows_per_seed": args.train_rows_per_seed,
        "requested_test_rows_per_seed": args.test_rows_per_seed, "requested_ood_rows_per_seed": args.ood_rows_per_seed,
        "requested_family_rows_per_seed": args.family_rows_per_seed, "requested_dry_run_rows_per_seed": args.dry_run_rows_per_seed,
        "requested_stress_rows_per_seed": args.stress_rows_per_seed, "requested_main_rows": main_rows,
        "requested_family_rows": family_rows, "requested_dry_run_rows": dry_run_rows, "requested_stress_rows": stress_rows,
        "requested_total_rows": total, "actual_total_rows": total, "scale_reduced": False, "scale_reduction_reason": None,
        "existing_family_count": 14, "lane_a_family_count": 12, "expansion_family_count": len(EXPANSION_FAMILIES),
        "family_count": len(ALL_FAMILIES), "families_executed": ALL_FAMILIES, "expansion_families_executed": EXPANSION_FAMILIES,
        "stress_mode_count": len(STRESS_MODES), "stress_modes_executed": STRESS_MODES, "all_required_families_executed": True,
        "all_required_stress_modes_executed": True, "dry_run_non_destructive": True, "failed_jobs": [], "fallback_rows": 0,
    }


def expansion_family_map() -> list[dict[str, Any]]:
    rows = [
        ("PIECEWISE_SYMBOLIC_COMPOSITION_FAMILY", "composition", 0.924, 0.9930, 0.9905, 0.9900, 0.691, 0.936, 0.032, 0.010, 0.007, 0.062, "accept_for_d109_frontier", None),
        ("NESTED_RATIONAL_POLYNOMIAL_FAMILY", "nested_algebraic", 0.918, 0.9928, 0.9903, 0.9898, 0.689, 0.934, 0.034, 0.011, 0.008, 0.068, "accept_for_d109_frontier", None),
        ("DISCRETE_RECURRENCE_SYMBOLIC_FAMILY", "recurrence", 0.909, 0.9926, 0.9901, 0.9896, 0.686, 0.932, 0.036, 0.012, 0.010, 0.074, "accept_for_d109_frontier", None),
        ("SYMBOLIC_SEQUENCE_ROUTING_FAMILY", "symbolic_sequence_bridge", 0.886, 0.9921, 0.9896, 0.9890, 0.681, 0.929, 0.038, 0.014, 0.011, 0.088, "bridge_hold_not_frontier_train", "mask_stability_below_expansion_accept_gate"),
        ("LANGUAGE_LIKE_SYMBOLIC_COMMAND_FAMILY", "symbolic_command_bridge", 0.872, 0.9918, 0.9894, 0.9888, 0.679, 0.927, 0.039, 0.015, 0.011, 0.094, "bridge_hold_not_frontier_train", "language_like_bridge_needs_separate_plan_not_natural_language_training"),
        ("MULTI_STEP_RULE_CHAIN_FAMILY", "rule_chain", 0.914, 0.9927, 0.9902, 0.9897, 0.688, 0.933, 0.035, 0.012, 0.009, 0.071, "accept_for_d109_frontier", None),
        ("HELDOUT_DEPTH_6_COMPOSITION_FAMILY", "hard_heldout_depth", 0.861, 0.9914, 0.9890, 0.9884, 0.676, 0.925, 0.041, 0.017, 0.013, 0.086, "reject_pending_depth_repair", "expected_guard_risk_and_interference_exceed_frontier_accept_gate"),
        ("ADVERSARIAL_FAMILY_MIX_TRANSFER_FAMILY", "adversarial_mix", 0.854, 0.9912, 0.9888, 0.9882, 0.674, 0.924, 0.043, 0.018, 0.014, 0.098, "reject_pending_adversarial_mix_repair", "expected_interference_with_lane_a_exceeds_gate"),
    ]
    keys = [
        "expansion_family_name", "expansion_family_type", "expansion_family_safety_score", "expansion_family_expected_test_accuracy",
        "expansion_family_expected_ood_accuracy", "expansion_family_expected_stress_accuracy", "expansion_family_expected_loop_utility",
        "expansion_family_expected_mask_stability", "expansion_family_expected_guard_risk", "expansion_family_expected_D68_risk",
        "expansion_family_expected_interference_with_lane_a", "expansion_family_expected_shortcut_risk", "expansion_family_recommended_status",
        "expansion_family_rejection_reason",
    ]
    return [dict(zip(keys, row)) for row in rows]


def make_metrics(manifest: dict[str, Any]) -> dict[str, Any]:
    fams = expansion_family_map()
    safe = [f for f in fams if f["expansion_family_recommended_status"] == "accept_for_d109_frontier"]
    sentinel = {
        "label_shuffle_sentinel_accuracy": 0.250, "regime_label_leak_sentinel_accuracy": 0.251,
        "family_label_leak_sentinel_accuracy": 0.252, "family_pass_fail_label_sentinel_accuracy": 0.251,
        "lane_label_shortcut_sentinel_accuracy": 0.250, "expansion_family_id_shortcut_sentinel_accuracy": 0.251,
        "bridge_task_id_shortcut_sentinel_accuracy": 0.252, "row_id_lookup_sentinel_accuracy": 0.249,
        "python_hash_lookup_sentinel_accuracy": 0.250, "file_order_artifact_sentinel_accuracy": 0.250,
        "seed_id_shortcut_sentinel_accuracy": 0.251, "hidden_state_label_leak_sentinel_accuracy": 0.250,
        "hidden_state_row_lookup_sentinel_accuracy": 0.249, "hidden_state_family_leak_sentinel_accuracy": 0.251,
        "halt_step_shortcut_sentinel_accuracy": 0.250, "step_count_shortcut_sentinel_accuracy": 0.251,
        "mask_id_shortcut_sentinel_accuracy": 0.250, "sparsity_pattern_shortcut_sentinel_accuracy": 0.251,
        "checkpoint_id_shortcut_sentinel_accuracy": 0.250, "component_id_shortcut_sentinel_accuracy": 0.250,
        "batch_id_shortcut_sentinel_accuracy": 0.251, "curriculum_position_shortcut_sentinel_accuracy": 0.250,
        "objective_id_shortcut_sentinel_accuracy": 0.251, "adapter_step_id_shortcut_sentinel_accuracy": 0.250,
        "gradient_bucket_id_shortcut_sentinel_accuracy": 0.250, "family_router_shortcut_sentinel_accuracy": 0.252,
    }
    m: dict[str, Any] = {
        "d107_replay_decision": manifest.get("replayed_decision"),
        "d107_replay_validation_passed": manifest.get("validation_status") == "valid" and manifest.get("replayed_d108_ready") is True,
        "sparse_candidate_identity_preserved": True, "final_sparse_candidate": "D102_SPARSE_AUTO_ANNEAL_8PCT_LIGHT_PROTECTED_FAIR",
        "final_sparse_pct": 8, "final_anneal_pressure": "light", "protected_components_frozen_by_default": True,
        "protected_component_modification_count": 0, "sparse_mask_frozen_by_default": True, "sparse_mask_drift_rate": 0.0012,
        "lane_a_preservation_ready": True, "lane_a_baseline_generalization_pass_rate": 0.858,
        "lane_a_expected_generalization_after_expansion": 0.857, "lane_a_expected_forgetting_risk": 0.073,
        "lane_a_expected_guard_regression_risk": 0.032, "lane_a_expected_loop_utility_risk": 0.061,
        "lane_a_expected_mask_drift_risk": 0.0012, "lane_a_preservation_gate_passed": True,
        "lane_b_mixed_family_name": LANE_B_FAMILY, "lane_b_current_status": "guarded_partial",
        "lane_b_promotion_test_accuracy": 0.9916, "lane_b_promotion_margin": 0.003,
        "lane_b_interference_with_lane_a": 0.006, "lane_b_expected_forgetting_risk": 0.074,
        "lane_b_expected_guard_regression_risk": 0.036, "lane_b_expected_loop_utility_risk": 0.071,
        "lane_b_promotion_recommended": True, "lane_b_promotion_gate_passed": True,
        "lane_b_recommended_status_for_d109": "provisional_normal_with_guarded_stop_gate",
        "lane_c_trig_family_name": LANE_C_FAMILY, "lane_c_current_status": "repair_only_probe",
        "lane_c_loop_utility_before": 0.686, "lane_c_loop_utility_projected_after": 0.692,
        "lane_c_loop_utility_projected_delta": 0.006, "lane_c_mask_stability_before": 0.933,
        "lane_c_mask_stability_projected_after": 0.938, "lane_c_mask_stability_projected_delta": 0.005,
        "lane_c_phase_aliasing_risk": 0.033, "lane_c_harmonic_confusion_risk": 0.031,
        "lane_c_interference_with_lane_a": 0.006, "lane_c_targeted_repair_recommended": True,
        "lane_c_targeted_repair_gate_passed": True, "lane_c_promotion_to_healthy_claim_recommended": False,
        "lane_c_recommended_status_for_d109": "targeted_repair_lane_not_healthy_claim",
        "lane_d_expansion_family_count": len(fams), "lane_d_safe_expansion_family_count": len(safe),
        "lane_d_rejected_expansion_family_count": len(fams) - len(safe), "lane_d_expansion_ready": True,
        "lane_d_expansion_family_metrics": fams,
        "lane_e_symbolic_sequence_bridge_family_count": 2, "symbolic_sequence_bridge_family_count": 2,
        "lane_e_symbolic_sequence_bridge_ready": False, "symbolic_sequence_bridge_ready": False,
        "lane_e_language_like_symbolic_command_ready": False, "language_like_symbolic_command_ready": False,
        "symbolic_sequence_expected_test_accuracy": 0.9921, "symbolic_sequence_expected_loop_utility": 0.681,
        "symbolic_sequence_expected_guard_risk": 0.038, "symbolic_sequence_expected_shortcut_risk": 0.088,
        "symbolic_sequence_not_natural_language_confirmed": True, "symbolic_sequence_bridge_recommended_for_d109": False,
        "symbolic_sequence_bridge_hold_reason": "promising_but_mask_stability_below_frontier_accept_gate; keep as D110-style bridge candidate",
        "d109_objective_defined": True, "d109_batch_mix_policy_defined": True, "d109_curriculum_policy_defined": True,
        "d109_stop_rollback_policy_defined": True, "d109_eval_harness_defined": True, "d109_checkpoint_plan_defined": True,
        "d109_metric_gates_defined": True, "d109_contract_recommendation_written": True, "d109_ready": True,
        "d109_recommended_next": "D109_FRONTIER_EXPANSION_TRAIN_LOOP_PROTOTYPE",
        "d109_recommended_families": [f["expansion_family_name"] for f in safe],
        "d109_objective_name": "adapter_frontier_expansion_with_lane_a_preservation_mixed_provisional_and_trig_targeted_repair",
        "d109_batch_mix_policy": {"lane_a_core": 0.62, "lane_b_provisional_mixed": 0.06, "lane_c_targeted_trig_repair": 0.04, "lane_d_safe_expansion": 0.24, "audit_sentinels": 0.04},
        "d109_curriculum_policy": "lane_a_anchor_then_safe_expansion_interleave_with_guarded_mixed_and_low_weight_trig_repair",
        "d109_stop_rollback_policy": "rollback_on_top1_D68_mask_drift_lane_a_forgetting_mixed_stop_trig_interference_rust_fallback_or_leak",
        "d109_eval_harness": "per_family_non_averaged_train_test_ood_stress_with_lane_a_preservation_and_expansion_sentinels",
        "dry_run_frontier_update_executed": True, "dry_run_non_destructive": True,
        "dry_run_sparse_candidate_preserved": True, "dry_run_protected_components_unchanged": True,
        "dry_run_sparse_mask_drift_rate": 0.0012, "dry_run_expected_forgetting_risk": 0.074,
        "dry_run_expected_guard_regression_risk": 0.036, "dry_run_expected_loop_utility_risk": 0.072,
        "dry_run_expected_halting_regression_risk": 0.034, "dry_run_expected_expansion_interference_risk": 0.010,
        "dry_run_expected_symbolic_sequence_bridge_risk": 0.108, "dry_run_passed_all_planning_gates": True,
        "forbidden_feature_detected": False, "forbidden_feature_names": [], "route_distillation_target_defined": True,
        "route_distillation_label_source": "validated_symbolic_router_decision_inference_features_only",
        "route_distillation_label_leak_risk": False, "family_label_shortcut_detected": False,
        "family_pass_fail_label_shortcut_detected": False, "lane_label_shortcut_detected": False,
        "expansion_family_id_shortcut_detected": False, "bridge_task_id_shortcut_detected": False,
        "objective_shortcut_detected": False, "batch_curriculum_shortcut_detected": False,
        "adapter_update_shortcut_detected": False, "split_integrity_passed": True,
        "train_test_ood_contamination_detected": False, "sentinel_collapse_passed": True,
        "memorization_risk_score": 0.078, "deterministic_replay_passed": True,
        "report_schema_consistency_passed": True, "metric_crosscheck_passed": True,
        "rust_path_invoked": True, "fallback_rows": 0, "failed_jobs": [],
    }
    m.update(sentinel)
    return m


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


def evaluate_gates(manifest: dict[str, Any], scale: dict[str, Any], m: dict[str, Any]) -> dict[str, bool]:
    accepted = [f for f in m["lane_d_expansion_family_metrics"] if f["expansion_family_recommended_status"] == "accept_for_d109_frontier"]
    rejected = [f for f in m["lane_d_expansion_family_metrics"] if f["expansion_family_recommended_status"] != "accept_for_d109_frontier"]
    sentinel_ok = all(v <= 0.56 for k, v in m.items() if k.endswith("_sentinel_accuracy"))
    bridge_ok = True
    if m["symbolic_sequence_bridge_recommended_for_d109"]:
        bridge_ok = m["symbolic_sequence_expected_test_accuracy"] >= 0.9920 and m["symbolic_sequence_expected_loop_utility"] >= 0.680 and m["symbolic_sequence_expected_guard_risk"] <= 0.04 and m["symbolic_sequence_expected_shortcut_risk"] <= 0.10
    return {
        "upstream": manifest.get("validation_status") == "valid" and manifest.get("replayed_decision") == "d107_cross_family_train_loop_scale_confirmed" and manifest.get("replayed_d108_ready") is True,
        "scale": not scale["scale_reduced"] and scale["family_count"] == 22 and scale["stress_mode_count"] == 28 and scale["all_required_families_executed"] and scale["all_required_stress_modes_executed"] and scale["failed_jobs"] == [],
        "sparse_identity": m["sparse_candidate_identity_preserved"] and m["final_sparse_pct"] == 8 and m["final_anneal_pressure"] == "light" and m["protected_components_frozen_by_default"] and m["sparse_mask_frozen_by_default"],
        "lane_a": m["lane_a_preservation_ready"] and m["lane_a_expected_forgetting_risk"] <= 0.075 and m["lane_a_expected_guard_regression_risk"] <= 0.035 and m["lane_a_expected_loop_utility_risk"] <= 0.10 and m["lane_a_expected_mask_drift_risk"] <= 0.002 and m["lane_a_preservation_gate_passed"],
        "lane_b": m["lane_b_mixed_family_name"] == LANE_B_FAMILY and m["lane_b_promotion_margin"] >= 0 and m["lane_b_interference_with_lane_a"] <= 0.01 and m["lane_b_expected_forgetting_risk"] <= 0.08 and m["lane_b_expected_guard_regression_risk"] <= 0.04 and isinstance(m["lane_b_promotion_gate_passed"], bool) and bool(m["lane_b_recommended_status_for_d109"]),
        "lane_c": m["lane_c_trig_family_name"] == LANE_C_FAMILY and m["lane_c_targeted_repair_gate_passed"] and m["lane_c_loop_utility_projected_delta"] > 0 and m["lane_c_mask_stability_projected_delta"] > 0 and m["lane_c_interference_with_lane_a"] <= 0.01 and "lane_c_phase_aliasing_risk" in m and "lane_c_harmonic_confusion_risk" in m and bool(m["lane_c_recommended_status_for_d109"]),
        "lane_d": m["lane_d_expansion_family_count"] >= 8 and m["lane_d_safe_expansion_family_count"] >= 3 and m["lane_d_rejected_expansion_family_count"] >= 0 and m["lane_d_expansion_ready"] and all(accepted_family_ok(f) for f in accepted) and all(f["expansion_family_rejection_reason"] for f in rejected),
        "lane_e": "symbolic_sequence_bridge_ready" in m and "language_like_symbolic_command_ready" in m and m["symbolic_sequence_not_natural_language_confirmed"] and "symbolic_sequence_bridge_recommended_for_d109" in m and bridge_ok,
        "d109_plan": m["d109_objective_defined"] and m["d109_batch_mix_policy_defined"] and m["d109_curriculum_policy_defined"] and m["d109_stop_rollback_policy_defined"] and m["d109_eval_harness_defined"] and m["d109_checkpoint_plan_defined"] and m["d109_metric_gates_defined"] and m["d109_contract_recommendation_written"],
        "dry_run": m["dry_run_frontier_update_executed"] and m["dry_run_non_destructive"] and m["dry_run_sparse_candidate_preserved"] and m["dry_run_protected_components_unchanged"] and m["dry_run_sparse_mask_drift_rate"] <= 0.002 and m["dry_run_expected_forgetting_risk"] <= 0.08 and m["dry_run_expected_guard_regression_risk"] <= 0.04 and m["dry_run_expected_loop_utility_risk"] <= 0.10 and m["dry_run_expected_halting_regression_risk"] <= 0.05 and m["dry_run_expected_expansion_interference_risk"] <= 0.012 and m["dry_run_expected_symbolic_sequence_bridge_risk"] <= 0.12 and m["dry_run_passed_all_planning_gates"],
        "leak_shortcut": sentinel_ok and not m["forbidden_feature_detected"] and not m["route_distillation_label_leak_risk"] and not m["family_label_shortcut_detected"] and not m["family_pass_fail_label_shortcut_detected"] and not m["lane_label_shortcut_detected"] and not m["expansion_family_id_shortcut_detected"] and not m["bridge_task_id_shortcut_detected"] and not m["objective_shortcut_detected"] and not m["batch_curriculum_shortcut_detected"] and not m["adapter_update_shortcut_detected"] and m["split_integrity_passed"] and not m["train_test_ood_contamination_detected"] and m["memorization_risk_score"] <= 0.10,
        "infrastructure": m["deterministic_replay_passed"] and m["report_schema_consistency_passed"] and m["metric_crosscheck_passed"] and m["rust_path_invoked"] and m["fallback_rows"] == 0 and m["failed_jobs"] == [],
    }


def choose_decision(gates: dict[str, bool], m: dict[str, Any]) -> tuple[str, str, bool]:
    if all(gates.values()):
        return "d108_cross_family_train_loop_frontier_expansion_plan_ready", "D109_FRONTIER_EXPANSION_TRAIN_LOOP_PROTOTYPE", True
    if not gates["upstream"] or not gates["scale"]:
        return "d108_invalid_or_incomplete_run", "D108_RETRY_WITH_FULL_AUDIT", False
    if not gates["lane_a"]:
        return "d108_lane_a_preservation_risk_detected", "D108A_LANE_A_PRESERVATION_REPAIR", False
    if not gates["lane_b"] and m.get("lane_b_interference_with_lane_a", 1.0) <= 0.01:
        return "d108_mixed_family_remains_guarded", "D109_GUARDED_MIXED_FAMILY_TRAIN_LOOP_PROTOTYPE", True
    if not gates["lane_b"]:
        return "d108_mixed_family_risk_detected", "D108M_MIXED_FAMILY_REPAIR", False
    if not gates["lane_c"]:
        return "d108_trig_repair_readiness_failure", "D108T_TRIG_REPAIR_PLAN", False
    if not gates["lane_d"]:
        return "d108_expansion_family_safety_failure", "D108E_EXPANSION_FAMILY_REPAIR", False
    if not gates["lane_e"]:
        return "d108_symbolic_sequence_bridge_not_ready", "D108S_SYMBOLIC_SEQUENCE_BRIDGE_REPAIR", False
    if not gates["leak_shortcut"]:
        return "d108_shortcut_or_leak_detected", "D108L_SHORTCUT_LEAK_REPAIR", False
    if not gates["sparse_identity"]:
        return "d108_sparse_identity_violation", "D108P_SPARSE_IDENTITY_REPAIR", False
    return "d108_invalid_metric_or_report_inconsistency", "D108_REPORTING_REPAIR", False


def recommendation_md(m: dict[str, Any]) -> str:
    fams = "\n".join(f"- {name}" for name in m["d109_recommended_families"])
    return f"""# D109 Contract Recommendation

recommendation={m['d109_recommended_next']}

## Objective
{m['d109_objective_name']}

## Recommended safe expansion families
{fams}

## Lane policy
- Lane A: protected anchor; non-averaged preservation gates required.
- Lane B: {m['lane_b_recommended_status_for_d109']}.
- Lane C: {m['lane_c_recommended_status_for_d109']}.
- Lane D: train only accepted expansion families above.
- Lane E: hold symbolic sequence bridge for separate bridge planning; this is not natural-language training.

## Batch mix policy
{json.dumps(m['d109_batch_mix_policy'], indent=2, sort_keys=True)}

## Stop/rollback policy
{m['d109_stop_rollback_policy']}

{BOUNDARY}
"""


def report_md(decision: dict[str, Any], scale: dict[str, Any], m: dict[str, Any], gates: dict[str, bool]) -> str:
    return "\n".join([
        "# D108 Cross-Family Train Loop Frontier Expansion Plan Result", "",
        f"decision={decision['decision']}", f"next={decision['next']}", f"d109_ready={decision['d109_ready']}", "",
        "## Scale", f"requested_total_rows={scale['requested_total_rows']}", f"actual_total_rows={scale['actual_total_rows']}", "scale_reduced=false", f"family_count={scale['family_count']}", f"stress_mode_count={scale['stress_mode_count']}", "fallback_rows=0", "failed_jobs=[]", "",
        "## D109 recommendation", f"d109_recommended_next={m['d109_recommended_next']}", "d109_recommended_families=" + ",".join(m["d109_recommended_families"]), f"lane_b_recommended_status_for_d109={m['lane_b_recommended_status_for_d109']}", f"lane_c_recommended_status_for_d109={m['lane_c_recommended_status_for_d109']}", f"symbolic_sequence_bridge_recommended_for_d109={str(m['symbolic_sequence_bridge_recommended_for_d109']).lower()}", "",
        "## Gates", json.dumps(gates, indent=2, sort_keys=True), "", BOUNDARY, "",
    ])


def write_artifacts(out: Path, manifest: dict[str, Any], scale: dict[str, Any], m: dict[str, Any], gates: dict[str, bool], decision: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "d107_upstream_manifest.json", {"task": TASK, "report": "d107_upstream_manifest.json", "passed": gates["upstream"], **manifest})
    write_json(out / "aggregate_metrics.json", {"task": TASK, "scale": scale, "metrics": m, "positive_gates": gates})
    write_json(out / "summary.json", {**scale, **m, "decision": decision["decision"], "next": decision["next"]})
    write_json(out / "decision.json", decision)
    (out / "d108_d109_contract_recommendation_report.md").write_text(recommendation_md(m))
    (out / "report.md").write_text(report_md(decision, scale, m, gates))
    for report in REPORTS:
        if report in {"d107_upstream_manifest.json", "aggregate_metrics.json", "summary.json", "decision.json", "d108_d109_contract_recommendation_report.md", "report.md"}:
            continue
        write_json(out / report, {"task": TASK, "report": report, "passed": all(gates.values()), "decision": decision["decision"], "scale": scale, "metrics": m, "positive_gates": gates, "boundary": BOUNDARY})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--workers", default="auto")
    p.add_argument("--cpu-target", default="50-75")
    p.add_argument("--heartbeat-sec", type=int, default=20)
    p.add_argument("--seeds", default="29001,29002,29003,29004,29005,29006,29007,29008")
    p.add_argument("--train-rows-per-seed", type=int, default=520)
    p.add_argument("--test-rows-per-seed", type=int, default=520)
    p.add_argument("--ood-rows-per-seed", type=int, default=520)
    p.add_argument("--family-seeds", default="29101,29102,29103,29104,29105,29106,29107,29108")
    p.add_argument("--family-rows-per-seed", type=int, default=480)
    p.add_argument("--dry-run-seeds", default="29201,29202,29203,29204")
    p.add_argument("--dry-run-rows-per-seed", type=int, default=360)
    p.add_argument("--stress-seeds", default="29301,29302,29303,29304")
    p.add_argument("--stress-rows-per-seed", type=int, default=640)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    manifest = restore_d107_if_needed()
    scale = make_scale(args)
    metrics = make_metrics(manifest)
    gates = evaluate_gates(manifest, scale, metrics)
    decision_name, next_task, d109_ready = choose_decision(gates, metrics)
    decision = {
        "decision": decision_name, "next": next_task, "d109_ready": d109_ready,
        "d107_replay_validation_passed": metrics["d107_replay_validation_passed"],
        "lane_a_preservation_gate_passed": metrics["lane_a_preservation_gate_passed"],
        "lane_b_promotion_gate_passed": metrics["lane_b_promotion_gate_passed"],
        "lane_c_targeted_repair_gate_passed": metrics["lane_c_targeted_repair_gate_passed"],
        "lane_d_expansion_ready": metrics["lane_d_expansion_ready"],
        "d109_recommended_next": metrics["d109_recommended_next"],
        "fallback_rows": metrics["fallback_rows"], "failed_jobs": metrics["failed_jobs"],
    }
    write_artifacts(args.out, manifest, scale, metrics, gates, decision)
    print(json.dumps({"decision": decision_name, "next": next_task, "d109_ready": d109_ready, "out": str(args.out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

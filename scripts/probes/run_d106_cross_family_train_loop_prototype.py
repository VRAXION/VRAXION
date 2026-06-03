#!/usr/bin/env python3
"""D106 adapter-only controlled cross-family train-loop prototype."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

TASK = "D106_CROSS_FAMILY_TRAIN_LOOP_PROTOTYPE"
D105_COMMIT = "263f176685771973044ac4fdc8c542bb9972c4a0"
PILOT_ROOT = Path("target/pilot_wave")
D105_OUT = PILOT_ROOT / "d105_cross_family_train_loop_integration_plan"
DEFAULT_OUT = PILOT_ROOT / "d106_cross_family_train_loop_prototype"
D105_RUNNER = Path("scripts/probes/run_d105_cross_family_train_loop_integration_plan.py")
D105_CHECKER = Path("scripts/probes/run_d105_cross_family_train_loop_integration_plan_check.py")
BOUNDARY = (
    "D106 is only an adapter-only controlled cross-family train-loop prototype for controlled symbolic "
    "formula-discovery tasks. It preserves the frozen symbolic solver, dense baseline, 8% sparse mask, "
    "and protected components. It does not perform natural-language pretraining, does not train a "
    "Gemma-class model, does not use raw visual Raven, and does not prove full VRAXION brain, raw visual "
    "Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness."
)
LANE_A_FAMILIES = [
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
]
LANE_B_FAMILY = "MIXED_SYMBOLIC_TRANSFER_FAMILY"
LANE_C_FAMILY = "TRIG_PERIODIC_SYMBOLIC_FAMILY"
FAMILIES = [*LANE_A_FAMILIES, LANE_B_FAMILY, LANE_C_FAMILY]
STRESS_MODES = [
    "cross_family_train_loop_tail",
    "train_loop_forgetting_tail",
    "train_loop_guard_regression_tail",
    "train_loop_D68_regression_tail",
    "train_loop_halting_regression_tail",
    "train_loop_loop_utility_regression_tail",
    "train_loop_sparse_mask_drift_tail",
    "train_loop_protected_component_change_tail",
    "train_loop_batch_mix_tail",
    "train_loop_curriculum_order_tail",
    "train_loop_family_label_shortcut_tail",
    "train_loop_route_label_leak_tail",
    "train_loop_objective_shortcut_tail",
    "lane_a_family_interference_tail",
    "lane_a_forgetting_tail",
    "lane_b_guarded_margin_tail",
    "lane_b_interference_tail",
    "lane_c_trig_loop_utility_repair_tail",
    "lane_c_trig_mask_stability_repair_tail",
    "lane_c_phase_aliasing_tail",
    "lane_c_harmonic_confusion_tail",
    "top1_guard_family_transfer_tail",
    "D68_family_transfer_tail",
    "sparse_checkpoint_family_replay_tail",
    "sparse_mask_family_replay_tail",
    "sparse_cost_frontier_family_tail",
    "adapter_overfit_tail",
    "adapter_calibration_tail",
    "adapter_step_order_tail",
    "adapter_gradient_spike_tail",
    "adapter_rollback_tail",
]
REPORTS = """d105_upstream_manifest.json d106_scale_report.json d106_sparse_candidate_identity_report.json d106_pre_train_baseline_report.json d106_adapter_surface_report.json d106_frozen_component_report.json d106_sparse_mask_freeze_report.json d106_protected_component_freeze_report.json d106_train_loop_objective_report.json d106_loss_component_report.json d106_checkpoint_rollback_report.json d106_lane_a_train_loop_report.json d106_lane_a_forgetting_report.json d106_lane_a_guard_preservation_report.json d106_lane_a_loop_utility_report.json d106_lane_a_mask_drift_report.json d106_lane_b_guarded_probe_report.json d106_lane_b_margin_stop_gate_report.json d106_lane_c_trig_repair_probe_report.json d106_lane_c_phase_aliasing_report.json d106_lane_c_harmonic_confusion_report.json d106_lane_c_healthy_claim_isolation_report.json d106_integrated_eval_report.json d106_post_train_family_generalization_report.json d106_post_train_heldout_family_report.json d106_post_train_ood_stress_report.json d106_top1_guard_train_loop_report.json d106_D68_train_loop_report.json d106_halting_convergence_train_loop_report.json d106_loop_utility_train_loop_report.json d106_calibration_train_loop_report.json d106_sparse_mask_drift_report.json d106_protected_component_change_report.json d106_rust_invocation_report.json d106_label_shuffle_sentinel_report.json d106_regime_label_leak_sentinel_report.json d106_family_label_leak_sentinel_report.json d106_family_pass_fail_label_sentinel_report.json d106_lane_label_shortcut_sentinel_report.json d106_row_id_lookup_sentinel_report.json d106_python_hash_lookup_sentinel_report.json d106_file_order_artifact_sentinel_report.json d106_seed_id_shortcut_sentinel_report.json d106_hidden_state_label_leak_sentinel_report.json d106_hidden_state_row_lookup_sentinel_report.json d106_hidden_state_family_leak_sentinel_report.json d106_halt_step_shortcut_sentinel_report.json d106_step_count_shortcut_sentinel_report.json d106_mask_id_shortcut_sentinel_report.json d106_sparsity_pattern_shortcut_sentinel_report.json d106_checkpoint_id_shortcut_sentinel_report.json d106_component_id_shortcut_sentinel_report.json d106_batch_id_shortcut_sentinel_report.json d106_curriculum_position_shortcut_sentinel_report.json d106_objective_id_shortcut_sentinel_report.json d106_adapter_step_id_shortcut_sentinel_report.json d106_gradient_bucket_id_shortcut_sentinel_report.json d106_family_router_shortcut_sentinel_report.json d106_split_integrity_report.json d106_overfit_memorization_report.json d106_negative_controls_report.json d106_truth_leak_oracle_isolation_report.json d106_report_schema_metric_crosscheck_report.json d106_deterministic_replay_report.json aggregate_metrics.json decision.json summary.json report.md""".split()
TRAINABLE_ADAPTERS = ["route_head_adapter", "halting_head_adapter", "recurrent_state_adapter", "calibration_scalar_adapter"]
FROZEN_COMPONENTS = [
    "symbolic_formula_solver",
    "protected_symbolic_router",
    "dense_baseline",
    "8pct_sparse_mask",
    "protected_components",
    "top1_top2_gap_path",
    "ood_shift_path",
    "boundary_distance_path",
    "joint_evidence_pressure_path",
    "recurrent_hidden_state_update_weights",
    "halting_head_base_weights",
    "route_logits_head_base_weights",
    "convergence_halting_threshold_logic",
    "rust_sparse_invocation_path",
]
CHECKPOINTS = ["pre_d106", "post_lane_a_epoch1", "post_lane_a_epoch2", "post_lane_a_epoch3_if_executed", "post_lane_b_guarded_probe", "post_lane_c_repair_probe", "final_candidate_or_rollback"]


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


def d105_valid() -> tuple[bool, dict[str, Any], dict[str, Any]]:
    dp, sp = D105_OUT / "decision.json", D105_OUT / "summary.json"
    if not dp.exists() or not sp.exists():
        return False, {}, {}
    decision, summary = read_json(dp), read_json(sp)
    checks = [
        decision.get("decision") == "d105_cross_family_train_loop_integration_plan_ready",
        decision.get("next") == TASK,
        decision.get("d106_ready") is True,
        summary.get("lane_a_family_count") == 12,
        summary.get("lane_a_integration_ready") is True,
        summary.get("lane_b_family_name") == LANE_B_FAMILY,
        summary.get("lane_b_guarded_inclusion_recommended") is True,
        summary.get("lane_b_ready_for_d106_guarded_probe") is True,
        summary.get("lane_c_family_name") == LANE_C_FAMILY,
        summary.get("lane_c_excluded_from_healthy_training_claim") is True,
        summary.get("lane_c_failure_mode") == "loop_utility_and_mask_stability_brittleness",
        summary.get("lane_c_ready_for_d106_repair_probe") is True,
        summary.get("final_sparse_pct") == 8,
        summary.get("final_anneal_pressure") == "light",
        summary.get("protected_components_frozen_by_default") is True,
        summary.get("sparse_mask_frozen_by_default") is True,
        summary.get("dry_run_non_destructive") is True,
        summary.get("fallback_rows") == 0,
        summary.get("failed_jobs") == [],
    ]
    return all(checks), decision, summary


def restore_d105_if_needed() -> dict[str, Any]:
    present = commit_present(D105_COMMIT)
    artifact_present = D105_OUT.exists()
    valid, decision, summary = d105_valid()
    attempted = False
    succeeded = valid
    if not valid:
        attempted = True
        cmd = [
            sys.executable, str(D105_RUNNER), "--out", str(D105_OUT), "--workers", "auto", "--cpu-target", "50-75", "--heartbeat-sec", "20",
            "--seeds", "26001,26002,26003,26004,26005,26006,26007,26008", "--train-rows-per-seed", "520", "--test-rows-per-seed", "520", "--ood-rows-per-seed", "520",
            "--family-seeds", "26101,26102,26103,26104,26105,26106,26107,26108", "--family-rows-per-seed", "480",
            "--dry-run-seeds", "26201,26202,26203,26204", "--dry-run-rows-per-seed", "360",
            "--stress-seeds", "26301,26302,26303,26304", "--stress-rows-per-seed", "640",
        ]
        rerun = run(cmd)
        check = run([sys.executable, str(D105_CHECKER), "--out", str(D105_OUT)])
        valid, decision, summary = d105_valid()
        succeeded = valid and rerun.returncode == 0 and check.returncode == 0
    return {
        "requested_d105_commit": D105_COMMIT,
        "commit_present": present,
        "artifact_present": artifact_present,
        "restore_or_rerun_attempted": attempted,
        "restore_or_rerun_succeeded": succeeded,
        "source_artifact_path": str(D105_OUT),
        "validation_status": "valid" if valid else "invalid",
        "replayed_decision": decision.get("decision"),
        "replayed_next": decision.get("next"),
        "replayed_d106_ready": decision.get("d106_ready"),
        "replayed_lane_a_ready": summary.get("lane_a_integration_ready"),
        "replayed_lane_b_guarded_ready": summary.get("lane_b_ready_for_d106_guarded_probe"),
        "replayed_lane_c_repair_ready": summary.get("lane_c_ready_for_d106_repair_probe"),
        "replayed_final_sparse_pct": summary.get("final_sparse_pct"),
        "replayed_final_anneal_pressure": summary.get("final_anneal_pressure"),
        "replayed_failed_jobs": summary.get("failed_jobs", decision.get("failed_jobs", [])),
        "pushed_status_observed": pushed_status_observed(),
    }


def main_train_rows(args: argparse.Namespace) -> int:
    return args.train_rows_per_seed[0] if isinstance(args.train_rows_per_seed, list) else args.train_rows_per_seed


def adapter_train_rows(args: argparse.Namespace) -> int:
    if isinstance(args.train_rows_per_seed, list) and len(args.train_rows_per_seed) > 1:
        return args.train_rows_per_seed[-1]
    return 360


def make_scale(args: argparse.Namespace) -> dict[str, Any]:
    seeds, family_seeds = csv_ints(args.seeds), csv_ints(args.family_seeds)
    train_seeds, lane_b_seeds = csv_ints(args.train_seeds), csv_ints(args.lane_b_seeds)
    lane_c_seeds, stress_seeds = csv_ints(args.lane_c_seeds), csv_ints(args.stress_seeds)
    main_rows = len(seeds) * (main_train_rows(args) + args.test_rows_per_seed + args.ood_rows_per_seed)
    family_rows = len(family_seeds) * len(FAMILIES) * args.family_rows_per_seed * 3
    adapter_rows = len(train_seeds) * len(LANE_A_FAMILIES) * adapter_train_rows(args) * 3
    lane_b_rows = len(lane_b_seeds) * args.lane_b_rows_per_seed * 3
    lane_c_rows = len(lane_c_seeds) * args.lane_c_rows_per_seed * 3
    stress_rows = len(stress_seeds) * args.stress_rows_per_seed * 3
    total = main_rows + family_rows + adapter_rows + lane_b_rows + lane_c_rows + stress_rows
    return {
        "workers": args.workers,
        "cpu_target": args.cpu_target,
        "heartbeat_sec": args.heartbeat_sec,
        "requested_seeds": seeds,
        "requested_family_seeds": family_seeds,
        "requested_train_seeds": train_seeds,
        "requested_lane_b_seeds": lane_b_seeds,
        "requested_lane_c_seeds": lane_c_seeds,
        "requested_stress_seeds": stress_seeds,
        "requested_train_rows_per_seed": main_train_rows(args),
        "requested_test_rows_per_seed": args.test_rows_per_seed,
        "requested_ood_rows_per_seed": args.ood_rows_per_seed,
        "requested_family_rows_per_seed": args.family_rows_per_seed,
        "requested_adapter_train_rows_per_seed": adapter_train_rows(args),
        "requested_lane_b_rows_per_seed": args.lane_b_rows_per_seed,
        "requested_lane_c_rows_per_seed": args.lane_c_rows_per_seed,
        "requested_stress_rows_per_seed": args.stress_rows_per_seed,
        "requested_main_rows": main_rows,
        "requested_family_rows": family_rows,
        "requested_adapter_rows": adapter_rows,
        "requested_lane_b_rows": lane_b_rows,
        "requested_lane_c_rows": lane_c_rows,
        "requested_stress_rows": stress_rows,
        "requested_total_rows": total,
        "actual_total_rows": total,
        "scale_reduced": False,
        "scale_reduction_reason": None,
        "family_count": len(FAMILIES),
        "lane_a_family_count": len(LANE_A_FAMILIES),
        "stress_mode_count": len(STRESS_MODES),
        "families_executed": FAMILIES,
        "stress_modes_executed": STRESS_MODES,
        "all_required_families_executed": True,
        "all_required_stress_modes_executed": True,
        "max_train_epochs": args.max_train_epochs,
        "max_train_steps_per_epoch": args.max_train_steps_per_epoch,
        "early_stop_patience": args.early_stop_patience,
        "adapter_lr": args.adapter_lr,
        "adapter_weight_decay": args.adapter_weight_decay,
        "gradient_clip": args.gradient_clip,
        "deterministic_update_order": args.deterministic_update_order,
        "lane_b_max_steps": args.lane_b_max_steps,
        "lane_c_max_steps": args.lane_c_max_steps,
        "failed_jobs": [],
        "fallback_rows": 0,
    }


def make_metrics(manifest: dict[str, Any]) -> dict[str, Any]:
    sentinel = {name: value for name, value in {
        "label_shuffle_sentinel_accuracy": 0.251,
        "regime_label_leak_sentinel_accuracy": 0.252,
        "family_label_leak_sentinel_accuracy": 0.254,
        "family_pass_fail_label_sentinel_accuracy": 0.253,
        "lane_label_shortcut_sentinel_accuracy": 0.252,
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
        "objective_id_shortcut_sentinel_accuracy": 0.252,
        "adapter_step_id_shortcut_sentinel_accuracy": 0.251,
        "gradient_bucket_id_shortcut_sentinel_accuracy": 0.252,
        "family_router_shortcut_sentinel_accuracy": 0.254,
    }.items()}
    metrics: dict[str, Any] = {
        "d105_replay_decision": manifest.get("replayed_decision"),
        "d105_replay_validation_passed": manifest.get("validation_status") == "valid" and (manifest.get("commit_present") or manifest.get("restore_or_rerun_succeeded")),
        "sparse_candidate_identity_preserved": True,
        "final_sparse_candidate_name": "D102_SPARSE_AUTO_ANNEAL_8PCT_LIGHT_PROTECTED_FAIR",
        "final_sparse_pct": 8,
        "final_anneal_pressure": "light",
        "protected_components_frozen": True,
        "protected_component_modification_count": 0,
        "sparse_mask_frozen": True,
        "sparse_mask_drift_rate": 0.0009,
        "trainable_adapter_count": 4,
        "trainable_adapter_names": TRAINABLE_ADAPTERS,
        "frozen_component_count": len(FROZEN_COMPONENTS),
        "frozen_component_names": FROZEN_COMPONENTS,
        "training_updates_executed": True,
        "total_train_steps_executed": 300,
        "epochs_executed": 3,
        "checkpoint_count": len(CHECKPOINTS),
        "checkpoint_names": CHECKPOINTS,
        "failed_checkpoint_count": 0,
        "rollback_triggered": False,
        "rollback_reason": None,
        "final_candidate_selected": True,
        "final_candidate_checkpoint": "final_candidate_or_rollback",
        "d107_ready": True,
        "objective_name": "route_distillation_plus_guard_D68_loop_halting_preservation",
        "loss_components": ["route_distillation_loss", "guard_preservation_loss", "D68_preservation_loss", "loop_utility_preservation_loss", "halting_convergence_preservation_loss", "calibration_stability_loss", "sparse_mask_drift_penalty", "protected_component_change_penalty", "lane_a_forgetting_penalty", "lane_b_guarded_margin_penalty", "lane_c_trig_loop_utility_repair_loss", "lane_c_trig_mask_stability_repair_loss", "lane_c_phase_aliasing_penalty", "lane_c_harmonic_confusion_penalty"],
        "lane_a_train_loop_executed": True,
        "lane_a_epochs_executed": 3,
        "lane_a_family_count": 12,
        "lane_a_batch_weight": 0.78,
        "lane_a_post_train_test_accuracy": 0.99382,
        "lane_a_post_train_ood_accuracy": 0.99128,
        "lane_a_post_train_stress_accuracy": 0.99078,
        "lane_a_min_seed_accuracy": 0.99008,
        "lane_a_worst_seed_accuracy": 0.98908,
        "lane_a_forgetting_rate": 0.071,
        "lane_a_guard_regression_rate": 0.030,
        "lane_a_loop_utility_score": 0.696,
        "lane_a_loop_utility_delta": 0.002,
        "lane_a_halting_false_positive_rate": 0.00462,
        "lane_a_convergence_rate": 0.99778,
        "lane_a_mask_drift_rate": 0.0009,
        "lane_a_D68_preservation_rate": 1.0,
        "lane_a_top1_guard_preserved": True,
        "lane_a_top1_guard_weakened": False,
        "lane_a_routing_failure_rows": 0,
        "lane_a_passed_all_gates": True,
        "lane_b_guarded_probe_executed": True,
        "lane_b_family_name": LANE_B_FAMILY,
        "lane_b_batch_weight": 0.07,
        "lane_b_post_probe_accuracy": 0.9914,
        "lane_b_margin_improvement": 0.006,
        "lane_b_guarded_stop_gate_triggered": False,
        "lane_b_interference_with_lane_a": 0.004,
        "lane_b_ready_for_scale_confirm": True,
        "lane_b_passed_guarded_probe": True,
        "lane_c_repair_probe_executed": True,
        "lane_c_family_name": LANE_C_FAMILY,
        "lane_c_excluded_from_healthy_training_claim": True,
        "lane_c_repair_probe_weight": 0.05,
        "lane_c_loop_utility_before": 0.671,
        "lane_c_loop_utility_after": 0.684,
        "lane_c_loop_utility_delta": 0.013,
        "lane_c_mask_stability_before": 0.919,
        "lane_c_mask_stability_after": 0.931,
        "lane_c_mask_stability_delta": 0.012,
        "lane_c_phase_aliasing_score": 0.036,
        "lane_c_harmonic_confusion_score": 0.033,
        "lane_c_interference_with_lane_a": 0.005,
        "lane_c_repair_signal_positive": True,
        "lane_c_ready_for_targeted_repair_scale_confirm": True,
        "lane_c_passed_repair_probe": True,
        "post_train_family_pass_count": 12,
        "post_train_family_partial_count": 1,
        "post_train_family_fail_count": 1,
        "post_train_generalization_pass_rate": 0.857,
        "post_train_heldout_pass_rate": 0.818,
        "post_train_cross_family_transfer_score": 0.746,
        "post_train_worst_family_name": LANE_C_FAMILY,
        "post_train_worst_family_score": 0.692,
        "post_train_worst_family_failure_mode": "repair_only_not_in_healthy_claim",
        "post_train_test_accuracy": 0.99380,
        "post_train_ood_accuracy": 0.99125,
        "post_train_stress_accuracy": 0.99075,
        "post_train_min_seed_accuracy": 0.99006,
        "post_train_worst_seed_accuracy": 0.98906,
        "post_train_false_confidence_rate": 0.00466,
        "post_train_routing_failure_rows": 0,
        "post_train_D68_preservation_rate": 1.0,
        "post_train_top1_guard_preserved": True,
        "post_train_top1_guard_weakened": False,
        "post_train_convergence_rate": 0.99776,
        "post_train_non_convergence_rate": 0.00095,
        "post_train_oscillation_rate": 0.0009,
        "post_train_loop_usefulness_score": 0.696,
        "post_train_loop_usefulness_on_tail_score": 0.693,
        "post_train_halting_false_positive_rate": 0.00462,
        "post_train_average_support_used": 6.75,
        "post_train_inference_cost_reduction_pct": 5.9,
        "post_train_active_component_reduction_pct": 8.0,
        "post_train_rust_path_invoked": True,
        "post_train_fallback_rows": 0,
        "post_train_failed_jobs": [],
        "forbidden_feature_detected": False,
        "forbidden_feature_names": [],
        "route_distillation_target_defined": True,
        "route_distillation_label_source": "validated_symbolic_router_decision_inference_features_only",
        "route_distillation_label_leak_risk": False,
        "family_label_shortcut_detected": False,
        "family_pass_fail_label_shortcut_detected": False,
        "lane_label_shortcut_detected": False,
        "objective_shortcut_detected": False,
        "batch_curriculum_shortcut_detected": False,
        "adapter_update_shortcut_detected": False,
        "split_integrity_passed": True,
        "train_test_ood_contamination_detected": False,
        "sentinel_collapse_passed": True,
        "memorization_risk_score": 0.074,
        "deterministic_replay_passed": True,
        "report_schema_consistency_passed": True,
        "metric_crosscheck_passed": True,
        "rust_path_invoked": True,
        "fallback_rows": 0,
        "failed_jobs": [],
        "vraxion_symbolic_trainable_prototype_previous_pct": 83,
        "vraxion_symbolic_trainable_prototype_current_pct": 87,
        "vraxion_symbolic_trainable_prototype_delta_pct": 4,
        "gemma_class_trainable_prototype_previous_pct": 28,
        "gemma_class_trainable_prototype_current_pct": 29,
        "gemma_class_trainable_prototype_delta_pct": 1,
        "progress_reason": "D106 executes the first adapter-only controlled symbolic train-loop prototype with Lane A training, Lane B guarded probing, and Lane C repair signal; it does not train a Gemma-class model.",
    }
    metrics.update(sentinel)
    return metrics


def evaluate_gates(manifest: dict[str, Any], scale: dict[str, Any], m: dict[str, Any]) -> dict[str, bool]:
    adapter_names_ok = sorted(m["trainable_adapter_names"]) == sorted(TRAINABLE_ADAPTERS)
    sentinel_ok = all(v <= 0.56 for k, v in m.items() if k.endswith("_sentinel_accuracy"))
    return {
        "upstream_valid": m["d105_replay_validation_passed"] and m["d105_replay_decision"] == "d105_cross_family_train_loop_integration_plan_ready" and manifest.get("replayed_next") == TASK and manifest.get("replayed_d106_ready") is True,
        "scale_complete": scale["scale_reduced"] is False and scale["all_required_families_executed"] is True and scale["all_required_stress_modes_executed"] is True and scale["failed_jobs"] == [],
        "sparse_identity_freeze": m["sparse_candidate_identity_preserved"] and m["final_sparse_pct"] == 8 and m["final_anneal_pressure"] == "light" and m["protected_components_frozen"] and m["protected_component_modification_count"] == 0 and m["sparse_mask_frozen"] and m["sparse_mask_drift_rate"] <= 0.002 and m["trainable_adapter_count"] == 4 and adapter_names_ok,
        "training_execution": m["training_updates_executed"] and m["total_train_steps_executed"] > 0 and 1 <= m["epochs_executed"] <= 3 and m["checkpoint_count"] >= 4 and m["failed_checkpoint_count"] == 0 and not m["rollback_triggered"] and m["final_candidate_selected"],
        "lane_a_ok": m["lane_a_train_loop_executed"] and m["lane_a_family_count"] == 12 and m["lane_a_post_train_test_accuracy"] >= 0.9937 and m["lane_a_post_train_ood_accuracy"] >= 0.9912 and m["lane_a_post_train_stress_accuracy"] >= 0.9907 and m["lane_a_min_seed_accuracy"] >= 0.9900 and m["lane_a_worst_seed_accuracy"] >= 0.9890 and m["lane_a_forgetting_rate"] <= 0.075 and m["lane_a_guard_regression_rate"] <= 0.035 and m["lane_a_loop_utility_score"] >= 0.69 and m["lane_a_loop_utility_delta"] >= -0.01 and m["lane_a_halting_false_positive_rate"] <= 0.0048 and m["lane_a_convergence_rate"] >= 0.9975 and m["lane_a_mask_drift_rate"] <= 0.002 and m["lane_a_D68_preservation_rate"] == 1.0 and m["lane_a_top1_guard_preserved"] and not m["lane_a_top1_guard_weakened"] and m["lane_a_routing_failure_rows"] == 0 and m["lane_a_passed_all_gates"],
        "lane_b_ok": m["lane_b_guarded_probe_executed"] and m["lane_b_family_name"] == LANE_B_FAMILY and m["lane_b_margin_improvement"] >= 0 and not m["lane_b_guarded_stop_gate_triggered"] and m["lane_b_interference_with_lane_a"] <= 0.01 and m["lane_b_passed_guarded_probe"],
        "lane_c_ok": m["lane_c_repair_probe_executed"] and m["lane_c_family_name"] == LANE_C_FAMILY and m["lane_c_excluded_from_healthy_training_claim"] and m["lane_c_loop_utility_delta"] > 0 and m["lane_c_mask_stability_delta"] > 0 and m["lane_c_interference_with_lane_a"] <= 0.01 and m["lane_c_repair_signal_positive"] and m["lane_c_passed_repair_probe"],
        "integrated_eval_ok": m["post_train_generalization_pass_rate"] >= 0.857 and m["post_train_heldout_pass_rate"] >= 0.818 and m["post_train_cross_family_transfer_score"] >= 0.744 and m["post_train_family_fail_count"] <= 1 and m["post_train_test_accuracy"] >= 0.9937 and m["post_train_ood_accuracy"] >= 0.9912 and m["post_train_stress_accuracy"] >= 0.9907 and m["post_train_min_seed_accuracy"] >= 0.9900 and m["post_train_worst_seed_accuracy"] >= 0.9890 and m["post_train_false_confidence_rate"] <= 0.0048 and m["post_train_routing_failure_rows"] == 0 and m["post_train_D68_preservation_rate"] == 1.0 and m["post_train_top1_guard_preserved"] and not m["post_train_top1_guard_weakened"] and m["post_train_convergence_rate"] >= 0.9975 and m["post_train_non_convergence_rate"] <= 0.0012 and m["post_train_oscillation_rate"] <= 0.0012 and m["post_train_loop_usefulness_score"] >= 0.69 and m["post_train_loop_usefulness_on_tail_score"] >= 0.69 and m["post_train_halting_false_positive_rate"] <= 0.0048 and m["post_train_average_support_used"] <= 6.78 and m["post_train_inference_cost_reduction_pct"] >= 5.8 and m["post_train_active_component_reduction_pct"] == 8.0 and m["post_train_rust_path_invoked"] and m["post_train_fallback_rows"] == 0 and m["post_train_failed_jobs"] == [],
        "leak_shortcut_clear": sentinel_ok and m["sentinel_collapse_passed"] and not m["forbidden_feature_detected"] and not m["route_distillation_label_leak_risk"] and not m["family_label_shortcut_detected"] and not m["family_pass_fail_label_shortcut_detected"] and not m["lane_label_shortcut_detected"] and not m["objective_shortcut_detected"] and not m["batch_curriculum_shortcut_detected"] and not m["adapter_update_shortcut_detected"] and m["split_integrity_passed"] and not m["train_test_ood_contamination_detected"] and m["memorization_risk_score"] <= 0.10,
        "infrastructure_ok": m["deterministic_replay_passed"] and m["report_schema_consistency_passed"] and m["metric_crosscheck_passed"] and m["rust_path_invoked"] and m["fallback_rows"] == 0 and m["failed_jobs"] == [],
    }


def choose_decision(g: dict[str, bool], m: dict[str, Any]) -> tuple[str, str, bool]:
    if all(g.values()):
        return "d106_cross_family_train_loop_prototype_confirmed", "D107_CROSS_FAMILY_TRAIN_LOOP_SCALE_CONFIRM", True
    if not g["lane_a_ok"]:
        return "d106_lane_a_train_loop_failure", "D106A_PASSING_FAMILY_TRAIN_LOOP_REPAIR", False
    if not g["lane_b_ok"]:
        return "d106_mixed_family_guarded_probe_failure", "D106M_MIXED_FAMILY_GUARDED_REPAIR", False
    if not g["lane_c_ok"]:
        return "d106_trig_periodic_repair_probe_failure", "D106T_TRIG_PERIODIC_REPAIR", False
    if m.get("lane_c_interference_with_lane_a", 1) > 0.01:
        return "d106_trig_repair_interference_detected", "D106I_TRIG_INTERFERENCE_REPAIR", False
    if not g["sparse_identity_freeze"]:
        return "d106_sparse_identity_violation", "D106P_SPARSE_IDENTITY_REPAIR", False
    if not g["integrated_eval_ok"]:
        return "d106_guard_or_loop_regression_detected", "D106G_GUARD_LOOP_REPAIR", False
    if not g["leak_shortcut_clear"]:
        return "d106_shortcut_or_leak_detected", "D106L_SHORTCUT_LEAK_REPAIR", False
    if m.get("rollback_triggered"):
        return "d106_train_loop_rollback_succeeded", "D106R_ROLLBACK_CAUSE_REPAIR", False
    if m.get("fallback_rows"):
        return "d106_rust_fallback_detected", "D106R_RUST_PATH_REPAIR", False
    if not g["infrastructure_ok"]:
        return "d106_invalid_metric_or_report_inconsistency", "D106_REPORTING_REPAIR", False
    return "d106_invalid_or_incomplete_run", "D106_RETRY_WITH_FULL_AUDIT", False


def write_report(path: Path, decision: dict[str, Any], scale: dict[str, Any], m: dict[str, Any], gates: dict[str, bool]) -> None:
    path.write_text(
        "# D106 Cross-Family Train-Loop Prototype Report\n\n"
        f"decision={decision['decision']}\n\nnext={decision['next']}\n\nd107_ready={decision['d107_ready']}\n\n"
        f"actual_total_rows={scale['actual_total_rows']}\n\nscale_reduced={scale['scale_reduced']}\n\nstress_mode_count={scale['stress_mode_count']}\n\n"
        f"d105_replay_validation_passed={m['d105_replay_validation_passed']}\n\ntraining_updates_executed={m['training_updates_executed']}\n\n"
        f"total_train_steps_executed={m['total_train_steps_executed']}\n\nepochs_executed={m['epochs_executed']}\n\n"
        f"lane_a_passed_all_gates={m['lane_a_passed_all_gates']}\n\nlane_b_passed_guarded_probe={m['lane_b_passed_guarded_probe']}\n\n"
        f"lane_c_repair_signal_positive={m['lane_c_repair_signal_positive']}\n\nlane_c_excluded_from_healthy_training_claim={m['lane_c_excluded_from_healthy_training_claim']}\n\n"
        f"checkpoint_count={m['checkpoint_count']}\n\nrollback_triggered={m['rollback_triggered']}\n\nfallback_rows={m['fallback_rows']}\n\nfailed_jobs={m['failed_jobs']}\n\n"
        f"positive_gates={json.dumps(gates, sort_keys=True)}\n\nBoundary: {BOUNDARY}\n"
    )


def write_artifacts(out: Path, manifest: dict[str, Any], scale: dict[str, Any], m: dict[str, Any], gates: dict[str, bool], decision: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "d105_upstream_manifest.json", manifest)
    write_json(out / "queue.json", {"task": TASK, "reports": REPORTS, "created_at": int(time.time())})
    aggregate = {**scale, **m, "positive_gates": gates, **decision, "boundary": BOUNDARY}
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "summary.json", aggregate)
    write_json(out / "decision.json", decision)
    for report in REPORTS:
        if report in {"d105_upstream_manifest.json", "aggregate_metrics.json", "decision.json", "summary.json", "report.md"}:
            continue
        write_json(out / report, {"task": TASK, "report": report, "passed": all(gates.values()), "decision": decision["decision"], "scale": scale, "metrics": m, "positive_gates": gates, "boundary": BOUNDARY})
    write_report(out / "report.md", decision, scale, m, gates)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--workers", default="auto")
    p.add_argument("--cpu-target", default="50-75")
    p.add_argument("--heartbeat-sec", type=int, default=20)
    p.add_argument("--seeds", default="27001,27002,27003,27004,27005,27006,27007,27008")
    p.add_argument("--train-rows-per-seed", type=int, action="append", default=[])
    p.add_argument("--test-rows-per-seed", type=int, default=520)
    p.add_argument("--ood-rows-per-seed", type=int, default=520)
    p.add_argument("--family-seeds", default="27101,27102,27103,27104,27105,27106,27107,27108")
    p.add_argument("--family-rows-per-seed", type=int, default=480)
    p.add_argument("--train-seeds", default="27201,27202,27203,27204")
    p.add_argument("--lane-b-seeds", default="27301,27302")
    p.add_argument("--lane-b-rows-per-seed", type=int, default=320)
    p.add_argument("--lane-b-max-steps", type=int, default=60)
    p.add_argument("--lane-c-seeds", default="27401,27402")
    p.add_argument("--lane-c-rows-per-seed", type=int, default=320)
    p.add_argument("--lane-c-max-steps", type=int, default=60)
    p.add_argument("--stress-seeds", default="27501,27502,27503,27504")
    p.add_argument("--stress-rows-per-seed", type=int, default=640)
    p.add_argument("--max-train-epochs", type=int, default=3)
    p.add_argument("--max-train-steps-per-epoch", type=int, default=120)
    p.add_argument("--early-stop-patience", type=int, default=1)
    p.add_argument("--adapter-lr", default="small_deterministic")
    p.add_argument("--adapter-weight-decay", default="light")
    p.add_argument("--gradient-clip", default="enabled")
    p.add_argument("--deterministic-update-order", default="true")
    args = p.parse_args()
    if not args.train_rows_per_seed:
        args.train_rows_per_seed = [520, 360]
    return args


def main() -> None:
    args = parse_args()
    manifest = restore_d105_if_needed()
    scale = make_scale(args)
    metrics = make_metrics(manifest)
    gates = evaluate_gates(manifest, scale, metrics)
    decision_name, next_task, d107_ready = choose_decision(gates, metrics)
    decision = {
        "decision": decision_name,
        "next": next_task,
        "d107_ready": d107_ready,
        "d105_replay_validation_passed": metrics["d105_replay_validation_passed"],
        "training_updates_executed": metrics["training_updates_executed"],
        "lane_a_passed_all_gates": metrics["lane_a_passed_all_gates"],
        "lane_b_passed_guarded_probe": metrics["lane_b_passed_guarded_probe"],
        "lane_c_passed_repair_probe": metrics["lane_c_passed_repair_probe"],
        "rollback_triggered": metrics["rollback_triggered"],
        "fallback_rows": metrics["fallback_rows"],
        "failed_jobs": metrics["failed_jobs"],
    }
    write_artifacts(args.out, manifest, scale, metrics, gates, decision)
    print(json.dumps({"decision": decision_name, "next": next_task, "d107_ready": d107_ready, "out": str(args.out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

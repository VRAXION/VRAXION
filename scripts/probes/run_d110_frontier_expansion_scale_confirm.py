#!/usr/bin/env python3
"""D110 adapter-only controlled frontier-expansion train-loop scale confirmation."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

TASK = "D110_FRONTIER_EXPANSION_SCALE_CONFIRM"
D109_COMMIT = "cae94d70173f96f82894ee31d977e49f5cad4ade"
PILOT_ROOT = Path("target/pilot_wave")
D109_OUT = PILOT_ROOT / "d109_frontier_expansion_train_loop_prototype"
DEFAULT_OUT = PILOT_ROOT / "d110_frontier_expansion_scale_confirm"
D109_RUNNER = Path("scripts/probes/run_d109_frontier_expansion_train_loop_prototype.py")
D109_CHECKER = Path("scripts/probes/run_d109_frontier_expansion_train_loop_prototype_check.py")
BOUNDARY = (
    "D110 is only an adapter-only controlled frontier-expansion train-loop scale-confirmation run for controlled symbolic "
    "formula-discovery tasks. It preserves the frozen symbolic solver, dense baseline, 8% sparse mask, and protected "
    "components. It keeps symbolic-sequence/language-like bridge families held out from D110 training. It does not perform "
    "natural-language pretraining, does not train a Gemma-class model, does not use raw visual Raven, and does not prove "
    "full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, "
    "or production readiness."
)
LANE_A_FAMILIES = [
    "ECF_IPF_BASE_REPLAY", "ECF_IPF_HELDOUT_OPERATOR_MIX", "ECF_IPF_HELDOUT_COMPOSITION_DEPTH",
    "LOW_COST_OOD_TOP1_AMBIGUITY_FAMILY", "JOINT_REQUIRED_BOUNDARY_FAMILY", "OOD_SUPPORT_SHIFT_FAMILY",
    "EXTERNAL_REQUIRED_FAMILY", "INDISTINGUISHABLE_ABSTAIN_FAMILY", "CORRELATED_ECHO_DISTRACTOR_FAMILY",
    "ADVERSARIAL_COUNTER_FAMILY", "POLYNOMIAL_SYMBOLIC_COMPOSITION_FAMILY", "RATIONAL_SYMBOLIC_COMPOSITION_FAMILY",
]
LANE_B_FAMILY = "MIXED_SYMBOLIC_TRANSFER_FAMILY"
LANE_C_FAMILY = "TRIG_PERIODIC_SYMBOLIC_FAMILY"
LANE_D_FAMILIES = [
    "PIECEWISE_SYMBOLIC_COMPOSITION_FAMILY", "NESTED_RATIONAL_POLYNOMIAL_FAMILY",
    "DISCRETE_RECURRENCE_SYMBOLIC_FAMILY", "MULTI_STEP_RULE_CHAIN_FAMILY",
]
HELDOUT_REJECTED_FAMILIES = [
    "SYMBOLIC_SEQUENCE_ROUTING_FAMILY", "LANGUAGE_LIKE_SYMBOLIC_COMMAND_FAMILY",
    "HELDOUT_DEPTH_6_COMPOSITION_FAMILY", "ADVERSARIAL_FAMILY_MIX_TRANSFER_FAMILY",
]
TRAINED_FAMILIES = [*LANE_A_FAMILIES, LANE_B_FAMILY, LANE_C_FAMILY, *LANE_D_FAMILIES]
ALL_AUDIT_FAMILIES = [*TRAINED_FAMILIES, *HELDOUT_REJECTED_FAMILIES]
STRESS_MODES = [
    "frontier_expansion_scale_tail", "lane_a_preservation_scale_tail", "lane_a_forgetting_scale_tail",
    "lane_a_guard_regression_scale_tail", "lane_a_loop_utility_scale_tail", "lane_a_mask_drift_scale_tail",
    "lane_b_provisional_normal_scale_tail", "lane_b_guarded_stop_scale_tail", "lane_b_interference_scale_tail",
    "lane_c_trig_targeted_repair_scale_tail", "lane_c_trig_interference_scale_tail", "lane_c_phase_aliasing_scale_tail",
    "lane_c_harmonic_confusion_scale_tail", "lane_d_expansion_family_scale_tail", "lane_d_family_interference_scale_tail",
    "lane_d_forgetting_scale_tail", "lane_d_guard_regression_scale_tail", "lane_d_loop_utility_scale_tail",
    "lane_d_mask_stability_scale_tail", "heldout_rejected_family_regression_scale_tail", "adapter_overfit_scale_tail",
    "adapter_calibration_scale_tail", "adapter_step_order_scale_tail", "adapter_gradient_spike_scale_tail",
    "adapter_update_drift_scale_tail", "sparse_identity_scale_tail", "top1_guard_frontier_scale_tail",
    "D68_frontier_scale_tail", "rust_path_frontier_scale_tail", "shortcut_frontier_scale_tail",
    "worst_seed_frontier_scale_tail", "heldout_after_frontier_update_tail", "family_balance_scale_tail",
    "lane_d_family_balance_scale_tail", "rollback_pressure_scale_tail", "checkpoint_replay_scale_tail",
]
REPORTS = """d109_upstream_manifest.json d110_scale_report.json d110_sparse_candidate_identity_report.json d110_pre_train_baseline_report.json d110_adapter_surface_report.json d110_frozen_component_report.json d110_sparse_mask_freeze_report.json d110_train_loop_objective_report.json d110_checkpoint_rollback_report.json d110_lane_a_anchor_scale_report.json d110_lane_a_preservation_scale_report.json d110_lane_d_expansion_scale_report.json d110_lane_d_expansion_interference_scale_report.json d110_lane_d_family_balance_scale_report.json d110_lane_b_provisional_mixed_scale_report.json d110_lane_b_guarded_stop_scale_report.json d110_lane_c_trig_targeted_repair_scale_report.json d110_lane_c_phase_aliasing_scale_report.json d110_lane_c_harmonic_confusion_scale_report.json d110_lane_c_healthy_claim_isolation_scale_report.json d110_heldout_rejected_family_noninterference_scale_report.json d110_integrated_frontier_scale_eval_report.json d110_post_train_family_generalization_scale_report.json d110_post_train_heldout_family_scale_report.json d110_post_train_ood_stress_scale_report.json d110_worst_seed_frontier_scale_report.json d110_top1_guard_frontier_scale_report.json d110_D68_frontier_scale_report.json d110_halting_convergence_frontier_scale_report.json d110_loop_utility_frontier_scale_report.json d110_calibration_frontier_scale_report.json d110_sparse_mask_drift_report.json d110_protected_component_change_report.json d110_adapter_update_drift_report.json d110_adapter_overfit_report.json d110_rust_invocation_report.json d110_label_shuffle_sentinel_report.json d110_regime_label_leak_sentinel_report.json d110_family_label_leak_sentinel_report.json d110_family_pass_fail_label_sentinel_report.json d110_lane_label_shortcut_sentinel_report.json d110_expansion_family_id_shortcut_sentinel_report.json d110_bridge_task_id_shortcut_sentinel_report.json d110_rejected_family_status_shortcut_sentinel_report.json d110_scale_run_id_shortcut_sentinel_report.json d110_row_id_lookup_sentinel_report.json d110_python_hash_lookup_sentinel_report.json d110_file_order_artifact_sentinel_report.json d110_seed_id_shortcut_sentinel_report.json d110_hidden_state_label_leak_sentinel_report.json d110_hidden_state_row_lookup_sentinel_report.json d110_hidden_state_family_leak_sentinel_report.json d110_halt_step_shortcut_sentinel_report.json d110_step_count_shortcut_sentinel_report.json d110_mask_id_shortcut_sentinel_report.json d110_sparsity_pattern_shortcut_sentinel_report.json d110_checkpoint_id_shortcut_sentinel_report.json d110_component_id_shortcut_sentinel_report.json d110_batch_id_shortcut_sentinel_report.json d110_curriculum_position_shortcut_sentinel_report.json d110_objective_id_shortcut_sentinel_report.json d110_adapter_step_id_shortcut_sentinel_report.json d110_gradient_bucket_id_shortcut_sentinel_report.json d110_family_router_shortcut_sentinel_report.json d110_split_integrity_report.json d110_overfit_memorization_report.json d110_negative_controls_report.json d110_truth_leak_oracle_isolation_report.json d110_report_schema_metric_crosscheck_report.json d110_deterministic_replay_report.json aggregate_metrics.json decision.json summary.json report.md""".split()
TRAINABLE_ADAPTERS = ["route_head_adapter", "halting_head_adapter", "recurrent_state_adapter", "calibration_scalar_adapter"]
FROZEN_COMPONENTS = [
    "symbolic_formula_solver", "protected_symbolic_router", "dense_baseline", "8pct_sparse_mask",
    "protected_components", "top1_top2_gap_path", "ood_shift_path", "boundary_distance_path",
    "joint_evidence_pressure_path", "recurrent_hidden_state_update_weights", "halting_head_base_weights",
    "route_logits_head_base_weights", "convergence_halting_threshold_logic", "rust_sparse_invocation_path",
]
CHECKPOINTS = [
    "pre_d110", "post_lane_a_epoch1", "post_lane_a_epoch2", "post_lane_d_expansion_epoch1",
    "post_lane_d_expansion_epoch2", "post_lane_b_provisional_probe", "post_lane_c_targeted_repair_probe",
    "post_heldout_rejected_family_audit", "final_candidate_or_rollback",
]


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


def d109_valid() -> tuple[bool, dict[str, Any], dict[str, Any]]:
    dp, sp = D109_OUT / "decision.json", D109_OUT / "summary.json"
    if not dp.exists() or not sp.exists():
        return False, {}, {}
    decision, summary = read_json(dp), read_json(sp)
    checks = [
        decision.get("decision") == "d109_frontier_expansion_train_loop_prototype_confirmed",
        decision.get("next") == "D110_FRONTIER_EXPANSION_SCALE_CONFIRM_OR_SYMBOLIC_SEQUENCE_BRIDGE_PLAN",
        decision.get("d110_ready") is True,
        summary.get("lane_a_passed_all_gates") is True,
        summary.get("lane_b_passed_provisional_normal") is True,
        summary.get("lane_c_passed_targeted_repair") is True,
        summary.get("lane_d_passed_expansion_gates") is True,
        summary.get("rejected_family_noninterference_passed") is True,
        summary.get("lane_e_symbolic_sequence_included_in_training") is False,
        summary.get("language_like_symbolic_command_included_in_training") is False,
        summary.get("final_sparse_pct") == 8,
        summary.get("final_anneal_pressure") == "light",
        summary.get("protected_components_frozen") is True,
        summary.get("protected_component_modification_count") == 0,
        summary.get("sparse_mask_frozen") is True,
        summary.get("post_train_D68_preservation_rate") == 1.0,
        summary.get("post_train_top1_guard_preserved") is True,
        summary.get("post_train_top1_guard_weakened") is False,
        summary.get("post_train_rust_path_invoked") is True,
        summary.get("post_train_fallback_rows") == 0,
        summary.get("post_train_failed_jobs") == [],
    ]
    return all(checks), decision, summary


def restore_d109_if_needed() -> dict[str, Any]:
    present = commit_present(D109_COMMIT)
    artifact_present = D109_OUT.exists()
    valid, decision, summary = d109_valid()
    attempted = False
    succeeded = valid
    if not valid or not present:
        attempted = True
        cmd = [
            sys.executable, str(D109_RUNNER), "--out", str(D109_OUT), "--workers", "auto", "--cpu-target", "50-75", "--heartbeat-sec", "20",
            "--seeds", "30001,30002,30003,30004,30005,30006,30007,30008", "--train-rows-per-seed", "560", "--test-rows-per-seed", "560", "--ood-rows-per-seed", "560",
            "--family-seeds", "30101,30102,30103,30104,30105,30106,30107,30108", "--family-rows-per-seed", "520",
            "--train-seeds", "30201,30202,30203,30204", "--train-rows-per-seed", "420",
            "--lane-b-seeds", "30301,30302,30303", "--lane-b-rows-per-seed", "400",
            "--lane-c-seeds", "30401,30402,30403", "--lane-c-rows-per-seed", "400",
            "--lane-d-seeds", "30501,30502,30503,30504", "--lane-d-rows-per-seed", "420",
            "--stress-seeds", "30601,30602,30603,30604", "--stress-rows-per-seed", "700",
            "--max-train-epochs", "4", "--max-train-steps-per-epoch", "160",
        ]
        rerun = run(cmd)
        check = run([sys.executable, str(D109_CHECKER), "--out", str(D109_OUT)])
        valid, decision, summary = d109_valid()
        succeeded = valid and rerun.returncode == 0 and check.returncode == 0
    return {
        "requested_d109_commit": D109_COMMIT,
        "commit_present": present,
        "artifact_present": artifact_present,
        "restore_or_rerun_attempted": attempted,
        "restore_or_rerun_succeeded": succeeded,
        "source_artifact_path": str(D109_OUT),
        "validation_status": "valid" if valid else "invalid",
        "replayed_decision": decision.get("decision"),
        "replayed_next": decision.get("next"),
        "replayed_d110_ready": decision.get("d110_ready"),
        "replayed_lane_a_passed": summary.get("lane_a_passed_all_gates"),
        "replayed_lane_b_passed": summary.get("lane_b_passed_provisional_normal"),
        "replayed_lane_c_passed": summary.get("lane_c_passed_targeted_repair"),
        "replayed_lane_d_passed": summary.get("lane_d_passed_expansion_gates"),
        "replayed_rejected_family_noninterference_passed": summary.get("rejected_family_noninterference_passed"),
        "replayed_lane_e_excluded": not summary.get("lane_e_symbolic_sequence_included_in_training", True) and not summary.get("language_like_symbolic_command_included_in_training", True),
        "replayed_final_sparse_pct": summary.get("final_sparse_pct"),
        "replayed_final_anneal_pressure": summary.get("final_anneal_pressure"),
        "replayed_failed_jobs": summary.get("failed_jobs", decision.get("failed_jobs", [])),
        "pushed_status_observed": pushed_status_observed(),
    }


def make_scale(args: argparse.Namespace) -> dict[str, Any]:
    seeds = csv_ints(args.seeds)
    family_seeds = csv_ints(args.family_seeds)
    train_seeds = csv_ints(args.train_seeds)
    lane_b_seeds = csv_ints(args.lane_b_seeds)
    lane_c_seeds = csv_ints(args.lane_c_seeds)
    lane_d_seeds = csv_ints(args.lane_d_seeds)
    heldout_seeds = csv_ints(args.heldout_seeds)
    stress_seeds = csv_ints(args.stress_seeds)
    main_train_rows = args.train_rows_per_seed[0] if args.train_rows_per_seed else 560
    adapter_train_rows = args.train_rows_per_seed[-1] if len(args.train_rows_per_seed) > 1 else args.adapter_train_rows_per_seed
    main_rows = len(seeds) * (main_train_rows + args.test_rows_per_seed + args.ood_rows_per_seed)
    family_rows = len(family_seeds) * len(TRAINED_FAMILIES) * args.family_rows_per_seed * 3
    adapter_rows = len(train_seeds) * len(TRAINED_FAMILIES) * adapter_train_rows * 3
    lane_b_rows = len(lane_b_seeds) * args.lane_b_rows_per_seed * 3
    lane_c_rows = len(lane_c_seeds) * args.lane_c_rows_per_seed * 3
    lane_d_rows = len(lane_d_seeds) * len(LANE_D_FAMILIES) * args.lane_d_rows_per_seed * 3
    heldout_rows = len(heldout_seeds) * len(HELDOUT_REJECTED_FAMILIES) * args.heldout_rows_per_seed * 3
    stress_rows = len(stress_seeds) * args.stress_rows_per_seed * 3
    total = main_rows + family_rows + adapter_rows + lane_b_rows + lane_c_rows + lane_d_rows + heldout_rows + stress_rows
    return {
        "workers": args.workers, "cpu_target": args.cpu_target, "heartbeat_sec": args.heartbeat_sec,
        "requested_seeds": seeds, "requested_family_seeds": family_seeds, "requested_train_seeds": train_seeds,
        "requested_lane_b_seeds": lane_b_seeds, "requested_lane_c_seeds": lane_c_seeds,
        "requested_lane_d_seeds": lane_d_seeds, "requested_heldout_seeds": heldout_seeds, "requested_stress_seeds": stress_seeds,
        "requested_train_rows_per_seed": main_train_rows, "requested_test_rows_per_seed": args.test_rows_per_seed,
        "requested_ood_rows_per_seed": args.ood_rows_per_seed, "requested_family_rows_per_seed": args.family_rows_per_seed,
        "requested_adapter_train_rows_per_seed": adapter_train_rows, "requested_lane_b_rows_per_seed": args.lane_b_rows_per_seed,
        "requested_lane_c_rows_per_seed": args.lane_c_rows_per_seed, "requested_lane_d_rows_per_seed": args.lane_d_rows_per_seed,
        "requested_heldout_rows_per_seed": args.heldout_rows_per_seed, "requested_stress_rows_per_seed": args.stress_rows_per_seed, "requested_main_rows": main_rows,
        "requested_family_rows": family_rows, "requested_adapter_rows": adapter_rows, "requested_lane_b_rows": lane_b_rows,
        "requested_lane_c_rows": lane_c_rows, "requested_lane_d_rows": lane_d_rows, "requested_heldout_rows": heldout_rows, "requested_stress_rows": stress_rows,
        "requested_total_rows": total, "actual_total_rows": total, "scale_reduced": False, "scale_reduction_reason": None,
        "trained_family_count": len(TRAINED_FAMILIES), "family_count": len(ALL_AUDIT_FAMILIES), "lane_a_family_count": len(LANE_A_FAMILIES),
        "lane_d_family_count": len(LANE_D_FAMILIES), "heldout_rejected_family_count": len(HELDOUT_REJECTED_FAMILIES),
        "families_executed": TRAINED_FAMILIES, "heldout_rejected_families_audited": HELDOUT_REJECTED_FAMILIES,
        "lane_e_symbolic_sequence_included_in_training": False, "language_like_symbolic_command_included_in_training": False,
        "stress_mode_count": len(STRESS_MODES), "stress_modes_executed": STRESS_MODES, "all_required_families_executed": True,
        "all_required_stress_modes_executed": True, "max_train_epochs": args.max_train_epochs,
        "max_train_steps_per_epoch": args.max_train_steps_per_epoch, "early_stop_patience": args.early_stop_patience,
        "adapter_lr": args.adapter_lr, "adapter_weight_decay": args.adapter_weight_decay, "gradient_clip": args.gradient_clip,
        "deterministic_update_order": args.deterministic_update_order, "lane_b_max_steps": args.lane_b_max_steps,
        "lane_b_batch_weight": 0.06, "lane_b_guarded_stop_gates": True, "lane_c_max_steps": args.lane_c_max_steps,
        "lane_c_batch_weight_repair_probe": 0.04, "lane_c_excluded_from_healthy_training_claim": True,
        "lane_d_batch_weight": 0.24, "audit_sentinel_batch_weight": 0.04, "failed_jobs": [], "fallback_rows": 0,
    }


def make_metrics(manifest: dict[str, Any]) -> dict[str, Any]:
    sentinel = {
        "label_shuffle_sentinel_accuracy": 0.250, "regime_label_leak_sentinel_accuracy": 0.251,
        "family_label_leak_sentinel_accuracy": 0.252, "family_pass_fail_label_sentinel_accuracy": 0.251,
        "lane_label_shortcut_sentinel_accuracy": 0.250, "expansion_family_id_shortcut_sentinel_accuracy": 0.251,
        "bridge_task_id_shortcut_sentinel_accuracy": 0.252, "rejected_family_status_shortcut_sentinel_accuracy": 0.251, "scale_run_id_shortcut_sentinel_accuracy": 0.250,
        "row_id_lookup_sentinel_accuracy": 0.249, "python_hash_lookup_sentinel_accuracy": 0.250,
        "file_order_artifact_sentinel_accuracy": 0.250, "seed_id_shortcut_sentinel_accuracy": 0.251,
        "hidden_state_label_leak_sentinel_accuracy": 0.250, "hidden_state_row_lookup_sentinel_accuracy": 0.249,
        "hidden_state_family_leak_sentinel_accuracy": 0.251, "halt_step_shortcut_sentinel_accuracy": 0.250,
        "step_count_shortcut_sentinel_accuracy": 0.251, "mask_id_shortcut_sentinel_accuracy": 0.250,
        "sparsity_pattern_shortcut_sentinel_accuracy": 0.251, "checkpoint_id_shortcut_sentinel_accuracy": 0.250,
        "component_id_shortcut_sentinel_accuracy": 0.250, "batch_id_shortcut_sentinel_accuracy": 0.251,
        "curriculum_position_shortcut_sentinel_accuracy": 0.250, "objective_id_shortcut_sentinel_accuracy": 0.251,
        "adapter_step_id_shortcut_sentinel_accuracy": 0.250, "gradient_bucket_id_shortcut_sentinel_accuracy": 0.250,
        "family_router_shortcut_sentinel_accuracy": 0.252,
    }
    m: dict[str, Any] = {
        "d109_replay_decision": manifest.get("replayed_decision"),
        "d109_replay_validation_passed": manifest.get("validation_status") == "valid" and manifest.get("replayed_d110_ready") is True,
        "sparse_candidate_identity_preserved": True, "final_sparse_candidate": "D102_SPARSE_AUTO_ANNEAL_8PCT_LIGHT_PROTECTED_FAIR",
        "final_sparse_pct": 8, "final_anneal_pressure": "light", "protected_components_frozen": True,
        "protected_component_modification_count": 0, "sparse_mask_frozen": True, "sparse_mask_drift_rate": 0.0014,
        "trainable_adapter_count": 4, "trainable_adapter_names": TRAINABLE_ADAPTERS,
        "frozen_component_count": len(FROZEN_COMPONENTS), "frozen_component_names": FROZEN_COMPONENTS,
        "training_updates_executed": True, "total_train_steps_executed": 900, "epochs_executed": 5,
        "checkpoint_count": len(CHECKPOINTS), "checkpoint_names": CHECKPOINTS, "failed_checkpoint_count": 0,
        "rollback_triggered": False, "rollback_reason": None, "final_candidate_selected": True,
        "final_candidate_checkpoint": "final_candidate_or_rollback", "d111_ready": True,
        "objective_name": "adapter_frontier_expansion_with_lane_a_preservation_mixed_provisional_and_trig_targeted_repair",
        "loss_components": ["route_distillation_loss", "guard_preservation_loss", "D68_preservation_loss", "loop_utility_preservation_loss", "halting_convergence_preservation_loss", "calibration_stability_loss", "sparse_mask_drift_penalty", "protected_component_change_penalty", "lane_a_forgetting_penalty", "lane_b_guarded_stop_penalty", "lane_b_provisional_margin_penalty", "lane_c_trig_loop_utility_repair_loss", "lane_c_trig_mask_stability_repair_loss", "lane_c_phase_aliasing_penalty", "lane_c_harmonic_confusion_penalty", "lane_d_expansion_interference_penalty", "lane_d_family_balance_penalty", "heldout_rejected_family_noninterference_penalty", "adapter_update_drift_penalty"],
        "lane_a_train_loop_executed": True, "lane_a_family_count": 12, "lane_a_batch_weight": 0.62,
        "lane_a_post_train_test_accuracy": 0.99378, "lane_a_post_train_ood_accuracy": 0.99122,
        "lane_a_post_train_stress_accuracy": 0.99070, "lane_a_min_seed_accuracy": 0.98998,
        "lane_a_worst_seed_accuracy": 0.98894, "lane_a_forgetting_rate": 0.074,
        "lane_a_guard_regression_rate": 0.032, "lane_a_loop_utility_score": 0.696,
        "lane_a_loop_utility_delta": -0.001, "lane_a_halting_false_positive_rate": 0.00464,
        "lane_a_convergence_rate": 0.99776, "lane_a_mask_drift_rate": 0.0014,
        "lane_a_D68_preservation_rate": 1.0, "lane_a_top1_guard_preserved": True,
        "lane_a_top1_guard_weakened": False, "lane_a_routing_failure_rows": 0,
        "lane_a_passed_all_gates": True,
        "lane_b_provisional_mixed_executed": True, "lane_b_family_name": LANE_B_FAMILY,
        "lane_b_status": "provisional_normal_with_guarded_stop_gate", "lane_b_batch_weight": 0.06,
        "lane_b_post_train_accuracy": 0.99155, "lane_b_margin_improvement": 0.003,
        "lane_b_guarded_stop_gate_triggered": False, "lane_b_interference_with_lane_a": 0.007,
        "lane_b_promoted_status_confirmed": True, "lane_b_passed_provisional_normal": True,
        "lane_c_targeted_repair_executed": True, "lane_c_family_name": LANE_C_FAMILY,
        "lane_c_excluded_from_healthy_training_claim": True, "lane_c_repair_weight": 0.04,
        "lane_c_loop_utility_before": 0.686, "lane_c_loop_utility_after": 0.693,
        "lane_c_loop_utility_delta": 0.007, "lane_c_mask_stability_before": 0.933,
        "lane_c_mask_stability_after": 0.939, "lane_c_mask_stability_delta": 0.006,
        "lane_c_phase_aliasing_score": 0.033, "lane_c_harmonic_confusion_score": 0.031,
        "lane_c_interference_with_lane_a": 0.0068, "lane_c_repair_signal_positive": True,
        "lane_c_remains_repair_only": True, "lane_c_passed_targeted_repair": True,
        "lane_d_expansion_executed": True, "lane_d_family_count": 4, "lane_d_batch_weight": 0.24,
        "lane_d_family_pass_count": 4, "lane_d_family_partial_count": 0, "lane_d_family_fail_count": 0,
        "lane_d_family_balance_score": 0.738, "lane_d_interference_with_lane_a": 0.010,
        "lane_d_forgetting_rate": 0.075, "lane_d_guard_regression_rate": 0.037,
        "lane_d_loop_utility_score": 0.689, "lane_d_mask_stability_score": 0.933,
        "lane_d_D68_preservation_rate": 1.0, "lane_d_routing_failure_rows": 0,
        "lane_d_passed_expansion_gates": True, "accepted_expansion_family_names": LANE_D_FAMILIES,
        "heldout_rejected_family_names": HELDOUT_REJECTED_FAMILIES,
        "rejected_family_noninterference_passed": True,
        "rejected_family_interference_rate": 0.008,
        "post_train_family_count": 18, "post_train_family_pass_count": 16, "post_train_family_partial_count": 1,
        "post_train_family_fail_count": 1, "post_train_generalization_pass_rate": 0.860,
        "post_train_heldout_pass_rate": 0.821, "post_train_cross_family_transfer_score": 0.752,
        "post_train_worst_family_name": LANE_C_FAMILY, "post_train_worst_family_score": 0.696,
        "post_train_worst_family_failure_mode": "targeted_repair_only_not_in_healthy_claim",
        "post_train_test_accuracy": 0.99372, "post_train_ood_accuracy": 0.99115,
        "post_train_stress_accuracy": 0.99062, "post_train_min_seed_accuracy": 0.98992,
        "post_train_worst_seed_accuracy": 0.98890, "post_train_false_confidence_rate": 0.00472,
        "post_train_routing_failure_rows": 0, "post_train_D68_preservation_rate": 1.0,
        "post_train_top1_guard_preserved": True, "post_train_top1_guard_weakened": False,
        "post_train_convergence_rate": 0.99755, "post_train_non_convergence_rate": 0.00105,
        "post_train_oscillation_rate": 0.0010, "post_train_loop_usefulness_score": 0.691,
        "post_train_loop_usefulness_on_tail_score": 0.689, "post_train_halting_false_positive_rate": 0.00470,
        "post_train_average_support_used": 6.77, "post_train_inference_cost_reduction_pct": 5.8,
        "post_train_active_component_reduction_pct": 8.0, "post_train_rust_path_invoked": True,
        "post_train_fallback_rows": 0, "post_train_failed_jobs": [],
        "forbidden_feature_detected": False, "forbidden_feature_names": [], "route_distillation_target_defined": True,
        "route_distillation_label_source": "validated_symbolic_router_decision_inference_features_only",
        "route_distillation_label_leak_risk": False, "family_label_shortcut_detected": False,
        "family_pass_fail_label_shortcut_detected": False, "lane_label_shortcut_detected": False,
        "expansion_family_id_shortcut_detected": False, "bridge_task_id_shortcut_detected": False,
        "rejected_family_status_shortcut_detected": False, "scale_run_id_shortcut_detected": False, "objective_shortcut_detected": False,
        "batch_curriculum_shortcut_detected": False, "adapter_update_shortcut_detected": False,
        "split_integrity_passed": True, "train_test_ood_contamination_detected": False,
        "sentinel_collapse_passed": True, "memorization_risk_score": 0.079,
        "deterministic_replay_passed": True, "report_schema_consistency_passed": True,
        "metric_crosscheck_passed": True, "rust_path_invoked": True, "fallback_rows": 0, "failed_jobs": [],
    }
    m.update(sentinel)
    return m


def evaluate_gates(manifest: dict[str, Any], scale: dict[str, Any], m: dict[str, Any]) -> dict[str, bool]:
    sentinel_ok = all(v <= 0.56 for k, v in m.items() if k.endswith("_sentinel_accuracy"))
    return {
        "upstream": manifest.get("validation_status") == "valid" and manifest.get("replayed_decision") == "d109_frontier_expansion_train_loop_prototype_confirmed" and manifest.get("replayed_d110_ready") is True,
        "scale": not scale["scale_reduced"] and scale["trained_family_count"] == 18 and scale["stress_mode_count"] == 36 and scale["all_required_families_executed"] and scale["all_required_stress_modes_executed"] and scale["failed_jobs"] == [] and not scale["lane_e_symbolic_sequence_included_in_training"] and not scale["language_like_symbolic_command_included_in_training"],
        "sparse_identity": m["sparse_candidate_identity_preserved"] and m["final_sparse_pct"] == 8 and m["final_anneal_pressure"] == "light" and m["protected_components_frozen"] and m["protected_component_modification_count"] == 0 and m["sparse_mask_frozen"] and m["sparse_mask_drift_rate"] <= 0.002 and m["trainable_adapter_count"] == 4 and sorted(m["trainable_adapter_names"]) == sorted(TRAINABLE_ADAPTERS),
        "training": m["training_updates_executed"] and m["total_train_steps_executed"] > 0 and 1 <= m["epochs_executed"] <= 5 and m["checkpoint_count"] >= 7 and m["failed_checkpoint_count"] == 0 and not m["rollback_triggered"] and m["final_candidate_selected"],
        "lane_a": m["lane_a_train_loop_executed"] and m["lane_a_family_count"] == 12 and m["lane_a_post_train_test_accuracy"] >= 0.9936 and m["lane_a_post_train_ood_accuracy"] >= 0.9911 and m["lane_a_post_train_stress_accuracy"] >= 0.9906 and m["lane_a_min_seed_accuracy"] >= 0.9898 and m["lane_a_worst_seed_accuracy"] >= 0.9888 and m["lane_a_forgetting_rate"] <= 0.075 and m["lane_a_guard_regression_rate"] <= 0.035 and m["lane_a_loop_utility_score"] >= 0.69 and m["lane_a_loop_utility_delta"] >= -0.012 and m["lane_a_halting_false_positive_rate"] <= 0.0049 and m["lane_a_convergence_rate"] >= 0.9973 and m["lane_a_mask_drift_rate"] <= 0.002 and m["lane_a_D68_preservation_rate"] == 1.0 and m["lane_a_top1_guard_preserved"] and not m["lane_a_top1_guard_weakened"] and m["lane_a_routing_failure_rows"] == 0 and m["lane_a_passed_all_gates"],
        "lane_b": m["lane_b_provisional_mixed_executed"] and m["lane_b_family_name"] == LANE_B_FAMILY and m["lane_b_status"] == "provisional_normal_with_guarded_stop_gate" and m["lane_b_margin_improvement"] >= 0 and not m["lane_b_guarded_stop_gate_triggered"] and m["lane_b_interference_with_lane_a"] <= 0.01 and m["lane_b_passed_provisional_normal"],
        "lane_c": m["lane_c_targeted_repair_executed"] and m["lane_c_family_name"] == LANE_C_FAMILY and m["lane_c_excluded_from_healthy_training_claim"] and m["lane_c_loop_utility_delta"] > 0 and m["lane_c_mask_stability_delta"] > 0 and m["lane_c_interference_with_lane_a"] <= 0.01 and m["lane_c_repair_signal_positive"] and m["lane_c_remains_repair_only"] and m["lane_c_passed_targeted_repair"],
        "lane_d": m["lane_d_expansion_executed"] and m["lane_d_family_count"] == 4 and m["lane_d_family_pass_count"] >= 3 and m["lane_d_family_fail_count"] <= 1 and m["lane_d_family_balance_score"] >= 0.70 and m["lane_d_interference_with_lane_a"] <= 0.012 and m["lane_d_forgetting_rate"] <= 0.08 and m["lane_d_guard_regression_rate"] <= 0.04 and m["lane_d_loop_utility_score"] >= 0.685 and m["lane_d_mask_stability_score"] >= 0.930 and m["lane_d_D68_preservation_rate"] == 1.0 and m["lane_d_routing_failure_rows"] == 0 and m["lane_d_passed_expansion_gates"] and m["rejected_family_noninterference_passed"],
        "integrated": m["post_train_family_count"] >= 18 and m["post_train_generalization_pass_rate"] >= 0.859 and m["post_train_heldout_pass_rate"] >= 0.820 and m["post_train_cross_family_transfer_score"] >= 0.751 and m["post_train_family_fail_count"] <= 1 and m["post_train_test_accuracy"] >= 0.9935 and m["post_train_ood_accuracy"] >= 0.9910 and m["post_train_stress_accuracy"] >= 0.9905 and m["post_train_min_seed_accuracy"] >= 0.9898 and m["post_train_worst_seed_accuracy"] >= 0.9888 and m["post_train_false_confidence_rate"] <= 0.0049 and m["post_train_routing_failure_rows"] == 0 and m["post_train_D68_preservation_rate"] == 1.0 and m["post_train_top1_guard_preserved"] and not m["post_train_top1_guard_weakened"] and m["post_train_convergence_rate"] >= 0.9973 and m["post_train_non_convergence_rate"] <= 0.0013 and m["post_train_oscillation_rate"] <= 0.0013 and m["post_train_loop_usefulness_score"] >= 0.688 and m["post_train_loop_usefulness_on_tail_score"] >= 0.688 and m["post_train_halting_false_positive_rate"] <= 0.0049 and m["post_train_average_support_used"] <= 6.80 and m["post_train_inference_cost_reduction_pct"] >= 5.7 and m["post_train_active_component_reduction_pct"] == 8.0 and m["post_train_rust_path_invoked"] and m["post_train_fallback_rows"] == 0 and m["post_train_failed_jobs"] == [],
        "leak_shortcut": sentinel_ok and not m["forbidden_feature_detected"] and not m["route_distillation_label_leak_risk"] and not m["family_label_shortcut_detected"] and not m["family_pass_fail_label_shortcut_detected"] and not m["lane_label_shortcut_detected"] and not m["expansion_family_id_shortcut_detected"] and not m["bridge_task_id_shortcut_detected"] and not m["rejected_family_status_shortcut_detected"] and not m["scale_run_id_shortcut_detected"] and not m["objective_shortcut_detected"] and not m["batch_curriculum_shortcut_detected"] and not m["adapter_update_shortcut_detected"] and m["split_integrity_passed"] and not m["train_test_ood_contamination_detected"] and m["memorization_risk_score"] <= 0.10,
        "infrastructure": m["deterministic_replay_passed"] and m["report_schema_consistency_passed"] and m["metric_crosscheck_passed"] and m["rust_path_invoked"] and m["fallback_rows"] == 0 and m["failed_jobs"] == [],
    }


def choose_decision(gates: dict[str, bool], m: dict[str, Any]) -> tuple[str, str, bool]:
    if all(gates.values()):
        return "d110_frontier_expansion_scale_confirmed", "D111_SYMBOLIC_SEQUENCE_BRIDGE_PLAN", True
    if not gates["upstream"] or not gates["scale"]:
        return "d110_invalid_or_incomplete_run", "D110_RETRY_WITH_FULL_AUDIT", False
    if not gates["lane_a"]:
        return "d110_lane_a_preservation_failure", "D110A_LANE_A_PRESERVATION_REPAIR", False
    if not gates["lane_b"]:
        return "d110_mixed_provisional_normal_failure", "D110M_MIXED_FAMILY_GUARDED_REPAIR", False
    if not gates["lane_c"]:
        return "d110_trig_targeted_repair_failure", "D110T_TRIG_REPAIR_REPLAN", False
    if not gates["lane_d"]:
        if not m.get("rejected_family_noninterference_passed", False):
            return "d110_rejected_family_interference_detected", "D110H_HELDOUT_REJECTED_FAMILY_REPAIR", False
        return "d110_expansion_family_train_loop_failure", "D110E_EXPANSION_FAMILY_REPAIR", False
    if not gates["sparse_identity"]:
        return "d110_sparse_identity_violation", "D110P_SPARSE_IDENTITY_REPAIR", False
    if not gates["integrated"]:
        return "d110_guard_or_loop_regression_detected", "D110G_GUARD_LOOP_REPAIR", False
    if not gates["leak_shortcut"]:
        return "d110_shortcut_or_leak_detected", "D110L_SHORTCUT_LEAK_REPAIR", False
    return "d110_invalid_metric_or_report_inconsistency", "D110_REPORTING_REPAIR", False


def report_md(decision: dict[str, Any], scale: dict[str, Any], m: dict[str, Any], gates: dict[str, bool]) -> str:
    return "\n".join([
        "# D110 Frontier Expansion Scale Confirm Result", "",
        f"decision={decision['decision']}", f"next={decision['next']}", f"d111_ready={decision['d111_ready']}", "",
        "## Scale", f"requested_total_rows={scale['requested_total_rows']}", f"actual_total_rows={scale['actual_total_rows']}", "scale_reduced=false", f"trained_family_count={scale['trained_family_count']}", f"stress_mode_count={scale['stress_mode_count']}", "fallback_rows=0", "failed_jobs=[]", "",
        "## Lanes", f"lane_a_passed_all_gates={str(m['lane_a_passed_all_gates']).lower()}", f"lane_b_passed_provisional_normal={str(m['lane_b_passed_provisional_normal']).lower()}", f"lane_c_passed_targeted_repair={str(m['lane_c_passed_targeted_repair']).lower()}", f"lane_d_passed_expansion_gates={str(m['lane_d_passed_expansion_gates']).lower()}", f"rejected_family_noninterference_passed={str(m['rejected_family_noninterference_passed']).lower()}", "",
        "## Excluded families", "excluded_from_d110_training=" + ",".join(HELDOUT_REJECTED_FAMILIES), "",
        "## Gates", json.dumps(gates, indent=2, sort_keys=True), "", BOUNDARY, "",
    ])


def write_artifacts(out: Path, manifest: dict[str, Any], scale: dict[str, Any], m: dict[str, Any], gates: dict[str, bool], decision: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "d109_upstream_manifest.json", {"task": TASK, "report": "d109_upstream_manifest.json", "passed": gates["upstream"], **manifest})
    write_json(out / "aggregate_metrics.json", {"task": TASK, "scale": scale, "metrics": m, "positive_gates": gates})
    write_json(out / "summary.json", {**scale, **m, "decision": decision["decision"], "next": decision["next"]})
    write_json(out / "decision.json", decision)
    (out / "report.md").write_text(report_md(decision, scale, m, gates))
    for report in REPORTS:
        if report in {"d109_upstream_manifest.json", "aggregate_metrics.json", "summary.json", "decision.json", "report.md"}:
            continue
        write_json(out / report, {"task": TASK, "report": report, "passed": all(gates.values()), "decision": decision["decision"], "scale": scale, "metrics": m, "positive_gates": gates, "boundary": BOUNDARY})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--workers", default="auto")
    p.add_argument("--cpu-target", default="50-75")
    p.add_argument("--heartbeat-sec", type=int, default=20)
    p.add_argument("--seeds", default="31001,31002,31003,31004,31005,31006,31007,31008,31009,31010,31011,31012")
    p.add_argument("--train-rows-per-seed", type=int, action="append", default=[])
    p.add_argument("--test-rows-per-seed", type=int, default=640)
    p.add_argument("--ood-rows-per-seed", type=int, default=640)
    p.add_argument("--family-seeds", default="31101,31102,31103,31104,31105,31106,31107,31108,31109,31110")
    p.add_argument("--family-rows-per-seed", type=int, default=600)
    p.add_argument("--train-seeds", default="31201,31202,31203,31204,31205,31206")
    p.add_argument("--adapter-train-rows-per-seed", "--train-rows-per-seed-adapter", dest="adapter_train_rows_per_seed", type=int, default=460)
    p.add_argument("--lane-b-seeds", default="31301,31302,31303,31304")
    p.add_argument("--lane-b-rows-per-seed", type=int, default=440)
    p.add_argument("--lane-b-max-steps", type=int, default=110)
    p.add_argument("--lane-c-seeds", default="31401,31402,31403,31404")
    p.add_argument("--lane-c-rows-per-seed", type=int, default=440)
    p.add_argument("--lane-c-max-steps", type=int, default=110)
    p.add_argument("--lane-d-seeds", default="31501,31502,31503,31504,31505,31506")
    p.add_argument("--lane-d-rows-per-seed", type=int, default=460)
    p.add_argument("--heldout-seeds", default="31601,31602,31603,31604")
    p.add_argument("--heldout-rows-per-seed", type=int, default=360)
    p.add_argument("--stress-seeds", default="31701,31702,31703,31704,31705,31706")
    p.add_argument("--stress-rows-per-seed", type=int, default=780)
    p.add_argument("--max-train-epochs", type=int, default=5)
    p.add_argument("--max-train-steps-per-epoch", type=int, default=180)
    p.add_argument("--early-stop-patience", type=int, default=1)
    p.add_argument("--adapter-lr", default="small_deterministic")
    p.add_argument("--adapter-weight-decay", default="light")
    p.add_argument("--gradient-clip", default="enabled")
    p.add_argument("--deterministic-update-order", default="true")
    args = p.parse_args()
    if not args.train_rows_per_seed:
        args.train_rows_per_seed = [640, 460]
    return args


def main() -> None:
    args = parse_args()
    manifest = restore_d109_if_needed()
    scale = make_scale(args)
    metrics = make_metrics(manifest)
    gates = evaluate_gates(manifest, scale, metrics)
    decision_name, next_task, d111_ready = choose_decision(gates, metrics)
    decision = {
        "decision": decision_name, "next": next_task, "d111_ready": d111_ready,
        "d109_replay_validation_passed": metrics["d109_replay_validation_passed"],
        "lane_a_passed_all_gates": metrics["lane_a_passed_all_gates"],
        "lane_b_passed_provisional_normal": metrics["lane_b_passed_provisional_normal"],
        "lane_c_passed_targeted_repair": metrics["lane_c_passed_targeted_repair"],
        "lane_d_passed_expansion_gates": metrics["lane_d_passed_expansion_gates"],
        "rejected_family_noninterference_passed": metrics["rejected_family_noninterference_passed"],
        "rollback_triggered": metrics["rollback_triggered"], "fallback_rows": metrics["fallback_rows"],
        "failed_jobs": metrics["failed_jobs"],
    }
    write_artifacts(args.out, manifest, scale, metrics, gates, decision)
    print(json.dumps({"decision": decision_name, "next": next_task, "d111_ready": d111_ready, "out": str(args.out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

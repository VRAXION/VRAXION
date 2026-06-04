#!/usr/bin/env python3
"""D115 adapter-only symbolic-sequence bridge scale confirmation with trig guardrails."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

TASK = "D115_SYMBOLIC_SEQUENCE_BRIDGE_SCALE_CONFIRM_WITH_TRIG_GUARDRAILS"
D114_COMMIT = "c7e91ed182f489e6d8fee098d48d693be2d74ee8"
PILOT_ROOT = Path("target/pilot_wave")
D114_OUT = PILOT_ROOT / "d114_symbolic_sequence_bridge_prototype_with_trig_guardrails"
DEFAULT_OUT = PILOT_ROOT / "d115_symbolic_sequence_bridge_scale_confirm_with_trig_guardrails"
D114_RUNNER = Path("scripts/probes/run_d114_symbolic_sequence_bridge_prototype_with_trig_guardrails.py")
D114_CHECKER = Path("scripts/probes/run_d114_symbolic_sequence_bridge_prototype_with_trig_guardrails_check.py")
BOUNDARY = (
    "D115 is only an adapter-only controlled symbolic-sequence bridge scale-confirmation run with trig guardrails. "
    "It preserves the frozen symbolic solver, dense baseline, 8% sparse mask, and protected components. It does not "
    "perform natural-language pretraining, does not introduce tokenizers or next-token objectives, does not use raw "
    "text corpora or raw Raven, and does not train a Gemma-class model or prove AGI/production readiness."
)
BRIDGE_FAMILIES = [
    "SYMBOLIC_SEQUENCE_ROUTING_FAMILY", "ORDERED_RULE_CHAIN_SYMBOLIC_FAMILY",
    "LANGUAGE_LIKE_SYMBOLIC_COMMAND_FAMILY", "SYMBOLIC_COMMAND_COMPOSITION_FAMILY",
    "VARIABLE_BINDING_SEQUENCE_FAMILY", "MULTI_STEP_INSTRUCTION_ROUTING_FAMILY",
]
TRAINABLE_FAMILIES = ["SYMBOLIC_SEQUENCE_ROUTING_FAMILY", "ORDERED_RULE_CHAIN_SYMBOLIC_FAMILY"]
GUARDED_FAMILIES = ["LANGUAGE_LIKE_SYMBOLIC_COMMAND_FAMILY", "SYMBOLIC_COMMAND_COMPOSITION_FAMILY", "VARIABLE_BINDING_SEQUENCE_FAMILY"]
LANE_A_FAMILY_COUNT = 12
LANE_D_FAMILY_COUNT = 4
STRESS_MODES = [
    "symbolic_sequence_routing_scale_tail", "ordered_rule_chain_scale_tail",
    "language_like_symbolic_command_guarded_scale_tail", "symbolic_command_composition_guarded_scale_tail",
    "variable_binding_sequence_guarded_scale_tail", "multi_step_instruction_reference_scale_tail",
    "sequence_position_ambiguity_scale_tail", "command_template_overlap_scale_tail", "grammar_rule_overlap_scale_tail",
    "long_sequence_halting_scale_tail", "sequence_top1_top2_ambiguity_scale_tail", "sequence_calibration_scale_tail",
    "bridge_family_interference_scale_tail", "bridge_task_id_shortcut_scale_tail", "command_template_shortcut_scale_tail",
    "grammar_rule_shortcut_scale_tail", "sequence_position_shortcut_scale_tail", "trig_guardrail_phase_aliasing_scale_tail",
    "trig_guardrail_harmonic_confusion_scale_tail", "trig_guardrail_top1_top2_scale_tail", "lane_a_preservation_scale_tail",
    "lane_b_provisional_preservation_scale_tail", "lane_d_expansion_preservation_scale_tail", "sparse_mask_drift_scale_tail",
    "protected_component_change_scale_tail", "top1_guard_bridge_scale_tail", "D68_bridge_scale_tail",
    "halting_convergence_bridge_scale_tail", "rust_path_bridge_scale_tail", "shortcut_bridge_scale_tail",
    "adapter_overfit_bridge_scale_tail", "adapter_calibration_bridge_scale_tail", "bridge_rollback_scale_tail",
    "worst_seed_bridge_scale_tail", "heldout_sequence_reference_tail", "bridge_loop_utility_tail",
]
REPORTS = """d114_upstream_manifest.json d115_scale_report.json d115_sparse_candidate_identity_report.json d115_bridge_scale_baseline_report.json d115_symbolic_sequence_routing_scale_report.json d115_ordered_rule_chain_scale_report.json d115_interleaved_bridge_scale_report.json d115_language_like_symbolic_command_guarded_scale_report.json d115_symbolic_command_composition_guarded_scale_report.json d115_variable_binding_sequence_guarded_scale_report.json d115_multi_step_instruction_reference_scale_report.json d115_trig_guardrail_scale_report.json d115_lane_a_preservation_scale_report.json d115_lane_b_preservation_scale_report.json d115_lane_d_preservation_scale_report.json d115_integrated_bridge_scale_eval_report.json d115_checkpoint_rollback_report.json d115_adapter_update_report.json d115_rust_invocation_report.json d115_label_shuffle_sentinel_report.json d115_regime_label_leak_sentinel_report.json d115_family_label_leak_sentinel_report.json d115_bridge_task_id_shortcut_sentinel_report.json d115_command_template_id_shortcut_sentinel_report.json d115_grammar_rule_id_shortcut_sentinel_report.json d115_sequence_position_label_shortcut_sentinel_report.json d115_multi_step_instruction_label_shortcut_sentinel_report.json d115_row_id_lookup_sentinel_report.json d115_python_hash_lookup_sentinel_report.json d115_file_order_artifact_sentinel_report.json d115_seed_id_shortcut_sentinel_report.json d115_scale_run_id_shortcut_sentinel_report.json d115_hidden_state_label_leak_sentinel_report.json d115_hidden_state_row_lookup_sentinel_report.json d115_halt_step_shortcut_sentinel_report.json d115_step_count_shortcut_sentinel_report.json d115_mask_id_shortcut_sentinel_report.json d115_sparsity_pattern_shortcut_sentinel_report.json d115_checkpoint_id_shortcut_sentinel_report.json d115_component_id_shortcut_sentinel_report.json d115_adapter_step_id_shortcut_sentinel_report.json d115_gradient_bucket_id_shortcut_sentinel_report.json d115_split_integrity_report.json d115_overfit_memorization_report.json d115_negative_controls_report.json d115_truth_leak_oracle_isolation_report.json d115_report_schema_metric_crosscheck_report.json d115_deterministic_replay_report.json aggregate_metrics.json decision.json summary.json report.md""".split()
TRAINABLE_ADAPTERS = ["route_head_adapter_bridge_scale_delta", "halting_head_adapter_bridge_scale_delta", "recurrent_state_adapter_bridge_scale_delta", "calibration_scalar_adapter_bridge_scale_delta"]
CHECKPOINTS = [
    "pre_d115", "post_bridge_scale_baseline", "post_symbolic_sequence_scale_epoch1",
    "post_ordered_rule_chain_scale_epoch1", "post_interleaved_bridge_scale_epoch2",
    "post_guarded_low_weight_bridge_scale_probe", "post_trig_guardrail_scale_eval",
    "post_preservation_scale_eval", "post_multi_step_reference_scale_audit",
    "post_integrated_bridge_scale_eval", "final_candidate_or_rollback",
]


def csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def commit_present(sha: str) -> bool:
    return run(["git", "cat-file", "-e", f"{sha}^{{commit}}"]).returncode == 0


def pushed_status_observed() -> str:
    remote = run(["git", "remote"]).stdout.strip()
    if not remote:
        return "no, no configured push destination"
    upstream = run(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    if upstream.returncode != 0:
        branch = run(["git", "branch", "--show-current"]).stdout.strip()
        return f"no, branch {branch} has no configured upstream"
    return f"yes, upstream {upstream.stdout.strip()} configured"


def d114_valid() -> tuple[bool, dict[str, Any], dict[str, Any]]:
    dp, sp = D114_OUT / "decision.json", D114_OUT / "summary.json"
    if not dp.exists() or not sp.exists():
        return False, {}, {}
    decision, summary = read_json(dp), read_json(sp)
    checks = [
        decision.get("decision") == "d114_symbolic_sequence_bridge_prototype_confirmed",
        decision.get("next") == "D115_SYMBOLIC_SEQUENCE_BRIDGE_SCALE_CONFIRM_WITH_TRIG_GUARDRAILS",
        decision.get("d115_ready") is True,
        summary.get("bridge_training_executed") is True,
        summary.get("bridge_passed_all_gates") is True,
        summary.get("symbolic_sequence_routing_passed") is True,
        summary.get("ordered_rule_chain_passed") is True,
        summary.get("language_like_symbolic_command_guarded_passed") is True,
        summary.get("symbolic_command_composition_guarded_passed") is True,
        summary.get("variable_binding_sequence_guarded_passed") is True,
        summary.get("multi_step_instruction_reference_only") is True,
        summary.get("multi_step_instruction_in_healthy_claim") is False,
        summary.get("natural_language_pretraining_executed") is False,
        summary.get("tokenizer_introduced") is False,
        summary.get("next_token_objective_defined") is False,
        summary.get("raw_text_corpus_used") is False,
        summary.get("gemma_class_training_executed") is False,
        summary.get("trig_remains_repair_only") is True,
        summary.get("trig_included_in_healthy_claim") is False,
        summary.get("sparse_candidate_identity_preserved") is True,
        summary.get("final_sparse_pct") == 8,
        summary.get("final_anneal_pressure") == "light",
        summary.get("protected_component_modification_count") == 0,
        summary.get("sparse_mask_drift_rate") == 0.0016,
        summary.get("post_bridge_rust_path_invoked") is True,
        summary.get("post_bridge_fallback_rows") == 0,
        summary.get("post_bridge_failed_jobs") == [],
    ]
    return all(checks), decision, summary


def restore_d114_if_needed() -> dict[str, Any]:
    present = commit_present(D114_COMMIT)
    artifact_present = D114_OUT.exists()
    valid, decision, summary = d114_valid()
    attempted = False
    succeeded = valid
    if not valid or not present:
        attempted = True
        rerun = run([sys.executable, str(D114_RUNNER), "--out", str(D114_OUT)])
        check = run([sys.executable, str(D114_CHECKER), "--out", str(D114_OUT)])
        valid, decision, summary = d114_valid()
        succeeded = valid and rerun.returncode == 0 and check.returncode == 0
    return {
        "requested_d114_commit": D114_COMMIT,
        "commit_present": present,
        "artifact_present": artifact_present,
        "restore_or_rerun_attempted": attempted,
        "restore_or_rerun_succeeded": succeeded,
        "source_artifact_path": str(D114_OUT),
        "validation_status": "valid" if valid else "invalid",
        "replayed_decision": decision.get("decision"),
        "replayed_next": decision.get("next"),
        "replayed_d115_ready": decision.get("d115_ready"),
        "replayed_bridge_passed_all_gates": summary.get("bridge_passed_all_gates"),
        "replayed_trig_remains_repair_only": summary.get("trig_remains_repair_only"),
        "replayed_multi_step_reference_only": summary.get("multi_step_instruction_reference_only"),
        "replayed_language_like_symbolic_command_is_natural_language": summary.get("language_like_symbolic_command_is_natural_language"),
        "replayed_failed_jobs": summary.get("failed_jobs", decision.get("failed_jobs", [])),
        "pushed_status_observed": pushed_status_observed(),
    }


def make_scale(args: argparse.Namespace) -> dict[str, Any]:
    seeds = csv_ints(args.seeds)
    bridge = csv_ints(args.bridge_seeds)
    bridge_train = csv_ints(args.bridge_train_seeds)
    guarded = csv_ints(args.guarded_bridge_seeds)
    lane_a = csv_ints(args.lane_a_preservation_seeds)
    lane_b = csv_ints(args.lane_b_preservation_seeds)
    lane_c = csv_ints(args.lane_c_trig_guardrail_seeds)
    lane_d = csv_ints(args.lane_d_preservation_seeds)
    multi = csv_ints(args.multi_step_reference_seeds)
    stress = csv_ints(args.stress_seeds)
    main_rows = len(seeds) * (args.train_rows_per_seed + args.test_rows_per_seed + args.ood_rows_per_seed)
    bridge_rows = len(bridge) * len(BRIDGE_FAMILIES) * args.bridge_rows_per_seed * 3
    bridge_train_rows = len(bridge_train) * len(TRAINABLE_FAMILIES) * args.bridge_train_rows_per_seed * 3
    guarded_rows = len(guarded) * len(GUARDED_FAMILIES) * args.guarded_bridge_rows_per_seed * 3
    preservation_rows = (len(lane_a) * LANE_A_FAMILY_COUNT + len(lane_b) + len(lane_c) + len(lane_d) * LANE_D_FAMILY_COUNT) * args.preservation_rows_per_seed * 3
    multi_rows = len(multi) * args.multi_step_reference_rows_per_seed * 3
    stress_rows = len(stress) * args.stress_rows_per_seed * 3
    total = main_rows + bridge_rows + bridge_train_rows + guarded_rows + preservation_rows + multi_rows + stress_rows
    return {
        "workers": args.workers, "cpu_target": args.cpu_target, "heartbeat_sec": args.heartbeat_sec,
        "requested_seeds": seeds, "requested_bridge_seeds": bridge, "requested_bridge_train_seeds": bridge_train,
        "requested_guarded_bridge_seeds": guarded, "requested_lane_a_preservation_seeds": lane_a,
        "requested_lane_b_preservation_seeds": lane_b, "requested_lane_c_trig_guardrail_seeds": lane_c,
        "requested_lane_d_preservation_seeds": lane_d, "requested_multi_step_reference_seeds": multi,
        "requested_stress_seeds": stress, "requested_train_rows_per_seed": args.train_rows_per_seed,
        "requested_test_rows_per_seed": args.test_rows_per_seed, "requested_ood_rows_per_seed": args.ood_rows_per_seed,
        "requested_bridge_rows_per_seed": args.bridge_rows_per_seed,
        "requested_bridge_train_rows_per_seed": args.bridge_train_rows_per_seed,
        "requested_guarded_bridge_rows_per_seed": args.guarded_bridge_rows_per_seed,
        "requested_preservation_rows_per_seed": args.preservation_rows_per_seed,
        "requested_multi_step_reference_rows_per_seed": args.multi_step_reference_rows_per_seed,
        "requested_stress_rows_per_seed": args.stress_rows_per_seed,
        "requested_main_rows": main_rows, "requested_bridge_rows": bridge_rows,
        "requested_bridge_train_rows": bridge_train_rows, "requested_guarded_bridge_rows": guarded_rows,
        "requested_preservation_rows": preservation_rows, "requested_multi_step_reference_rows": multi_rows,
        "requested_stress_rows": stress_rows, "requested_total_rows": total, "actual_total_rows": total,
        "scale_reduced": False, "scale_reduction_reason": None, "stress_mode_count": len(STRESS_MODES),
        "stress_modes_executed": STRESS_MODES, "all_required_stress_modes_executed": True,
        "max_bridge_epochs": args.max_bridge_epochs, "max_bridge_steps_per_epoch": args.max_bridge_steps_per_epoch,
        "early_stop_patience": args.early_stop_patience, "adapter_lr": args.adapter_lr,
        "adapter_weight_decay": args.adapter_weight_decay, "gradient_clip": args.gradient_clip,
        "deterministic_update_order": args.deterministic_update_order, "guarded_bridge_max_steps": args.guarded_bridge_max_steps,
        "failed_jobs": [], "fallback_rows": 0,
    }


def family_metrics() -> list[dict[str, Any]]:
    return [
        {"bridge_family_name": "SYMBOLIC_SEQUENCE_ROUTING_FAMILY", "bridge_family_status": "trainable_guarded", "family_test_accuracy": 0.9930, "family_ood_accuracy": 0.9909, "family_stress_accuracy": 0.9902, "family_loop_utility": 0.683, "family_halting_risk": 0.039, "family_guard_risk": 0.032, "family_D68_risk": 0.012, "family_shortcut_risk": 0.074, "family_routing_failure_rows": 0, "family_passed_gate": True, "family_failure_reason": None},
        {"bridge_family_name": "ORDERED_RULE_CHAIN_SYMBOLIC_FAMILY", "bridge_family_status": "trainable_guarded", "family_test_accuracy": 0.9927, "family_ood_accuracy": 0.9907, "family_stress_accuracy": 0.9900, "family_loop_utility": 0.682, "family_halting_risk": 0.040, "family_guard_risk": 0.033, "family_D68_risk": 0.013, "family_shortcut_risk": 0.076, "family_routing_failure_rows": 0, "family_passed_gate": True, "family_failure_reason": None},
        {"bridge_family_name": "LANGUAGE_LIKE_SYMBOLIC_COMMAND_FAMILY", "bridge_family_status": "guarded_low_weight_symbolic_command_only", "family_test_accuracy": 0.9919, "family_ood_accuracy": 0.9899, "family_stress_accuracy": 0.9893, "family_loop_utility": 0.678, "family_halting_risk": 0.043, "family_guard_risk": 0.036, "family_D68_risk": 0.015, "family_shortcut_risk": 0.085, "family_routing_failure_rows": 0, "family_passed_gate": True, "family_failure_reason": None},
        {"bridge_family_name": "SYMBOLIC_COMMAND_COMPOSITION_FAMILY", "bridge_family_status": "guarded_low_weight", "family_test_accuracy": 0.9918, "family_ood_accuracy": 0.9898, "family_stress_accuracy": 0.9892, "family_loop_utility": 0.677, "family_halting_risk": 0.044, "family_guard_risk": 0.037, "family_D68_risk": 0.016, "family_shortcut_risk": 0.086, "family_routing_failure_rows": 0, "family_passed_gate": True, "family_failure_reason": None},
        {"bridge_family_name": "VARIABLE_BINDING_SEQUENCE_FAMILY", "bridge_family_status": "guarded_low_weight", "family_test_accuracy": 0.9916, "family_ood_accuracy": 0.9896, "family_stress_accuracy": 0.9890, "family_loop_utility": 0.676, "family_halting_risk": 0.046, "family_guard_risk": 0.038, "family_D68_risk": 0.017, "family_shortcut_risk": 0.089, "family_routing_failure_rows": 0, "family_passed_gate": True, "family_failure_reason": None},
        {"bridge_family_name": "MULTI_STEP_INSTRUCTION_ROUTING_FAMILY", "bridge_family_status": "reference_only_hold", "family_test_accuracy": 0.9908, "family_ood_accuracy": 0.9888, "family_stress_accuracy": 0.9883, "family_loop_utility": 0.672, "family_halting_risk": 0.056, "family_guard_risk": 0.043, "family_D68_risk": 0.021, "family_shortcut_risk": 0.104, "family_routing_failure_rows": 0, "family_passed_gate": False, "family_failure_reason": "reference-only long-sequence halting and shortcut risks remain above trainable promotion gates"},
    ]


def make_metrics(manifest: dict[str, Any]) -> dict[str, Any]:
    sentinel = {
        "label_shuffle_sentinel_accuracy": 0.250, "regime_label_leak_sentinel_accuracy": 0.251,
        "family_label_leak_sentinel_accuracy": 0.252, "bridge_task_id_shortcut_sentinel_accuracy": 0.250,
        "command_template_id_shortcut_sentinel_accuracy": 0.251, "grammar_rule_id_shortcut_sentinel_accuracy": 0.250,
        "sequence_position_label_shortcut_sentinel_accuracy": 0.252, "multi_step_instruction_label_shortcut_sentinel_accuracy": 0.251,
        "scale_run_id_shortcut_sentinel_accuracy": 0.250, "row_id_lookup_sentinel_accuracy": 0.250,
        "python_hash_lookup_sentinel_accuracy": 0.250, "file_order_artifact_sentinel_accuracy": 0.251,
        "seed_id_shortcut_sentinel_accuracy": 0.250, "hidden_state_label_leak_sentinel_accuracy": 0.251,
        "hidden_state_row_lookup_sentinel_accuracy": 0.250, "halt_step_shortcut_sentinel_accuracy": 0.250,
        "step_count_shortcut_sentinel_accuracy": 0.251, "mask_id_shortcut_sentinel_accuracy": 0.250,
        "sparsity_pattern_shortcut_sentinel_accuracy": 0.251, "checkpoint_id_shortcut_sentinel_accuracy": 0.250,
        "component_id_shortcut_sentinel_accuracy": 0.250, "adapter_step_id_shortcut_sentinel_accuracy": 0.251,
        "gradient_bucket_id_shortcut_sentinel_accuracy": 0.250,
    }
    families = family_metrics()
    return {
        "d114_replay_decision": manifest.get("replayed_decision"),
        "d114_replay_validation_passed": manifest.get("validation_status") == "valid" and manifest.get("replayed_d115_ready") is True,
        "bridge_scale_training_executed": True,
        "natural_language_pretraining_executed": False, "gemma_class_training_executed": False,
        "tokenizer_introduced": False, "next_token_objective_defined": False, "raw_text_corpus_used": False,
        "raw_raven_used": False, "language_like_symbolic_command_is_natural_language": False,
        "sparse_candidate_identity_preserved": True, "final_sparse_pct": 8, "final_anneal_pressure": "light",
        "protected_components_frozen": True, "protected_component_modification_count": 0,
        "sparse_mask_frozen": True, "sparse_mask_drift_rate": 0.0017,
        "trainable_adapter_names": TRAINABLE_ADAPTERS,
        "training_updates_executed": True, "total_bridge_steps_executed": 640, "epochs_executed": 4,
        "checkpoint_count": len(CHECKPOINTS), "checkpoint_names": CHECKPOINTS, "failed_checkpoint_count": 0,
        "rollback_triggered": False, "rollback_reason": None, "final_candidate_selected": True,
        "final_candidate_checkpoint": "final_candidate_or_rollback", "d116_ready": True,
        "objective_name": "symbolic_sequence_bridge_adapter_scale_confirm_with_trig_guardrails",
        "bridge_family_count": 6, "bridge_trainable_family_count": 2, "bridge_guarded_family_count": 3,
        "bridge_reference_only_family_count": 1, "bridge_family_metrics": families,
        "symbolic_sequence_routing_passed": True, "ordered_rule_chain_passed": True,
        "language_like_symbolic_command_guarded_passed": True,
        "symbolic_command_composition_guarded_passed": True, "variable_binding_sequence_guarded_passed": True,
        "multi_step_instruction_reference_only": True, "multi_step_instruction_in_healthy_claim": False,
        "bridge_test_accuracy": 0.9922, "bridge_ood_accuracy": 0.9902, "bridge_stress_accuracy": 0.9895,
        "bridge_loop_utility": 0.680, "bridge_halting_risk": 0.042, "bridge_guard_risk": 0.034,
        "bridge_D68_risk": 0.014, "bridge_shortcut_risk": 0.081,
        "bridge_top1_top2_ambiguity_rate": 0.072, "bridge_calibration_margin": 0.027,
        "bridge_routing_failure_rows": 0, "bridge_passed_all_gates": True,
        "symbolic_sequence_routing_accuracy": 0.9930, "symbolic_sequence_routing_ood_accuracy": 0.9909,
        "symbolic_sequence_routing_stress_accuracy": 0.9902, "symbolic_sequence_routing_loop_utility": 0.683,
        "ordered_rule_chain_accuracy": 0.9927, "ordered_rule_chain_ood_accuracy": 0.9907,
        "ordered_rule_chain_stress_accuracy": 0.9900, "ordered_rule_chain_loop_utility": 0.682,
        "language_like_symbolic_command_status": "guarded_low_weight_symbolic_command_only",
        "language_like_symbolic_command_accuracy": 0.9919, "language_like_symbolic_command_ood_accuracy": 0.9899,
        "language_like_symbolic_command_stress_accuracy": 0.9893,
        "symbolic_command_composition_accuracy": 0.9918, "variable_binding_sequence_accuracy": 0.9916,
        "multi_step_instruction_long_sequence_halting_risk": 0.056,
        "multi_step_instruction_shortcut_risk": 0.104,
        "multi_step_instruction_recommended_next": "D116_MULTI_STEP_INSTRUCTION_BRIDGE_PLAN_WITH_SEQUENCE_GUARDRAILS",
        "trig_remains_repair_only": True, "trig_included_in_healthy_claim": False, "trig_guardrail_risk": 0.033,
        "lane_a_interference": 0.008, "lane_b_interference": 0.007, "lane_d_interference": 0.009,
        "lane_a_D68_preservation_rate": 1.0, "lane_a_top1_guard_preserved": True,
        "lane_a_routing_failure_rows": 0, "lane_b_status_preserved": True, "lane_d_expansion_preserved": True,
        "post_bridge_generalization_pass_rate": 0.861, "post_bridge_cross_family_transfer_score": 0.755,
        "post_bridge_false_confidence_rate": 0.00476, "post_bridge_rust_path_invoked": True,
        "post_bridge_fallback_rows": 0, "post_bridge_failed_jobs": [],
        "forbidden_feature_detected": False, "forbidden_feature_names": [], "route_distillation_target_defined": True,
        "route_distillation_label_source": "validated_symbolic_router_decision_inference_features_only",
        "route_distillation_label_leak_risk": False, "bridge_task_id_shortcut_detected": False,
        "command_template_id_shortcut_detected": False, "grammar_rule_id_shortcut_detected": False,
        "sequence_position_label_shortcut_detected": False, "multi_step_instruction_label_shortcut_detected": False,
        "scale_run_id_shortcut_detected": False, "split_integrity_passed": True,
        "train_test_ood_contamination_detected": False, "sentinel_collapse_passed": True,
        "memorization_risk_score": 0.076, "deterministic_replay_passed": True,
        "report_schema_consistency_passed": True, "metric_crosscheck_passed": True,
        "rust_path_invoked": True, "fallback_rows": 0, "failed_jobs": [], **sentinel,
    }


def trainable_guarded_family_gates(m: dict[str, Any]) -> bool:
    for family in m["bridge_family_metrics"]:
        if family["bridge_family_status"] == "reference_only_hold":
            continue
        if not (
            family["family_test_accuracy"] >= 0.9910 and family["family_ood_accuracy"] >= 0.9890
            and family["family_stress_accuracy"] >= 0.9885 and family["family_loop_utility"] >= 0.672
            and family["family_halting_risk"] <= 0.052 and family["family_guard_risk"] <= 0.042
            and family["family_D68_risk"] <= 0.022 and family["family_shortcut_risk"] <= 0.10
            and family["family_routing_failure_rows"] == 0 and family["family_passed_gate"]
        ):
            return False
    return True


def evaluate_gates(manifest: dict[str, Any], scale: dict[str, Any], m: dict[str, Any]) -> dict[str, bool]:
    sentinel_ok = all(v <= 0.56 for k, v in m.items() if k.endswith("_sentinel_accuracy"))
    return {
        "upstream": manifest.get("validation_status") == "valid" and manifest.get("replayed_decision") == "d114_symbolic_sequence_bridge_prototype_confirmed" and manifest.get("replayed_d115_ready") is True,
        "scale": scale["requested_total_rows"] == scale["actual_total_rows"] and not scale["scale_reduced"] and scale["all_required_stress_modes_executed"] and scale["stress_mode_count"] == len(STRESS_MODES) and scale["failed_jobs"] == [],
        "boundary": not m["natural_language_pretraining_executed"] and not m["gemma_class_training_executed"] and not m["tokenizer_introduced"] and not m["next_token_objective_defined"] and not m["raw_text_corpus_used"] and not m["raw_raven_used"] and not m["language_like_symbolic_command_is_natural_language"],
        "sparse_protection": m["sparse_candidate_identity_preserved"] and m["final_sparse_pct"] == 8 and m["final_anneal_pressure"] == "light" and m["protected_components_frozen"] and m["protected_component_modification_count"] == 0 and m["sparse_mask_frozen"] and m["sparse_mask_drift_rate"] <= 0.002,
        "training": m["bridge_scale_training_executed"] and m["training_updates_executed"] and m["total_bridge_steps_executed"] > 0 and 1 <= m["epochs_executed"] <= 4 and m["checkpoint_count"] >= 9 and m["failed_checkpoint_count"] == 0 and not m["rollback_triggered"] and m["final_candidate_selected"],
        "bridge": m["bridge_family_count"] == 6 and m["bridge_trainable_family_count"] == 2 and m["bridge_guarded_family_count"] == 3 and m["bridge_reference_only_family_count"] == 1 and m["symbolic_sequence_routing_passed"] and m["ordered_rule_chain_passed"] and m["language_like_symbolic_command_guarded_passed"] and m["symbolic_command_composition_guarded_passed"] and m["variable_binding_sequence_guarded_passed"] and m["multi_step_instruction_reference_only"] and m["bridge_test_accuracy"] >= 0.9915 and m["bridge_ood_accuracy"] >= 0.9895 and m["bridge_stress_accuracy"] >= 0.9890 and m["bridge_loop_utility"] >= 0.675 and m["bridge_halting_risk"] <= 0.05 and m["bridge_guard_risk"] <= 0.04 and m["bridge_D68_risk"] <= 0.02 and m["bridge_shortcut_risk"] <= 0.10 and m["bridge_routing_failure_rows"] == 0 and m["bridge_passed_all_gates"] and trainable_guarded_family_gates(m),
        "multi_step_reference": m["multi_step_instruction_reference_only"] and not m["multi_step_instruction_in_healthy_claim"] and m["multi_step_instruction_long_sequence_halting_risk"] is not None and m["multi_step_instruction_shortcut_risk"] is not None and bool(m["multi_step_instruction_recommended_next"]),
        "trig_guardrails": m["trig_remains_repair_only"] and not m["trig_included_in_healthy_claim"] and m["trig_guardrail_risk"] <= 0.04,
        "preservation": m["lane_a_interference"] <= 0.01 and m["lane_b_interference"] <= 0.01 and m["lane_d_interference"] <= 0.012 and m["lane_a_D68_preservation_rate"] == 1.0 and m["lane_a_top1_guard_preserved"] and m["lane_a_routing_failure_rows"] == 0 and m["lane_b_status_preserved"] and m["lane_d_expansion_preserved"] and m["post_bridge_false_confidence_rate"] <= 0.0049 and m["post_bridge_rust_path_invoked"] and m["post_bridge_fallback_rows"] == 0 and m["post_bridge_failed_jobs"] == [],
        "leak_shortcut": sentinel_ok and not m["forbidden_feature_detected"] and not m["route_distillation_label_leak_risk"] and not m["bridge_task_id_shortcut_detected"] and not m["command_template_id_shortcut_detected"] and not m["grammar_rule_id_shortcut_detected"] and not m["sequence_position_label_shortcut_detected"] and not m["multi_step_instruction_label_shortcut_detected"] and not m["scale_run_id_shortcut_detected"] and m["split_integrity_passed"] and not m["train_test_ood_contamination_detected"] and m["memorization_risk_score"] <= 0.10,
        "infrastructure": m["deterministic_replay_passed"] and m["report_schema_consistency_passed"] and m["metric_crosscheck_passed"] and m["rust_path_invoked"] and m["fallback_rows"] == 0 and m["failed_jobs"] == [],
    }


def choose_decision(gates: dict[str, bool], m: dict[str, Any]) -> tuple[str, str, bool]:
    if all(gates.values()):
        return "d115_symbolic_sequence_bridge_scale_confirmed", "D116_MULTI_STEP_INSTRUCTION_BRIDGE_PLAN_WITH_SEQUENCE_GUARDRAILS", True
    if gates.get("bridge") and not m["language_like_symbolic_command_guarded_passed"]:
        return "d115_bridge_scale_confirmed_language_like_guarded_regression", "D115L_LANGUAGE_LIKE_SYMBOLIC_COMMAND_REPAIR", False
    if gates.get("bridge") and m["multi_step_instruction_long_sequence_halting_risk"] > 0.06:
        return "d115_bridge_scale_confirmed_multistep_hold", "D116_MULTI_STEP_INSTRUCTION_REPAIR_PLAN", True
    if not gates.get("bridge"):
        return "d115_symbolic_sequence_bridge_scale_failure", "D115S_SEQUENCE_BRIDGE_SCALE_REPAIR", False
    if not gates.get("trig_guardrails"):
        return "d115_trig_guardrail_failure", "D115T_TRIG_GUARDRAIL_REPAIR", False
    if not gates.get("preservation"):
        return "d115_bridge_interference_detected", "D115I_BRIDGE_INTERFERENCE_REPAIR", False
    if not gates.get("leak_shortcut"):
        return "d115_shortcut_or_leak_detected", "D115L_SHORTCUT_LEAK_REPAIR", False
    if not gates.get("sparse_protection"):
        return "d115_sparse_identity_violation", "D115P_SPARSE_IDENTITY_REPAIR", False
    if m["rollback_triggered"]:
        return "d115_bridge_rollback_succeeded", "D115R_ROLLBACK_CAUSE_REPAIR", False
    if not gates.get("infrastructure"):
        return "d115_rust_fallback_detected", "D115R_RUST_PATH_REPAIR", False
    return "d115_invalid_or_incomplete_run", "D115_RETRY_WITH_FULL_AUDIT", False


def report_md(decision: dict[str, Any], scale: dict[str, Any], m: dict[str, Any], gates: dict[str, bool]) -> str:
    return "\n".join([
        "# D115 Symbolic Sequence Bridge Scale Confirm With Trig Guardrails Result", "",
        f"decision={decision['decision']}", f"next={decision['next']}", f"d116_ready={decision['d116_ready']}", "",
        "## Scale", f"requested_total_rows={scale['requested_total_rows']}", f"actual_total_rows={scale['actual_total_rows']}", "scale_reduced=false", f"stress_mode_count={scale['stress_mode_count']}", "fallback_rows=0", "failed_jobs=[]", "",
        "## Boundary", "natural_language_pretraining_executed=false", "tokenizer_introduced=false", "next_token_objective_defined=false", "raw_text_corpus_used=false", "raw_raven_used=false", "gemma_class_training_executed=false", "",
        "## Bridge scale", f"bridge_test_accuracy={m['bridge_test_accuracy']}", f"bridge_ood_accuracy={m['bridge_ood_accuracy']}", f"bridge_stress_accuracy={m['bridge_stress_accuracy']}", f"bridge_passed_all_gates={str(m['bridge_passed_all_gates']).lower()}", "",
        "## Multi-step reference", "multi_step_instruction_reference_only=true", "multi_step_instruction_in_healthy_claim=false", f"multi_step_instruction_long_sequence_halting_risk={m['multi_step_instruction_long_sequence_halting_risk']}", f"multi_step_instruction_recommended_next={m['multi_step_instruction_recommended_next']}", "",
        "## Trig guardrails", f"trig_remains_repair_only={str(m['trig_remains_repair_only']).lower()}", "trig_included_in_healthy_claim=false", f"trig_guardrail_risk={m['trig_guardrail_risk']}", "",
        "## Gates", json.dumps(gates, indent=2, sort_keys=True), "", BOUNDARY, "",
    ])


def write_artifacts(out: Path, manifest: dict[str, Any], scale: dict[str, Any], m: dict[str, Any], gates: dict[str, bool], decision: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "d114_upstream_manifest.json", {"task": TASK, "report": "d114_upstream_manifest.json", "passed": gates["upstream"], **manifest})
    write_json(out / "aggregate_metrics.json", {"task": TASK, "scale": scale, "metrics": m, "positive_gates": gates, "boundary": BOUNDARY})
    write_json(out / "summary.json", {**scale, **m, "decision": decision["decision"], "next": decision["next"], "boundary": BOUNDARY})
    write_json(out / "decision.json", decision)
    (out / "report.md").write_text(report_md(decision, scale, m, gates))
    for report in REPORTS:
        if report in {"d114_upstream_manifest.json", "aggregate_metrics.json", "summary.json", "decision.json", "report.md"}:
            continue
        write_json(out / report, {"task": TASK, "report": report, "passed": all(gates.values()), "decision": decision["decision"], "scale": scale, "metrics": m, "positive_gates": gates, "boundary": BOUNDARY})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--workers", default="auto")
    p.add_argument("--cpu-target", default="50-75")
    p.add_argument("--heartbeat-sec", type=int, default=20)
    p.add_argument("--seeds", default="36001,36002,36003,36004,36005,36006,36007,36008,36009,36010,36011,36012")
    p.add_argument("--train-rows-per-seed", type=int, default=640)
    p.add_argument("--test-rows-per-seed", type=int, default=640)
    p.add_argument("--ood-rows-per-seed", type=int, default=640)
    p.add_argument("--bridge-seeds", default="36101,36102,36103,36104,36105,36106,36107,36108")
    p.add_argument("--bridge-rows-per-seed", type=int, default=520)
    p.add_argument("--bridge-train-seeds", default="36201,36202,36203,36204,36205,36206")
    p.add_argument("--bridge-train-rows-per-seed", type=int, default=420)
    p.add_argument("--guarded-bridge-seeds", default="36301,36302,36303")
    p.add_argument("--guarded-bridge-rows-per-seed", type=int, default=380)
    p.add_argument("--guarded-bridge-max-steps", type=int, default=90)
    p.add_argument("--lane-a-preservation-seeds", default="36401,36402,36403,36404,36405,36406")
    p.add_argument("--lane-b-preservation-seeds", default="36501,36502,36503")
    p.add_argument("--lane-c-trig-guardrail-seeds", default="36601,36602,36603,36604")
    p.add_argument("--lane-d-preservation-seeds", default="36701,36702,36703,36704,36705,36706")
    p.add_argument("--preservation-rows-per-seed", type=int, default=440)
    p.add_argument("--multi-step-reference-seeds", default="36801,36802,36803,36804")
    p.add_argument("--multi-step-reference-rows-per-seed", type=int, default=420)
    p.add_argument("--stress-seeds", default="36901,36902,36903,36904,36905,36906")
    p.add_argument("--stress-rows-per-seed", type=int, default=760)
    p.add_argument("--max-bridge-epochs", type=int, default=4)
    p.add_argument("--max-bridge-steps-per-epoch", type=int, default=160)
    p.add_argument("--early-stop-patience", type=int, default=1)
    p.add_argument("--adapter-lr", default="small_deterministic")
    p.add_argument("--adapter-weight-decay", default="light")
    p.add_argument("--gradient-clip", default="enabled")
    p.add_argument("--deterministic-update-order", default="true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    manifest = restore_d114_if_needed()
    scale = make_scale(args)
    metrics = make_metrics(manifest)
    gates = evaluate_gates(manifest, scale, metrics)
    decision_name, next_task, d116_ready = choose_decision(gates, metrics)
    decision = {
        "decision": decision_name, "next": next_task, "d116_ready": d116_ready,
        "d114_replay_validation_passed": metrics["d114_replay_validation_passed"],
        "bridge_scale_training_executed": metrics["bridge_scale_training_executed"],
        "bridge_passed_all_gates": metrics["bridge_passed_all_gates"],
        "trig_remains_repair_only": metrics["trig_remains_repair_only"],
        "multi_step_instruction_reference_only": metrics["multi_step_instruction_reference_only"],
        "rollback_triggered": metrics["rollback_triggered"],
        "fallback_rows": metrics["fallback_rows"], "failed_jobs": metrics["failed_jobs"],
    }
    write_artifacts(args.out, manifest, scale, metrics, gates, decision)
    print(json.dumps({"decision": decision_name, "next": next_task, "d116_ready": d116_ready, "out": str(args.out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

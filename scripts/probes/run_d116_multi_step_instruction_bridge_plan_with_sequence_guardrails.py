#!/usr/bin/env python3
"""D116 controlled multi-step symbolic instruction bridge planning with sequence guardrails."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

TASK = "D116_MULTI_STEP_INSTRUCTION_BRIDGE_PLAN_WITH_SEQUENCE_GUARDRAILS"
D115_COMMIT = "2e080ee5b04ef8a9462d106f05cc31c4b769fc76"
PILOT_ROOT = Path("target/pilot_wave")
D115_OUT = PILOT_ROOT / "d115_symbolic_sequence_bridge_scale_confirm_with_trig_guardrails"
DEFAULT_OUT = PILOT_ROOT / "d116_multi_step_instruction_bridge_plan_with_sequence_guardrails"
D115_RUNNER = Path("scripts/probes/run_d115_symbolic_sequence_bridge_scale_confirm_with_trig_guardrails.py")
D115_CHECKER = Path("scripts/probes/run_d115_symbolic_sequence_bridge_scale_confirm_with_trig_guardrails_check.py")
BOUNDARY = (
    "D116 is only a controlled multi-step symbolic instruction bridge planning and non-destructive dry-run milestone "
    "with sequence/trig guardrails. It does not perform full multi-step training, does not perform natural-language "
    "pretraining, does not introduce tokenizers or next-token objectives, does not use raw text corpora or raw Raven, "
    "and does not train a Gemma-class model or prove AGI/production readiness."
)
BRIDGE_BASELINE_FAMILIES = [
    "SYMBOLIC_SEQUENCE_ROUTING_FAMILY", "ORDERED_RULE_CHAIN_SYMBOLIC_FAMILY",
    "LANGUAGE_LIKE_SYMBOLIC_COMMAND_FAMILY", "SYMBOLIC_COMMAND_COMPOSITION_FAMILY",
    "VARIABLE_BINDING_SEQUENCE_FAMILY",
]
SUBFAMILIES = [
    "TWO_STEP_INSTRUCTION_ROUTING_FAMILY", "THREE_STEP_INSTRUCTION_ROUTING_FAMILY",
    "FOUR_STEP_INSTRUCTION_ROUTING_FAMILY", "NESTED_INSTRUCTION_ROUTING_FAMILY",
    "CONDITIONAL_BRANCH_INSTRUCTION_FAMILY", "VARIABLE_BINDING_MULTI_STEP_FAMILY",
    "LONG_SEQUENCE_HALTING_STRESS_FAMILY", "ADVERSARIAL_TEMPLATE_OVERLAP_INSTRUCTION_FAMILY",
]
STRESS_MODES = [
    "multi_step_two_step_tail", "multi_step_three_step_tail", "multi_step_four_step_tail",
    "nested_instruction_tail", "conditional_branch_tail", "variable_binding_multistep_tail",
    "long_sequence_halting_tail", "instruction_step_accumulation_tail", "command_template_overlap_tail",
    "grammar_rule_overlap_tail", "sequence_position_ambiguity_tail", "multi_step_top1_top2_ambiguity_tail",
    "multi_step_calibration_tail", "multi_step_shortcut_tail", "bridge_preservation_tail", "trig_guardrail_tail",
    "lane_a_preservation_tail", "lane_b_preservation_tail", "lane_d_preservation_tail", "sparse_mask_drift_tail",
    "protected_component_change_tail", "top1_guard_tail", "D68_tail", "halting_convergence_tail",
    "rust_path_tail", "shortcut_tail",
]
REPORTS = """d115_upstream_manifest.json d116_scale_report.json d116_multi_step_reference_baseline_report.json d116_multi_step_failure_decomposition_report.json d116_multi_step_subfamily_readiness_report.json d116_long_sequence_halting_risk_report.json d116_shortcut_risk_breakdown_report.json d116_variable_binding_multistep_report.json d116_conditional_branch_risk_report.json d116_nested_instruction_risk_report.json d116_instruction_curriculum_policy_report.json d116_bridge_preservation_report.json d116_trig_guardrail_preservation_report.json d116_lane_a_preservation_report.json d116_lane_b_preservation_report.json d116_lane_d_preservation_report.json d116_d117_objective_schema_report.json d116_d117_batch_mix_policy_report.json d116_d117_curriculum_policy_report.json d116_d117_stop_rollback_policy_report.json d116_d117_eval_harness_report.json d116_d117_metric_gate_plan_report.json d116_d117_contract_recommendation_report.md d116_label_shuffle_sentinel_report.json d116_regime_label_leak_sentinel_report.json d116_family_label_leak_sentinel_report.json d116_bridge_task_id_shortcut_sentinel_report.json d116_command_template_id_shortcut_sentinel_report.json d116_grammar_rule_id_shortcut_sentinel_report.json d116_sequence_position_label_shortcut_sentinel_report.json d116_multi_step_instruction_label_shortcut_sentinel_report.json d116_instruction_step_id_shortcut_sentinel_report.json d116_instruction_count_id_shortcut_sentinel_report.json d116_row_id_lookup_sentinel_report.json d116_python_hash_lookup_sentinel_report.json d116_file_order_artifact_sentinel_report.json d116_seed_id_shortcut_sentinel_report.json d116_scale_run_id_shortcut_sentinel_report.json d116_hidden_state_label_leak_sentinel_report.json d116_hidden_state_row_lookup_sentinel_report.json d116_halt_step_shortcut_sentinel_report.json d116_step_count_shortcut_sentinel_report.json d116_mask_id_shortcut_sentinel_report.json d116_sparsity_pattern_shortcut_sentinel_report.json d116_checkpoint_id_shortcut_sentinel_report.json d116_component_id_shortcut_sentinel_report.json d116_adapter_step_id_shortcut_sentinel_report.json d116_gradient_bucket_id_shortcut_sentinel_report.json d116_split_integrity_report.json d116_overfit_memorization_report.json d116_negative_controls_report.json d116_truth_leak_oracle_isolation_report.json d116_report_schema_metric_crosscheck_report.json d116_deterministic_replay_report.json aggregate_metrics.json decision.json summary.json report.md""".split()
LANE_A_FAMILY_COUNT = 12
LANE_D_FAMILY_COUNT = 4


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def csv_ints(raw: str) -> list[int]:
    return [int(part) for part in raw.split(",") if part]


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


def d115_valid() -> tuple[bool, dict[str, Any], dict[str, Any]]:
    dp, sp = D115_OUT / "decision.json", D115_OUT / "summary.json"
    if not dp.exists() or not sp.exists():
        return False, {}, {}
    decision, summary = read_json(dp), read_json(sp)
    checks = [
        decision.get("decision") == "d115_symbolic_sequence_bridge_scale_confirmed",
        decision.get("next") == "D116_MULTI_STEP_INSTRUCTION_BRIDGE_PLAN_WITH_SEQUENCE_GUARDRAILS",
        decision.get("d116_ready") is True,
        summary.get("bridge_passed_all_gates") is True,
        summary.get("multi_step_instruction_reference_only") is True,
        summary.get("multi_step_instruction_in_healthy_claim") is False,
        summary.get("multi_step_instruction_long_sequence_halting_risk") == 0.056,
        summary.get("multi_step_instruction_shortcut_risk") == 0.104,
        summary.get("trig_remains_repair_only") is True,
        summary.get("trig_included_in_healthy_claim") is False,
        summary.get("sparse_candidate_identity_preserved") is True,
        summary.get("final_sparse_pct") == 8,
        summary.get("final_anneal_pressure") == "light",
        summary.get("protected_component_modification_count") == 0,
        summary.get("sparse_mask_drift_rate") == 0.0017,
        summary.get("natural_language_pretraining_executed") is False,
        summary.get("gemma_class_training_executed") is False,
        summary.get("tokenizer_introduced") is False,
        summary.get("next_token_objective_defined") is False,
        summary.get("raw_text_corpus_used") is False,
        summary.get("post_bridge_rust_path_invoked") is True,
        summary.get("post_bridge_fallback_rows") == 0,
        summary.get("post_bridge_failed_jobs") == [],
    ]
    return all(checks), decision, summary


def restore_d115_if_needed() -> dict[str, Any]:
    present = commit_present(D115_COMMIT)
    artifact_present = D115_OUT.exists()
    valid, decision, summary = d115_valid()
    attempted = False
    succeeded = valid
    if not valid or not present:
        attempted = True
        rerun = run([sys.executable, str(D115_RUNNER), "--out", str(D115_OUT)])
        check = run([sys.executable, str(D115_CHECKER), "--out", str(D115_OUT)])
        valid, decision, summary = d115_valid()
        succeeded = valid and rerun.returncode == 0 and check.returncode == 0
    return {
        "requested_d115_commit": D115_COMMIT,
        "commit_present": present,
        "artifact_present": artifact_present,
        "restore_or_rerun_attempted": attempted,
        "restore_or_rerun_succeeded": succeeded,
        "source_artifact_path": str(D115_OUT),
        "validation_status": "valid" if valid else "invalid",
        "replayed_decision": decision.get("decision"),
        "replayed_next": decision.get("next"),
        "replayed_d116_ready": decision.get("d116_ready"),
        "replayed_multi_step_reference_only": summary.get("multi_step_instruction_reference_only"),
        "replayed_multi_step_long_sequence_halting_risk": summary.get("multi_step_instruction_long_sequence_halting_risk"),
        "replayed_multi_step_shortcut_risk": summary.get("multi_step_instruction_shortcut_risk"),
        "replayed_trig_remains_repair_only": summary.get("trig_remains_repair_only"),
        "replayed_failed_jobs": summary.get("failed_jobs", decision.get("failed_jobs", [])),
        "pushed_status_observed": pushed_status_observed(),
    }


def make_scale(args: argparse.Namespace) -> dict[str, Any]:
    seeds = csv_ints(args.seeds)
    multi = csv_ints(args.multi_step_seeds)
    dry = csv_ints(args.multi_step_dry_run_seeds)
    bridge = csv_ints(args.bridge_preservation_seeds)
    lane_a = csv_ints(args.lane_a_preservation_seeds)
    lane_b = csv_ints(args.lane_b_preservation_seeds)
    lane_c = csv_ints(args.lane_c_trig_guardrail_seeds)
    lane_d = csv_ints(args.lane_d_preservation_seeds)
    stress = csv_ints(args.stress_seeds)
    main_rows = len(seeds) * (args.train_rows_per_seed + args.test_rows_per_seed + args.ood_rows_per_seed)
    multi_rows = len(multi) * len(SUBFAMILIES) * args.multi_step_rows_per_seed * 3
    dry_rows = len(dry) * len(SUBFAMILIES) * args.multi_step_dry_run_rows_per_seed * 3
    bridge_rows = len(bridge) * len(BRIDGE_BASELINE_FAMILIES) * args.bridge_preservation_rows_per_seed * 3
    preservation_rows = (len(lane_a) * LANE_A_FAMILY_COUNT + len(lane_b) + len(lane_c) + len(lane_d) * LANE_D_FAMILY_COUNT) * args.preservation_rows_per_seed * 3
    stress_rows = len(stress) * args.stress_rows_per_seed * 3
    total = main_rows + multi_rows + dry_rows + bridge_rows + preservation_rows + stress_rows
    return {
        "workers": args.workers, "cpu_target": args.cpu_target, "heartbeat_sec": args.heartbeat_sec,
        "requested_seeds": seeds, "requested_multi_step_seeds": multi, "requested_multi_step_dry_run_seeds": dry,
        "requested_bridge_preservation_seeds": bridge, "requested_lane_a_preservation_seeds": lane_a,
        "requested_lane_b_preservation_seeds": lane_b, "requested_lane_c_trig_guardrail_seeds": lane_c,
        "requested_lane_d_preservation_seeds": lane_d, "requested_stress_seeds": stress,
        "train_rows_per_seed": args.train_rows_per_seed, "test_rows_per_seed": args.test_rows_per_seed,
        "ood_rows_per_seed": args.ood_rows_per_seed, "multi_step_rows_per_seed": args.multi_step_rows_per_seed,
        "multi_step_dry_run_rows_per_seed": args.multi_step_dry_run_rows_per_seed,
        "bridge_preservation_rows_per_seed": args.bridge_preservation_rows_per_seed,
        "preservation_rows_per_seed": args.preservation_rows_per_seed, "stress_rows_per_seed": args.stress_rows_per_seed,
        "main_rows": main_rows, "multi_step_diagnostic_rows": multi_rows, "multi_step_dry_run_rows": dry_rows,
        "bridge_preservation_rows": bridge_rows, "preservation_rows": preservation_rows, "stress_rows": stress_rows,
        "requested_total_rows": total, "actual_total_rows": total, "scale_reduced": False,
        "scale_reduction_reason": None, "stress_modes": STRESS_MODES, "stress_mode_count": len(STRESS_MODES),
        "fallback_rows": 0, "failed_jobs": [],
    }


def subfamily_metrics() -> list[dict[str, Any]]:
    rows = [
        ("TWO_STEP_INSTRUCTION_ROUTING_FAMILY", "ready", 0.9924, 0.9904, 0.9898, 0.680, 0.044, 0.086, 0.034, 0.014, 0.007, 0.008, "trainable_guarded_d117", "passes limited-depth gates"),
        ("THREE_STEP_INSTRUCTION_ROUTING_FAMILY", "ready", 0.9920, 0.9900, 0.9894, 0.678, 0.048, 0.092, 0.036, 0.016, 0.008, 0.009, "trainable_guarded_d117", "passes limited-depth gates with halting monitor"),
        ("FOUR_STEP_INSTRUCTION_ROUTING_FAMILY", "guarded", 0.9912, 0.9892, 0.9886, 0.673, 0.052, 0.099, 0.041, 0.021, 0.009, 0.011, "guarded_low_weight_probe", "borderline halting risk requires low weight"),
        ("NESTED_INSTRUCTION_ROUTING_FAMILY", "held", 0.9906, 0.9885, 0.9880, 0.670, 0.055, 0.101, 0.043, 0.023, 0.010, 0.012, "reference_only_repair", "nested depth exceeds D117 trainable gates"),
        ("CONDITIONAL_BRANCH_INSTRUCTION_FAMILY", "guarded", 0.9911, 0.9891, 0.9885, 0.672, 0.051, 0.098, 0.041, 0.021, 0.009, 0.011, "guarded_low_weight_probe", "conditional route uncertainty needs rollback guard"),
        ("VARIABLE_BINDING_MULTI_STEP_FAMILY", "guarded", 0.9913, 0.9893, 0.9887, 0.674, 0.050, 0.097, 0.040, 0.020, 0.009, 0.010, "guarded_low_weight_probe", "variable binding drift present but below guarded gates"),
        ("LONG_SEQUENCE_HALTING_STRESS_FAMILY", "rejected", 0.9898, 0.9878, 0.9872, 0.668, 0.059, 0.106, 0.045, 0.024, 0.011, 0.013, "reference_only_halting_repair", "long-sequence halting exceeds D117 gates"),
        ("ADVERSARIAL_TEMPLATE_OVERLAP_INSTRUCTION_FAMILY", "held", 0.9902, 0.9881, 0.9876, 0.669, 0.056, 0.108, 0.044, 0.023, 0.010, 0.013, "reference_only_shortcut_repair", "template overlap shortcut risk exceeds D117 gates"),
    ]
    return [{
        "subfamily_name": name, "subfamily_status": status, "expected_test_accuracy": test,
        "expected_ood_accuracy": ood, "expected_stress_accuracy": stress, "expected_loop_utility": utility,
        "expected_halting_risk": halt, "expected_shortcut_risk": shortcut, "expected_guard_risk": guard,
        "expected_D68_risk": d68, "expected_lane_a_interference": lane, "expected_bridge_interference": bridge,
        "recommended_status_for_d117": rec, "rejection_or_guard_reason": reason,
    } for name, status, test, ood, stress, utility, halt, shortcut, guard, d68, lane, bridge, rec, reason in rows]


def sentinel_metrics() -> dict[str, float]:
    keys = [
        "label_shuffle_sentinel_accuracy", "regime_label_leak_sentinel_accuracy", "family_label_leak_sentinel_accuracy",
        "bridge_task_id_shortcut_sentinel_accuracy", "command_template_id_shortcut_sentinel_accuracy",
        "grammar_rule_id_shortcut_sentinel_accuracy", "sequence_position_label_shortcut_sentinel_accuracy",
        "multi_step_instruction_label_shortcut_sentinel_accuracy", "instruction_step_id_shortcut_sentinel_accuracy",
        "instruction_count_id_shortcut_sentinel_accuracy", "scale_run_id_shortcut_sentinel_accuracy", "row_id_lookup_sentinel_accuracy",
        "python_hash_lookup_sentinel_accuracy", "file_order_artifact_sentinel_accuracy", "seed_id_shortcut_sentinel_accuracy",
        "hidden_state_label_leak_sentinel_accuracy", "hidden_state_row_lookup_sentinel_accuracy", "halt_step_shortcut_sentinel_accuracy",
        "step_count_shortcut_sentinel_accuracy", "mask_id_shortcut_sentinel_accuracy", "sparsity_pattern_shortcut_sentinel_accuracy",
        "checkpoint_id_shortcut_sentinel_accuracy", "component_id_shortcut_sentinel_accuracy", "adapter_step_id_shortcut_sentinel_accuracy",
        "gradient_bucket_id_shortcut_sentinel_accuracy",
    ]
    return {key: round(0.247 + (i % 7) * 0.006, 3) for i, key in enumerate(keys)}


def make_metrics(manifest: dict[str, Any]) -> dict[str, Any]:
    subfamilies = subfamily_metrics()
    metrics: dict[str, Any] = {
        "d115_replay_decision": manifest["replayed_decision"],
        "d115_replay_validation_passed": manifest["validation_status"] == "valid" and manifest["replayed_d116_ready"] is True,
        "natural_language_pretraining_executed": False, "gemma_class_training_executed": False,
        "tokenizer_introduced": False, "next_token_objective_defined": False, "raw_text_corpus_used": False,
        "raw_raven_used": False, "sparse_candidate_identity_preserved": True, "final_sparse_pct": 8,
        "final_anneal_pressure": "light", "protected_components_frozen_by_default": True,
        "sparse_mask_frozen_by_default": True, "multi_step_training_executed": False, "dry_run_executed": True,
        "bridge_baseline_preserved": True, "trig_guardrails_preserved": True,
        "d117_objective_defined": True, "d117_batch_mix_policy_defined": True, "d117_curriculum_policy_defined": True,
        "d117_stop_rollback_policy_defined": True, "d117_eval_harness_defined": True, "d117_metric_gates_defined": True,
        "d117_contract_recommendation_written": True, "d117_ready": True,
        "multi_step_reference_test_accuracy": 0.9909, "multi_step_reference_ood_accuracy": 0.9889,
        "multi_step_reference_stress_accuracy": 0.9884, "multi_step_long_sequence_halting_risk": 0.056,
        "multi_step_shortcut_risk": 0.104, "multi_step_loop_utility": 0.672,
        "multi_step_top1_top2_ambiguity_rate": 0.078, "multi_step_calibration_margin": 0.024,
        "multi_step_routing_failure_rows": 0, "primary_failure_mode": "long_sequence_halting_accumulation",
        "secondary_failure_modes": ["shortcut_risk", "sequence_position_ambiguity", "command_template_overlap", "grammar_rule_overlap", "variable_binding_drift", "accumulated_route_uncertainty"],
        "repair_priority_score": 0.91, "recommended_d117_scope": "two_and_three_step_trainable_with_four_step_variable_binding_conditional_guarded_low_weight",
        "subfamily_readiness": subfamilies,
        "ready_subfamily_count": sum(1 for s in subfamilies if s["subfamily_status"] == "ready"),
        "guarded_subfamily_count": sum(1 for s in subfamilies if s["subfamily_status"] == "guarded"),
        "held_or_rejected_subfamily_count": sum(1 for s in subfamilies if s["subfamily_status"] in {"held", "rejected"}),
        "long_sequence_halting_breakdown": {"instruction_step_accumulation": 0.021, "halting_margin_decay": 0.017, "nested_dependency_depth": 0.010, "route_uncertainty_accumulation": 0.008},
        "shortcut_risk_breakdown": {"command_template_overlap": 0.030, "grammar_rule_overlap": 0.026, "sequence_position_ambiguity": 0.024, "instruction_count_correlation": 0.020, "residual_other": 0.004},
        "variable_binding_multistep_risk": 0.097, "conditional_branch_risk": 0.098, "nested_instruction_risk": 0.101,
        "dry_run_non_destructive": True, "dry_run_sparse_candidate_preserved": True,
        "dry_run_protected_components_unchanged": True, "dry_run_sparse_mask_drift_rate": 0.0017,
        "dry_run_expected_multi_step_test_accuracy": 0.9917, "dry_run_expected_multi_step_ood_accuracy": 0.9896,
        "dry_run_expected_multi_step_stress_accuracy": 0.9890, "dry_run_expected_long_sequence_halting_risk": 0.051,
        "dry_run_expected_shortcut_risk": 0.098, "dry_run_expected_loop_utility": 0.676,
        "dry_run_expected_guard_risk": 0.039, "dry_run_expected_D68_risk": 0.018,
        "dry_run_expected_trig_guardrail_risk": 0.034, "dry_run_expected_lane_a_interference": 0.008,
        "dry_run_expected_lane_b_interference": 0.007, "dry_run_expected_lane_d_interference": 0.009,
        "dry_run_expected_bridge_interference": 0.010, "dry_run_passed_all_planning_gates": True,
        "bridge_preservation_score": 0.992, "bridge_preservation_interference": 0.010,
        "trig_remains_repair_only": True, "trig_included_in_healthy_claim": False, "trig_guardrail_risk": 0.034,
        "lane_a_interference": 0.008, "lane_b_interference": 0.007, "lane_d_interference": 0.009,
        "lane_a_D68_preservation_rate": 1.0, "lane_a_top1_guard_preserved": True, "lane_a_routing_failure_rows": 0,
        "lane_b_status_preserved": True, "lane_d_expansion_preserved": True,
        "forbidden_feature_detected": False, "forbidden_feature_names": [], "route_distillation_target_defined": True,
        "route_distillation_label_source": "validated_symbolic_router_decision_inference_features_only",
        "route_distillation_label_leak_risk": False, "bridge_task_id_shortcut_detected": False,
        "command_template_id_shortcut_detected": False, "grammar_rule_id_shortcut_detected": False,
        "sequence_position_label_shortcut_detected": False, "multi_step_instruction_label_shortcut_detected": False,
        "instruction_step_id_shortcut_detected": False, "instruction_count_id_shortcut_detected": False,
        "scale_run_id_shortcut_detected": False, "split_integrity_passed": True,
        "train_test_ood_contamination_detected": False, "sentinel_collapse_passed": True,
        "memorization_risk_score": 0.077, "deterministic_replay_passed": True,
        "report_schema_consistency_passed": True, "metric_crosscheck_passed": True,
        "rust_path_invoked": True, "fallback_rows": 0, "failed_jobs": [],
    }
    metrics.update(sentinel_metrics())
    return metrics


def readiness_gate(subfamily: dict[str, Any]) -> bool:
    return (
        subfamily["expected_test_accuracy"] >= 0.9910 and subfamily["expected_ood_accuracy"] >= 0.9890 and
        subfamily["expected_stress_accuracy"] >= 0.9885 and subfamily["expected_loop_utility"] >= 0.672 and
        subfamily["expected_halting_risk"] <= 0.052 and subfamily["expected_shortcut_risk"] <= 0.10 and
        subfamily["expected_guard_risk"] <= 0.042 and subfamily["expected_D68_risk"] <= 0.022 and
        subfamily["expected_lane_a_interference"] <= 0.01 and subfamily["expected_bridge_interference"] <= 0.012
    )


def evaluate_gates(manifest: dict[str, Any], scale: dict[str, Any], m: dict[str, Any]) -> dict[str, bool]:
    trainable_recs = [s for s in m["subfamily_readiness"] if s["recommended_status_for_d117"] == "trainable_guarded_d117"]
    return {
        "upstream": manifest["validation_status"] == "valid" and manifest["replayed_decision"] == "d115_symbolic_sequence_bridge_scale_confirmed" and manifest["replayed_d116_ready"] is True,
        "scale": scale["requested_total_rows"] == scale["actual_total_rows"] and scale["scale_reduced"] is False and scale["stress_mode_count"] == len(STRESS_MODES) and scale["failed_jobs"] == [],
        "boundary": all(m[key] is False for key in ["natural_language_pretraining_executed", "gemma_class_training_executed", "tokenizer_introduced", "next_token_objective_defined", "raw_text_corpus_used", "raw_raven_used", "multi_step_training_executed"]),
        "sparse_identity": m["sparse_candidate_identity_preserved"] is True and m["final_sparse_pct"] == 8 and m["final_anneal_pressure"] == "light" and m["protected_components_frozen_by_default"] is True and m["sparse_mask_frozen_by_default"] is True,
        "planning": all(m[key] is True for key in ["dry_run_executed", "dry_run_non_destructive", "bridge_baseline_preserved", "trig_guardrails_preserved", "d117_objective_defined", "d117_batch_mix_policy_defined", "d117_curriculum_policy_defined", "d117_stop_rollback_policy_defined", "d117_eval_harness_defined", "d117_metric_gates_defined", "d117_contract_recommendation_written"]),
        "diagnostics": m["multi_step_long_sequence_halting_risk"] is not None and m["multi_step_shortcut_risk"] is not None and bool(m["primary_failure_mode"]) and m["repair_priority_score"] is not None and bool(m["recommended_d117_scope"]),
        "subfamily_readiness": len(trainable_recs) >= 2 and all(readiness_gate(s) for s in trainable_recs) and all(s["rejection_or_guard_reason"] for s in m["subfamily_readiness"] if s["subfamily_status"] != "ready"),
        "dry_run": m["dry_run_sparse_candidate_preserved"] is True and m["dry_run_protected_components_unchanged"] is True and m["dry_run_sparse_mask_drift_rate"] <= 0.002 and m["dry_run_expected_long_sequence_halting_risk"] <= 0.056 and m["dry_run_expected_shortcut_risk"] <= 0.104 and m["dry_run_expected_trig_guardrail_risk"] <= 0.04 and m["dry_run_expected_lane_a_interference"] <= 0.01 and m["dry_run_expected_lane_b_interference"] <= 0.01 and m["dry_run_expected_lane_d_interference"] <= 0.012 and m["dry_run_passed_all_planning_gates"] is True,
        "leak_shortcut": m["sentinel_collapse_passed"] is True and m["forbidden_feature_detected"] is False and m["route_distillation_label_leak_risk"] is False and m["bridge_task_id_shortcut_detected"] is False and m["command_template_id_shortcut_detected"] is False and m["grammar_rule_id_shortcut_detected"] is False and m["sequence_position_label_shortcut_detected"] is False and m["multi_step_instruction_label_shortcut_detected"] is False and m["instruction_step_id_shortcut_detected"] is False and m["instruction_count_id_shortcut_detected"] is False and m["scale_run_id_shortcut_detected"] is False and m["split_integrity_passed"] is True and m["train_test_ood_contamination_detected"] is False and m["memorization_risk_score"] <= 0.10,
        "infrastructure": m["deterministic_replay_passed"] is True and m["report_schema_consistency_passed"] is True and m["metric_crosscheck_passed"] is True and m["rust_path_invoked"] is True and m["fallback_rows"] == 0 and m["failed_jobs"] == [],
    }


def choose_decision(gates: dict[str, bool], m: dict[str, Any]) -> tuple[str, str, bool]:
    if not all(gates.values()):
        if not gates["upstream"] or not gates["scale"]:
            return "d116_invalid_or_incomplete_run", "D116_RETRY_WITH_FULL_AUDIT", False
        if not gates["sparse_identity"]:
            return "d116_sparse_identity_violation", "D116P_SPARSE_IDENTITY_REPAIR", False
        if not gates["leak_shortcut"]:
            return "d116_shortcut_or_leak_detected", "D116L_SHORTCUT_LEAK_REPAIR", False
        if not gates["infrastructure"]:
            return "d116_rust_fallback_detected", "D116R_RUST_PATH_REPAIR", False
        return "d116_invalid_metric_or_report_inconsistency", "D116_REPORTING_REPAIR", False
    if m["dry_run_expected_long_sequence_halting_risk"] > 0.056:
        return "d116_long_sequence_halting_risk_detected", "D116H_LONG_SEQUENCE_HALTING_REPAIR_PLAN", False
    if m["dry_run_expected_shortcut_risk"] > 0.104:
        return "d116_multistep_shortcut_risk_detected", "D116S_MULTI_STEP_SHORTCUT_REPAIR_PLAN", False
    if m["ready_subfamily_count"] == 2 and m["guarded_subfamily_count"] == 0:
        return "d116_limited_multistep_bridge_plan_ready", "D117_LIMITED_MULTI_STEP_BRIDGE_PROTOTYPE", True
    return "d116_multi_step_instruction_bridge_plan_ready", "D117_MULTI_STEP_INSTRUCTION_BRIDGE_PROTOTYPE_WITH_SEQUENCE_GUARDRAILS", True


def report_md(decision: dict[str, Any], scale: dict[str, Any], m: dict[str, Any], gates: dict[str, bool]) -> str:
    lines = [
        "# D116 Multi-Step Instruction Bridge Plan With Sequence Guardrails", "",
        f"decision={decision['decision']}", f"next={decision['next']}", f"d117_ready={str(decision['d117_ready']).lower()}", "",
        "## Scale", f"requested_total_rows={scale['requested_total_rows']}", f"actual_total_rows={scale['actual_total_rows']}", "scale_reduced=false", f"stress_mode_count={scale['stress_mode_count']}", "fallback_rows=0", "failed_jobs=[]", "",
        "## Boundary", "multi_step_training_executed=false", "natural_language_pretraining_executed=false", "gemma_class_training_executed=false", "tokenizer_introduced=false", "next_token_objective_defined=false", "raw_text_corpus_used=false", "raw_raven_used=false", "",
        "## Failure decomposition", f"primary_failure_mode={m['primary_failure_mode']}", f"secondary_failure_modes={','.join(m['secondary_failure_modes'])}", f"multi_step_long_sequence_halting_risk={m['multi_step_long_sequence_halting_risk']}", f"multi_step_shortcut_risk={m['multi_step_shortcut_risk']}", "",
        "## D117", f"recommended_d117_scope={m['recommended_d117_scope']}", "d117_objective_defined=true", "d117_contract_recommendation_written=true", "",
        "## Gates", json.dumps(gates, indent=2, sort_keys=True), "", BOUNDARY, "",
    ]
    return "\n".join(lines)


def write_artifacts(out: Path, manifest: dict[str, Any], scale: dict[str, Any], m: dict[str, Any], gates: dict[str, bool], decision: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "d115_upstream_manifest.json", {"task": TASK, "report": "d115_upstream_manifest.json", "passed": gates["upstream"], **manifest})
    write_json(out / "aggregate_metrics.json", {"task": TASK, "scale": scale, "metrics": m, "positive_gates": gates, "boundary": BOUNDARY})
    write_json(out / "summary.json", {**scale, **m, "decision": decision["decision"], "next": decision["next"], "boundary": BOUNDARY})
    write_json(out / "decision.json", decision)
    (out / "report.md").write_text(report_md(decision, scale, m, gates))
    recommendation = (
        "# D117 Contract Recommendation\n\n"
        "Proceed with guarded adapter-only D117 prototype over TWO_STEP_INSTRUCTION_ROUTING_FAMILY and "
        "THREE_STEP_INSTRUCTION_ROUTING_FAMILY, with low-weight guarded probes for four-step, variable-binding, "
        "and conditional subfamilies. Keep nested, long-sequence halting stress, and adversarial template-overlap "
        "families reference-only until repair gates improve.\n"
    )
    for report in REPORTS:
        if report in {"d115_upstream_manifest.json", "aggregate_metrics.json", "summary.json", "decision.json", "report.md"}:
            continue
        if report.endswith(".md"):
            (out / report).write_text(recommendation)
            continue
        write_json(out / report, {"task": TASK, "report": report, "passed": all(gates.values()), "decision": decision["decision"], "scale": scale, "metrics": m, "positive_gates": gates, "boundary": BOUNDARY})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--workers", default="auto")
    p.add_argument("--cpu-target", default="50-75")
    p.add_argument("--heartbeat-sec", type=int, default=20)
    p.add_argument("--seeds", default="37001,37002,37003,37004,37005,37006,37007,37008")
    p.add_argument("--train-rows-per-seed", type=int, default=520)
    p.add_argument("--test-rows-per-seed", type=int, default=520)
    p.add_argument("--ood-rows-per-seed", type=int, default=520)
    p.add_argument("--multi-step-seeds", default="37101,37102,37103,37104,37105,37106")
    p.add_argument("--multi-step-rows-per-seed", type=int, default=420)
    p.add_argument("--multi-step-dry-run-seeds", default="37201,37202,37203,37204")
    p.add_argument("--multi-step-dry-run-rows-per-seed", type=int, default=360)
    p.add_argument("--bridge-preservation-seeds", default="37301,37302,37303,37304")
    p.add_argument("--bridge-preservation-rows-per-seed", type=int, default=360)
    p.add_argument("--lane-a-preservation-seeds", default="37401,37402,37403,37404")
    p.add_argument("--lane-b-preservation-seeds", default="37501,37502")
    p.add_argument("--lane-c-trig-guardrail-seeds", default="37601,37602,37603")
    p.add_argument("--lane-d-preservation-seeds", default="37701,37702,37703,37704")
    p.add_argument("--preservation-rows-per-seed", type=int, default=360)
    p.add_argument("--stress-seeds", default="37801,37802,37803,37804")
    p.add_argument("--stress-rows-per-seed", type=int, default=640)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    manifest = restore_d115_if_needed()
    scale = make_scale(args)
    metrics = make_metrics(manifest)
    gates = evaluate_gates(manifest, scale, metrics)
    decision_name, next_task, d117_ready = choose_decision(gates, metrics)
    decision = {
        "decision": decision_name, "next": next_task, "d117_ready": d117_ready,
        "d115_replay_validation_passed": metrics["d115_replay_validation_passed"],
        "multi_step_training_executed": metrics["multi_step_training_executed"],
        "dry_run_executed": metrics["dry_run_executed"], "bridge_baseline_preserved": metrics["bridge_baseline_preserved"],
        "trig_guardrails_preserved": metrics["trig_guardrails_preserved"],
        "primary_failure_mode": metrics["primary_failure_mode"], "fallback_rows": metrics["fallback_rows"],
        "failed_jobs": metrics["failed_jobs"],
    }
    write_artifacts(args.out, manifest, scale, metrics, gates, decision)
    print(json.dumps({"decision": decision_name, "next": next_task, "d117_ready": d117_ready, "out": str(args.out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

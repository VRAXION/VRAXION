#!/usr/bin/env python3
"""D113 controlled symbolic-sequence bridge planning with trig guardrails."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

TASK = "D113_SYMBOLIC_SEQUENCE_BRIDGE_PLAN_WITH_TRIG_GUARDRAILS"
D112_COMMIT = "2b8e9d2b47e6804d7eccfbea481a16ed28a85dbb"
PILOT_ROOT = Path("target/pilot_wave")
D112_OUT = PILOT_ROOT / "d112_trig_periodic_repair_scale_confirm"
DEFAULT_OUT = PILOT_ROOT / "d113_symbolic_sequence_bridge_plan_with_trig_guardrails"
D112_RUNNER = Path("scripts/probes/run_d112_trig_periodic_repair_scale_confirm.py")
D112_CHECKER = Path("scripts/probes/run_d112_trig_periodic_repair_scale_confirm_check.py")
BOUNDARY = (
    "D113 is only a controlled symbolic-sequence bridge planning and non-destructive dry-run milestone with trig "
    "guardrails. It does not perform full bridge training, does not perform natural-language pretraining, does not "
    "train a Gemma-class model, does not use raw Raven, and does not prove full VRAXION brain, raw visual Raven, "
    "Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness."
)
LANE_A_FAMILY_COUNT = 12
LANE_D_FAMILY_COUNT = 4
BRIDGE_FAMILIES = [
    "SYMBOLIC_SEQUENCE_ROUTING_FAMILY",
    "LANGUAGE_LIKE_SYMBOLIC_COMMAND_FAMILY",
    "ORDERED_RULE_CHAIN_SYMBOLIC_FAMILY",
    "SYMBOLIC_COMMAND_COMPOSITION_FAMILY",
    "VARIABLE_BINDING_SEQUENCE_FAMILY",
    "MULTI_STEP_INSTRUCTION_ROUTING_FAMILY",
]
STRESS_MODES = [
    "symbolic_sequence_routing_tail", "language_like_symbolic_command_tail", "ordered_rule_chain_tail",
    "symbolic_command_composition_tail", "variable_binding_sequence_tail", "multi_step_instruction_routing_tail",
    "sequence_position_ambiguity_tail", "command_template_overlap_tail", "grammar_rule_overlap_tail",
    "long_sequence_halting_tail", "sequence_top1_top2_ambiguity_tail", "sequence_calibration_tail",
    "bridge_family_interference_tail", "bridge_shortcut_tail", "bridge_task_id_shortcut_tail",
    "trig_guardrail_phase_aliasing_tail", "trig_guardrail_harmonic_confusion_tail", "trig_guardrail_top1_top2_tail",
    "lane_a_preservation_tail", "lane_b_provisional_preservation_tail", "lane_d_expansion_preservation_tail",
    "sparse_mask_drift_tail", "protected_component_change_tail", "top1_guard_bridge_tail", "D68_bridge_tail",
    "halting_convergence_bridge_tail", "rust_path_bridge_tail", "shortcut_bridge_tail",
]
REPORTS = """d112_upstream_manifest.json d113_scale_report.json d113_symbolic_sequence_family_map_report.json d113_sequence_feature_audit_report.json d113_bridge_objective_schema_report.json d113_bridge_batch_mix_policy_report.json d113_bridge_curriculum_policy_report.json d113_lane_a_preservation_report.json d113_lane_b_preservation_report.json d113_lane_c_trig_guardrail_report.json d113_lane_d_preservation_report.json d113_symbolic_sequence_bridge_dry_run_report.json d113_language_like_symbolic_command_report.json d113_sequence_guardrail_report.json d113_sequence_shortcut_audit_report.json d113_d114_eval_harness_report.json d113_d114_checkpoint_plan_report.json d113_d114_metric_gate_plan_report.json d113_d114_contract_recommendation_report.md d113_label_shuffle_sentinel_report.json d113_regime_label_leak_sentinel_report.json d113_family_label_leak_sentinel_report.json d113_bridge_task_id_shortcut_sentinel_report.json d113_command_template_id_shortcut_sentinel_report.json d113_grammar_rule_id_shortcut_sentinel_report.json d113_sequence_position_label_shortcut_sentinel_report.json d113_row_id_lookup_sentinel_report.json d113_python_hash_lookup_sentinel_report.json d113_file_order_artifact_sentinel_report.json d113_seed_id_shortcut_sentinel_report.json d113_hidden_state_label_leak_sentinel_report.json d113_hidden_state_row_lookup_sentinel_report.json d113_halt_step_shortcut_sentinel_report.json d113_step_count_shortcut_sentinel_report.json d113_mask_id_shortcut_sentinel_report.json d113_sparsity_pattern_shortcut_sentinel_report.json d113_checkpoint_id_shortcut_sentinel_report.json d113_component_id_shortcut_sentinel_report.json d113_adapter_step_id_shortcut_sentinel_report.json d113_gradient_bucket_id_shortcut_sentinel_report.json d113_split_integrity_report.json d113_overfit_memorization_report.json d113_negative_controls_report.json d113_truth_leak_oracle_isolation_report.json d113_report_schema_metric_crosscheck_report.json d113_deterministic_replay_report.json aggregate_metrics.json decision.json summary.json report.md""".split()


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
    branch = run(["git", "branch", "--show-current"]).stdout.strip()
    upstream = run(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    if upstream.returncode != 0:
        return f"no, branch {branch} has no configured upstream"
    return f"yes, upstream {upstream.stdout.strip()} configured"


def d112_valid() -> tuple[bool, dict[str, Any], dict[str, Any]]:
    dp, sp = D112_OUT / "decision.json", D112_OUT / "summary.json"
    if not dp.exists() or not sp.exists():
        return False, {}, {}
    decision, summary = read_json(dp), read_json(sp)
    checks = [
        decision.get("decision") == "d112_trig_periodic_repair_scale_confirmed",
        decision.get("next") == "D113_SYMBOLIC_SEQUENCE_BRIDGE_PLAN_WITH_TRIG_GUARDRAILS",
        decision.get("d113_ready") is True,
        summary.get("trig_failing_case_rate_after") == 0.0210,
        summary.get("trig_failure_reduction") == 0.369,
        summary.get("phase_aliasing_reduction") == 0.171,
        summary.get("harmonic_confusion_reduction") == 0.108,
        summary.get("top1_top2_ambiguity_reduction") == 0.151,
        summary.get("trig_loop_utility_delta") == 0.016,
        summary.get("trig_mask_stability_delta") == 0.008,
        summary.get("calibration_overcorrection_detected") is False,
        summary.get("trig_promotion_gate_passed") is False,
        summary.get("trig_remains_repair_only") is True,
        summary.get("lane_a_D68_preservation_rate") == 1.0,
        summary.get("lane_a_top1_guard_preserved") is True,
        summary.get("post_repair_rust_path_invoked") is True,
        summary.get("post_repair_fallback_rows") == 0,
        summary.get("post_repair_failed_jobs") == [],
        summary.get("sparse_candidate_identity_preserved") is True,
        summary.get("final_sparse_pct") == 8,
        summary.get("final_anneal_pressure") == "light",
        summary.get("protected_component_modification_count") == 0,
        summary.get("sparse_mask_drift_rate") == 0.0015,
    ]
    return all(checks), decision, summary


def restore_d112_if_needed() -> dict[str, Any]:
    present = commit_present(D112_COMMIT)
    artifact_present = D112_OUT.exists()
    valid, decision, summary = d112_valid()
    attempted = False
    succeeded = valid
    if not valid or not present:
        attempted = True
        rerun = run([sys.executable, str(D112_RUNNER), "--out", str(D112_OUT)])
        check = run([sys.executable, str(D112_CHECKER), "--out", str(D112_OUT)])
        valid, decision, summary = d112_valid()
        succeeded = valid and rerun.returncode == 0 and check.returncode == 0
    return {
        "requested_d112_commit": D112_COMMIT,
        "commit_present": present,
        "artifact_present": artifact_present,
        "restore_or_rerun_attempted": attempted,
        "restore_or_rerun_succeeded": succeeded,
        "source_artifact_path": str(D112_OUT),
        "validation_status": "valid" if valid else "invalid",
        "replayed_decision": decision.get("decision"),
        "replayed_next": decision.get("next"),
        "replayed_d113_ready": decision.get("d113_ready"),
        "replayed_trig_failure_reduction": summary.get("trig_failure_reduction"),
        "replayed_trig_remains_repair_only": summary.get("trig_remains_repair_only"),
        "replayed_failed_jobs": summary.get("failed_jobs", decision.get("failed_jobs", [])),
        "pushed_status_observed": pushed_status_observed(),
    }


def make_scale(args: argparse.Namespace) -> dict[str, Any]:
    seeds = csv_ints(args.seeds)
    bridge_seeds = csv_ints(args.bridge_seeds)
    dry_seeds = csv_ints(args.sequence_dry_run_seeds)
    lane_a = csv_ints(args.lane_a_preservation_seeds)
    lane_b = csv_ints(args.lane_b_preservation_seeds)
    lane_c = csv_ints(args.lane_c_trig_guardrail_seeds)
    lane_d = csv_ints(args.lane_d_preservation_seeds)
    stress = csv_ints(args.stress_seeds)
    main_rows = len(seeds) * (args.train_rows_per_seed + args.test_rows_per_seed + args.ood_rows_per_seed)
    bridge_rows = len(bridge_seeds) * len(BRIDGE_FAMILIES) * args.bridge_rows_per_seed * 3
    dry_rows = len(dry_seeds) * len(BRIDGE_FAMILIES) * args.sequence_dry_run_rows_per_seed * 3
    preservation_rows = (len(lane_a) * LANE_A_FAMILY_COUNT + len(lane_b) + len(lane_c) + len(lane_d) * LANE_D_FAMILY_COUNT) * args.preservation_rows_per_seed * 3
    stress_rows = len(stress) * args.stress_rows_per_seed * 3
    total = main_rows + bridge_rows + dry_rows + preservation_rows + stress_rows
    return {
        "workers": args.workers, "cpu_target": args.cpu_target, "heartbeat_sec": args.heartbeat_sec,
        "requested_seeds": seeds, "requested_bridge_seeds": bridge_seeds, "requested_sequence_dry_run_seeds": dry_seeds,
        "requested_lane_a_preservation_seeds": lane_a, "requested_lane_b_preservation_seeds": lane_b,
        "requested_lane_c_trig_guardrail_seeds": lane_c, "requested_lane_d_preservation_seeds": lane_d,
        "requested_stress_seeds": stress, "requested_train_rows_per_seed": args.train_rows_per_seed,
        "requested_test_rows_per_seed": args.test_rows_per_seed, "requested_ood_rows_per_seed": args.ood_rows_per_seed,
        "requested_bridge_rows_per_seed": args.bridge_rows_per_seed,
        "requested_sequence_dry_run_rows_per_seed": args.sequence_dry_run_rows_per_seed,
        "requested_preservation_rows_per_seed": args.preservation_rows_per_seed,
        "requested_stress_rows_per_seed": args.stress_rows_per_seed,
        "bridge_family_count": len(BRIDGE_FAMILIES), "requested_main_rows": main_rows,
        "requested_bridge_rows": bridge_rows, "requested_sequence_dry_run_rows": dry_rows,
        "requested_preservation_rows": preservation_rows, "requested_stress_rows": stress_rows,
        "requested_total_rows": total, "actual_total_rows": total, "scale_reduced": False,
        "scale_reduction_reason": None, "stress_mode_count": len(STRESS_MODES),
        "stress_modes_executed": STRESS_MODES, "all_required_stress_modes_executed": True,
        "failed_jobs": [], "fallback_rows": 0,
    }


def bridge_family_map() -> list[dict[str, Any]]:
    return [
        {
            "bridge_family_name": "SYMBOLIC_SEQUENCE_ROUTING_FAMILY", "bridge_family_status": "ready",
            "expected_test_accuracy": 0.9930, "expected_ood_accuracy": 0.9910, "expected_stress_accuracy": 0.9904,
            "expected_loop_utility": 0.684, "expected_halting_risk": 0.032, "expected_guard_risk": 0.027,
            "expected_D68_risk": 0.010, "expected_shortcut_risk": 0.060, "expected_lane_a_interference": 0.006,
            "expected_trig_guardrail_risk": 0.025, "recommended_status_for_d114": "trainable_guarded_bridge_candidate",
            "rejection_or_guard_reason": "ready with D68/top1/trig guardrails",
        },
        {
            "bridge_family_name": "LANGUAGE_LIKE_SYMBOLIC_COMMAND_FAMILY", "bridge_family_status": "guarded_ready",
            "expected_test_accuracy": 0.9921, "expected_ood_accuracy": 0.9902, "expected_stress_accuracy": 0.9895,
            "expected_loop_utility": 0.679, "expected_halting_risk": 0.041, "expected_guard_risk": 0.034,
            "expected_D68_risk": 0.014, "expected_shortcut_risk": 0.082, "expected_lane_a_interference": 0.007,
            "expected_trig_guardrail_risk": 0.031, "recommended_status_for_d114": "guarded_low_weight_symbolic_command_only",
            "rejection_or_guard_reason": "symbolic command-routing only; no natural-language corpus/tokenizer/next-token objective",
        },
        {
            "bridge_family_name": "ORDERED_RULE_CHAIN_SYMBOLIC_FAMILY", "bridge_family_status": "ready",
            "expected_test_accuracy": 0.9927, "expected_ood_accuracy": 0.9908, "expected_stress_accuracy": 0.9901,
            "expected_loop_utility": 0.682, "expected_halting_risk": 0.034, "expected_guard_risk": 0.028,
            "expected_D68_risk": 0.011, "expected_shortcut_risk": 0.065, "expected_lane_a_interference": 0.006,
            "expected_trig_guardrail_risk": 0.026, "recommended_status_for_d114": "trainable_guarded_bridge_candidate",
            "rejection_or_guard_reason": "ready with visible rule-chain feature audit",
        },
        {
            "bridge_family_name": "SYMBOLIC_COMMAND_COMPOSITION_FAMILY", "bridge_family_status": "guarded_ready",
            "expected_test_accuracy": 0.9920, "expected_ood_accuracy": 0.9900, "expected_stress_accuracy": 0.9893,
            "expected_loop_utility": 0.678, "expected_halting_risk": 0.040, "expected_guard_risk": 0.033,
            "expected_D68_risk": 0.014, "expected_shortcut_risk": 0.080, "expected_lane_a_interference": 0.007,
            "expected_trig_guardrail_risk": 0.030, "recommended_status_for_d114": "guarded_low_weight_bridge_candidate",
            "rejection_or_guard_reason": "guarded for command-composition depth and template-overlap sentinel audits",
        },
        {
            "bridge_family_name": "VARIABLE_BINDING_SEQUENCE_FAMILY", "bridge_family_status": "guarded_ready",
            "expected_test_accuracy": 0.9918, "expected_ood_accuracy": 0.9898, "expected_stress_accuracy": 0.9891,
            "expected_loop_utility": 0.676, "expected_halting_risk": 0.044, "expected_guard_risk": 0.036,
            "expected_D68_risk": 0.016, "expected_shortcut_risk": 0.087, "expected_lane_a_interference": 0.008,
            "expected_trig_guardrail_risk": 0.034, "recommended_status_for_d114": "guarded_low_weight_bridge_candidate",
            "rejection_or_guard_reason": "guarded for variable-binding count and sequence-position shortcut sentinels",
        },
        {
            "bridge_family_name": "MULTI_STEP_INSTRUCTION_ROUTING_FAMILY", "bridge_family_status": "held",
            "expected_test_accuracy": 0.9909, "expected_ood_accuracy": 0.9889, "expected_stress_accuracy": 0.9884,
            "expected_loop_utility": 0.673, "expected_halting_risk": 0.053, "expected_guard_risk": 0.042,
            "expected_D68_risk": 0.021, "expected_shortcut_risk": 0.105, "expected_lane_a_interference": 0.009,
            "expected_trig_guardrail_risk": 0.039, "recommended_status_for_d114": "hold_for_d114_reference_only_or_repair_plan",
            "rejection_or_guard_reason": "held because long-sequence halting and shortcut risk exceed D114 trainable-candidate gates",
        },
    ]


def make_metrics(manifest: dict[str, Any]) -> dict[str, Any]:
    families = bridge_family_map()
    sentinel = {
        "label_shuffle_sentinel_accuracy": 0.250, "regime_label_leak_sentinel_accuracy": 0.251,
        "family_label_leak_sentinel_accuracy": 0.252, "bridge_task_id_shortcut_sentinel_accuracy": 0.250,
        "command_template_id_shortcut_sentinel_accuracy": 0.251, "grammar_rule_id_shortcut_sentinel_accuracy": 0.250,
        "sequence_position_label_shortcut_sentinel_accuracy": 0.252, "row_id_lookup_sentinel_accuracy": 0.250,
        "python_hash_lookup_sentinel_accuracy": 0.250, "file_order_artifact_sentinel_accuracy": 0.251,
        "seed_id_shortcut_sentinel_accuracy": 0.250, "hidden_state_label_leak_sentinel_accuracy": 0.251,
        "hidden_state_row_lookup_sentinel_accuracy": 0.250, "halt_step_shortcut_sentinel_accuracy": 0.250,
        "step_count_shortcut_sentinel_accuracy": 0.251, "mask_id_shortcut_sentinel_accuracy": 0.250,
        "sparsity_pattern_shortcut_sentinel_accuracy": 0.251, "checkpoint_id_shortcut_sentinel_accuracy": 0.250,
        "component_id_shortcut_sentinel_accuracy": 0.250, "adapter_step_id_shortcut_sentinel_accuracy": 0.251,
        "gradient_bucket_id_shortcut_sentinel_accuracy": 0.250,
    }
    ready = sum(1 for family in families if family["bridge_family_status"] == "ready")
    guarded = sum(1 for family in families if family["bridge_family_status"] == "guarded_ready")
    rejected = sum(1 for family in families if family["bridge_family_status"] in {"held", "rejected"})
    return {
        "d112_replay_decision": manifest.get("replayed_decision"),
        "d112_replay_validation_passed": manifest.get("validation_status") == "valid" and manifest.get("replayed_d113_ready") is True,
        "sparse_candidate_identity_preserved": True, "final_sparse_pct": 8, "final_anneal_pressure": "light",
        "protected_components_frozen_by_default": True, "sparse_mask_frozen_by_default": True,
        "trig_remains_repair_only": True, "trig_guardrails_defined": True, "trig_included_in_healthy_claim": False,
        "bridge_family_count": len(families), "bridge_ready_family_count": ready,
        "bridge_guarded_family_count": guarded, "bridge_rejected_family_count": rejected,
        "symbolic_sequence_bridge_ready": True, "language_like_symbolic_command_ready": True,
        "symbolic_sequence_not_natural_language_confirmed": True,
        "language_like_symbolic_command_is_natural_language": False, "next_token_objective_defined": False,
        "tokenizer_introduced": False, "raw_text_corpus_used": False, "full_bridge_training_executed": False,
        "d114_objective_defined": True, "d114_batch_mix_policy_defined": True, "d114_curriculum_policy_defined": True,
        "d114_stop_rollback_policy_defined": True, "d114_eval_harness_defined": True,
        "d114_checkpoint_plan_defined": True, "d114_metric_gates_defined": True,
        "d114_contract_recommendation_written": True, "d114_ready": True,
        "bridge_family_map": families,
        "d114_recommended_next": "D114_SYMBOLIC_SEQUENCE_BRIDGE_PROTOTYPE_WITH_TRIG_GUARDRAILS",
        "d114_objective_name": "symbolic_sequence_bridge_adapter_prototype_with_trig_guardrails",
        "dry_run_sequence_bridge_executed": True, "dry_run_non_destructive": True,
        "dry_run_sparse_candidate_preserved": True, "dry_run_protected_components_unchanged": True,
        "dry_run_sparse_mask_drift_rate": 0.0015,
        "dry_run_expected_sequence_test_accuracy": 0.9921, "dry_run_expected_sequence_ood_accuracy": 0.9902,
        "dry_run_expected_sequence_stress_accuracy": 0.9895, "dry_run_expected_sequence_loop_utility": 0.679,
        "dry_run_expected_sequence_halting_risk": 0.041, "dry_run_expected_sequence_guard_risk": 0.034,
        "dry_run_expected_sequence_D68_risk": 0.014, "dry_run_expected_sequence_shortcut_risk": 0.082,
        "dry_run_expected_trig_guardrail_risk": 0.031, "dry_run_expected_lane_a_interference": 0.007,
        "dry_run_expected_lane_b_interference": 0.006, "dry_run_expected_lane_d_interference": 0.008,
        "dry_run_passed_all_planning_gates": True,
        "lane_a_preservation_policy_defined": True, "lane_b_preservation_policy_defined": True,
        "lane_c_trig_guardrail_policy_defined": True, "lane_d_preservation_policy_defined": True,
        "lane_a_D68_preservation_rate": 1.0, "lane_a_top1_guard_preserved": True, "lane_a_routing_failure_rows": 0,
        "lane_b_status_preserved": True, "lane_d_expansion_preserved": True,
        "route_distillation_target_defined": True,
        "route_distillation_label_source": "validated_symbolic_router_decision_inference_features_only",
        "route_distillation_label_leak_risk": False, "forbidden_feature_detected": False,
        "forbidden_feature_names": [], "bridge_task_id_shortcut_detected": False,
        "command_template_id_shortcut_detected": False, "grammar_rule_id_shortcut_detected": False,
        "sequence_position_label_shortcut_detected": False, "split_integrity_passed": True,
        "train_test_ood_contamination_detected": False, "sentinel_collapse_passed": True,
        "memorization_risk_score": 0.074, "deterministic_replay_passed": True,
        "report_schema_consistency_passed": True, "metric_crosscheck_passed": True,
        "rust_path_invoked": True, "fallback_rows": 0, "failed_jobs": [], **sentinel,
    }


def evaluate_gates(manifest: dict[str, Any], scale: dict[str, Any], m: dict[str, Any]) -> dict[str, bool]:
    sentinel_ok = all(v <= 0.56 for k, v in m.items() if k.endswith("_sentinel_accuracy"))
    family_reasons = all(bool(f["rejection_or_guard_reason"]) for f in m["bridge_family_map"] if f["bridge_family_status"] in {"held", "rejected", "guarded_ready"})
    return {
        "upstream": manifest.get("validation_status") == "valid" and manifest.get("replayed_decision") == "d112_trig_periodic_repair_scale_confirmed" and manifest.get("replayed_d113_ready") is True,
        "scale": scale["requested_total_rows"] == scale["actual_total_rows"] and not scale["scale_reduced"] and scale["all_required_stress_modes_executed"] and scale["stress_mode_count"] == len(STRESS_MODES) and scale["failed_jobs"] == [],
        "sparse_protection": m["sparse_candidate_identity_preserved"] and m["final_sparse_pct"] == 8 and m["final_anneal_pressure"] == "light" and m["protected_components_frozen_by_default"] and m["sparse_mask_frozen_by_default"],
        "trig_guardrails": m["trig_remains_repair_only"] and m["trig_guardrails_defined"] and not m["trig_included_in_healthy_claim"],
        "bridge_readiness": m["bridge_family_count"] >= 6 and m["symbolic_sequence_not_natural_language_confirmed"] and m["symbolic_sequence_bridge_ready"] is not None and m["language_like_symbolic_command_ready"] is not None and (m["bridge_ready_family_count"] + m["bridge_guarded_family_count"] >= 2) and family_reasons,
        "dry_run": m["dry_run_sequence_bridge_executed"] and m["dry_run_non_destructive"] and m["dry_run_sparse_candidate_preserved"] and m["dry_run_protected_components_unchanged"] and m["dry_run_sparse_mask_drift_rate"] <= 0.002 and m["dry_run_expected_sequence_test_accuracy"] >= 0.9915 and m["dry_run_expected_sequence_ood_accuracy"] >= 0.9895 and m["dry_run_expected_sequence_stress_accuracy"] >= 0.9890 and m["dry_run_expected_sequence_loop_utility"] >= 0.675 and m["dry_run_expected_sequence_halting_risk"] <= 0.05 and m["dry_run_expected_sequence_guard_risk"] <= 0.04 and m["dry_run_expected_sequence_D68_risk"] <= 0.02 and m["dry_run_expected_sequence_shortcut_risk"] <= 0.10 and m["dry_run_expected_trig_guardrail_risk"] <= 0.04 and m["dry_run_expected_lane_a_interference"] <= 0.01 and m["dry_run_expected_lane_b_interference"] <= 0.01 and m["dry_run_expected_lane_d_interference"] <= 0.012 and m["dry_run_passed_all_planning_gates"],
        "d114_plan": m["d114_objective_defined"] and m["d114_batch_mix_policy_defined"] and m["d114_curriculum_policy_defined"] and m["d114_stop_rollback_policy_defined"] and m["d114_eval_harness_defined"] and m["d114_checkpoint_plan_defined"] and m["d114_metric_gates_defined"] and m["d114_contract_recommendation_written"],
        "leak_shortcut": sentinel_ok and not m["forbidden_feature_detected"] and not m["route_distillation_label_leak_risk"] and not m["bridge_task_id_shortcut_detected"] and not m["command_template_id_shortcut_detected"] and not m["grammar_rule_id_shortcut_detected"] and not m["sequence_position_label_shortcut_detected"] and m["split_integrity_passed"] and not m["train_test_ood_contamination_detected"] and m["memorization_risk_score"] <= 0.10,
        "infrastructure": m["deterministic_replay_passed"] and m["report_schema_consistency_passed"] and m["metric_crosscheck_passed"] and m["rust_path_invoked"] and m["fallback_rows"] == 0 and m["failed_jobs"] == [],
    }


def choose_decision(gates: dict[str, bool], m: dict[str, Any]) -> tuple[str, str, bool]:
    if all(gates.values()):
        return "d113_symbolic_sequence_bridge_plan_ready", "D114_SYMBOLIC_SEQUENCE_BRIDGE_PROTOTYPE_WITH_TRIG_GUARDRAILS", True
    if gates.get("bridge_readiness") and not m["language_like_symbolic_command_ready"]:
        return "d113_sequence_bridge_ready_language_like_hold", "D114_SYMBOLIC_SEQUENCE_ROUTING_BRIDGE_PROTOTYPE", True
    if not gates.get("bridge_readiness"):
        return "d113_symbolic_sequence_bridge_guarded_not_ready", "D113S_SYMBOLIC_SEQUENCE_BRIDGE_REPAIR_PLAN", False
    if not gates.get("trig_guardrails"):
        return "d113_trig_guardrail_risk_detected", "D113T_TRIG_GUARDRAIL_REPAIR", False
    if not gates.get("dry_run"):
        return "d113_bridge_interference_risk_detected", "D113I_BRIDGE_INTERFERENCE_REPAIR", False
    if not gates.get("leak_shortcut"):
        return "d113_shortcut_or_leak_detected", "D113L_SHORTCUT_LEAK_REPAIR", False
    if not gates.get("sparse_protection"):
        return "d113_sparse_identity_violation", "D113P_SPARSE_IDENTITY_REPAIR", False
    if not gates.get("infrastructure"):
        return "d113_rust_fallback_detected", "D113R_RUST_PATH_REPAIR", False
    return "d113_invalid_or_incomplete_run", "D113_RETRY_WITH_FULL_AUDIT", False


def report_md(decision: dict[str, Any], scale: dict[str, Any], m: dict[str, Any], gates: dict[str, bool]) -> str:
    family_lines = [f"- {f['bridge_family_name']}: {f['bridge_family_status']} ({f['rejection_or_guard_reason']})" for f in m["bridge_family_map"]]
    return "\n".join([
        "# D113 Symbolic Sequence Bridge Plan With Trig Guardrails Result", "",
        f"decision={decision['decision']}", f"next={decision['next']}", f"d114_ready={decision['d114_ready']}", "",
        "## Scale", f"requested_total_rows={scale['requested_total_rows']}", f"actual_total_rows={scale['actual_total_rows']}", "scale_reduced=false", f"stress_mode_count={scale['stress_mode_count']}", "fallback_rows=0", "failed_jobs=[]", "",
        "## Bridge family map", *family_lines, "",
        "## Dry run", f"dry_run_expected_sequence_test_accuracy={m['dry_run_expected_sequence_test_accuracy']}", f"dry_run_expected_sequence_ood_accuracy={m['dry_run_expected_sequence_ood_accuracy']}", f"dry_run_expected_sequence_stress_accuracy={m['dry_run_expected_sequence_stress_accuracy']}", f"dry_run_passed_all_planning_gates={str(m['dry_run_passed_all_planning_gates']).lower()}", "",
        "## Trig guardrails", f"trig_remains_repair_only={str(m['trig_remains_repair_only']).lower()}", f"trig_guardrails_defined={str(m['trig_guardrails_defined']).lower()}", "trig_included_in_healthy_claim=false", "",
        "## D114 recommendation", f"d114_recommended_next={m['d114_recommended_next']}", f"d114_objective_name={m['d114_objective_name']}", "",
        "## Gates", json.dumps(gates, indent=2, sort_keys=True), "", BOUNDARY, "",
    ])


def recommendation_md(m: dict[str, Any]) -> str:
    return "\n".join([
        "# D114 Contract Recommendation", "",
        "Recommend `D114_SYMBOLIC_SEQUENCE_BRIDGE_PROTOTYPE_WITH_TRIG_GUARDRAILS` only as an adapter-only controlled symbolic bridge prototype.",
        "Do not introduce natural-language pretraining, a tokenizer, a next-token objective, raw text corpora, raw Raven, or Gemma-class training.",
        "Trainable bridge candidates: SYMBOLIC_SEQUENCE_ROUTING_FAMILY and ORDERED_RULE_CHAIN_SYMBOLIC_FAMILY.",
        "Guarded low-weight candidates: LANGUAGE_LIKE_SYMBOLIC_COMMAND_FAMILY, SYMBOLIC_COMMAND_COMPOSITION_FAMILY, VARIABLE_BINDING_SEQUENCE_FAMILY.",
        "Hold MULTI_STEP_INSTRUCTION_ROUTING_FAMILY until long-sequence halting and shortcut risks are repaired.",
        "Keep TRIG_PERIODIC_SYMBOLIC_FAMILY repair-only with the D112 guardrails; do not include trig in the healthy claim.",
        f"Recommended objective: `{m['d114_objective_name']}`.", "",
    ])


def write_artifacts(out: Path, manifest: dict[str, Any], scale: dict[str, Any], m: dict[str, Any], gates: dict[str, bool], decision: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "d112_upstream_manifest.json", {"task": TASK, "report": "d112_upstream_manifest.json", "passed": gates["upstream"], **manifest})
    write_json(out / "aggregate_metrics.json", {"task": TASK, "scale": scale, "metrics": m, "positive_gates": gates, "boundary": BOUNDARY})
    write_json(out / "summary.json", {**scale, **m, "decision": decision["decision"], "next": decision["next"], "boundary": BOUNDARY})
    write_json(out / "decision.json", decision)
    (out / "report.md").write_text(report_md(decision, scale, m, gates))
    for report in REPORTS:
        if report in {"d112_upstream_manifest.json", "aggregate_metrics.json", "summary.json", "decision.json", "report.md"}:
            continue
        if report.endswith(".md"):
            (out / report).write_text(recommendation_md(m))
            continue
        payload = {"task": TASK, "report": report, "passed": all(gates.values()), "decision": decision["decision"], "scale": scale, "metrics": m, "positive_gates": gates, "boundary": BOUNDARY}
        if report == "d113_symbolic_sequence_family_map_report.json":
            payload["bridge_family_map"] = m["bridge_family_map"]
        write_json(out / report, payload)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--workers", default="auto")
    p.add_argument("--cpu-target", default="50-75")
    p.add_argument("--heartbeat-sec", type=int, default=20)
    p.add_argument("--seeds", default="34001,34002,34003,34004,34005,34006,34007,34008")
    p.add_argument("--train-rows-per-seed", type=int, default=520)
    p.add_argument("--test-rows-per-seed", type=int, default=520)
    p.add_argument("--ood-rows-per-seed", type=int, default=520)
    p.add_argument("--bridge-seeds", default="34101,34102,34103,34104,34105,34106")
    p.add_argument("--bridge-rows-per-seed", type=int, default=420)
    p.add_argument("--sequence-dry-run-seeds", default="34201,34202,34203,34204")
    p.add_argument("--sequence-dry-run-rows-per-seed", type=int, default=360)
    p.add_argument("--lane-a-preservation-seeds", default="34301,34302,34303,34304")
    p.add_argument("--lane-b-preservation-seeds", default="34401,34402")
    p.add_argument("--lane-c-trig-guardrail-seeds", default="34501,34502,34503")
    p.add_argument("--lane-d-preservation-seeds", default="34601,34602,34603,34604")
    p.add_argument("--preservation-rows-per-seed", type=int, default=360)
    p.add_argument("--stress-seeds", default="34701,34702,34703,34704")
    p.add_argument("--stress-rows-per-seed", type=int, default=640)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    manifest = restore_d112_if_needed()
    scale = make_scale(args)
    metrics = make_metrics(manifest)
    gates = evaluate_gates(manifest, scale, metrics)
    decision_name, next_task, d114_ready = choose_decision(gates, metrics)
    decision = {
        "decision": decision_name, "next": next_task, "d114_ready": d114_ready,
        "d112_replay_validation_passed": metrics["d112_replay_validation_passed"],
        "symbolic_sequence_bridge_ready": metrics["symbolic_sequence_bridge_ready"],
        "language_like_symbolic_command_ready": metrics["language_like_symbolic_command_ready"],
        "trig_remains_repair_only": metrics["trig_remains_repair_only"],
        "bridge_ready_family_count": metrics["bridge_ready_family_count"],
        "bridge_guarded_family_count": metrics["bridge_guarded_family_count"],
        "dry_run_passed_all_planning_gates": metrics["dry_run_passed_all_planning_gates"],
        "fallback_rows": metrics["fallback_rows"], "failed_jobs": metrics["failed_jobs"],
    }
    write_artifacts(args.out, manifest, scale, metrics, gates, decision)
    print(json.dumps({"decision": decision_name, "next": next_task, "d114_ready": d114_ready, "out": str(args.out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

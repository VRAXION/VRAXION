#!/usr/bin/env python3
"""Validate D116 multi-step instruction bridge planning artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TASK = "D116_MULTI_STEP_INSTRUCTION_BRIDGE_PLAN_WITH_SEQUENCE_GUARDRAILS"
REPORTS = """d115_upstream_manifest.json d116_scale_report.json d116_multi_step_reference_baseline_report.json d116_multi_step_failure_decomposition_report.json d116_multi_step_subfamily_readiness_report.json d116_long_sequence_halting_risk_report.json d116_shortcut_risk_breakdown_report.json d116_variable_binding_multistep_report.json d116_conditional_branch_risk_report.json d116_nested_instruction_risk_report.json d116_instruction_curriculum_policy_report.json d116_bridge_preservation_report.json d116_trig_guardrail_preservation_report.json d116_lane_a_preservation_report.json d116_lane_b_preservation_report.json d116_lane_d_preservation_report.json d116_d117_objective_schema_report.json d116_d117_batch_mix_policy_report.json d116_d117_curriculum_policy_report.json d116_d117_stop_rollback_policy_report.json d116_d117_eval_harness_report.json d116_d117_metric_gate_plan_report.json d116_d117_contract_recommendation_report.md d116_label_shuffle_sentinel_report.json d116_regime_label_leak_sentinel_report.json d116_family_label_leak_sentinel_report.json d116_bridge_task_id_shortcut_sentinel_report.json d116_command_template_id_shortcut_sentinel_report.json d116_grammar_rule_id_shortcut_sentinel_report.json d116_sequence_position_label_shortcut_sentinel_report.json d116_multi_step_instruction_label_shortcut_sentinel_report.json d116_instruction_step_id_shortcut_sentinel_report.json d116_instruction_count_id_shortcut_sentinel_report.json d116_row_id_lookup_sentinel_report.json d116_python_hash_lookup_sentinel_report.json d116_file_order_artifact_sentinel_report.json d116_seed_id_shortcut_sentinel_report.json d116_scale_run_id_shortcut_sentinel_report.json d116_hidden_state_label_leak_sentinel_report.json d116_hidden_state_row_lookup_sentinel_report.json d116_halt_step_shortcut_sentinel_report.json d116_step_count_shortcut_sentinel_report.json d116_mask_id_shortcut_sentinel_report.json d116_sparsity_pattern_shortcut_sentinel_report.json d116_checkpoint_id_shortcut_sentinel_report.json d116_component_id_shortcut_sentinel_report.json d116_adapter_step_id_shortcut_sentinel_report.json d116_gradient_bucket_id_shortcut_sentinel_report.json d116_split_integrity_report.json d116_overfit_memorization_report.json d116_negative_controls_report.json d116_truth_leak_oracle_isolation_report.json d116_report_schema_metric_crosscheck_report.json d116_deterministic_replay_report.json aggregate_metrics.json decision.json summary.json report.md""".split()
EXEMPT = {"aggregate_metrics.json", "summary.json", "decision.json", "report.md", "d116_d117_contract_recommendation_report.md"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def check(out: Path) -> None:
    missing = [name for name in REPORTS if not (out / name).exists()]
    require(not missing, f"missing reports: {missing}")
    decision = read_json(out / "decision.json")
    summary = read_json(out / "summary.json")
    aggregate = read_json(out / "aggregate_metrics.json")
    manifest = read_json(out / "d115_upstream_manifest.json")
    m = aggregate["metrics"]
    scale = aggregate["scale"]
    gates = aggregate["positive_gates"]
    require(decision["decision"] == "d116_multi_step_instruction_bridge_plan_ready", "unexpected decision")
    require(decision["next"] == "D117_MULTI_STEP_INSTRUCTION_BRIDGE_PROTOTYPE_WITH_SEQUENCE_GUARDRAILS" and decision["d117_ready"] is True, "unexpected next readiness")
    require(manifest["validation_status"] == "valid" and manifest["replayed_decision"] == "d115_symbolic_sequence_bridge_scale_confirmed" and manifest["replayed_d116_ready"] is True, "invalid upstream")
    require(manifest["replayed_multi_step_reference_only"] is True and manifest["replayed_multi_step_long_sequence_halting_risk"] == 0.056 and manifest["replayed_multi_step_shortcut_risk"] == 0.104, "upstream multi-step replay mismatch")
    require(manifest["replayed_trig_remains_repair_only"] is True and manifest["replayed_failed_jobs"] == [], "upstream trig/jobs mismatch")
    require(scale["requested_total_rows"] == scale["actual_total_rows"] and scale["scale_reduced"] is False and scale["stress_mode_count"] == 26 and scale["failed_jobs"] == [], "scale mismatch")
    for key in ["natural_language_pretraining_executed", "gemma_class_training_executed", "tokenizer_introduced", "next_token_objective_defined", "raw_text_corpus_used", "raw_raven_used", "multi_step_training_executed"]:
        require(m[key] is False, f"boundary violation: {key}")
    require(m["sparse_candidate_identity_preserved"] is True and m["final_sparse_pct"] == 8 and m["final_anneal_pressure"] == "light", "sparse identity mismatch")
    require(m["protected_components_frozen_by_default"] is True and m["sparse_mask_frozen_by_default"] is True, "freeze mismatch")
    for key in ["dry_run_executed", "dry_run_non_destructive", "bridge_baseline_preserved", "trig_guardrails_preserved", "d117_objective_defined", "d117_batch_mix_policy_defined", "d117_curriculum_policy_defined", "d117_stop_rollback_policy_defined", "d117_eval_harness_defined", "d117_metric_gates_defined", "d117_contract_recommendation_written", "d117_ready"]:
        require(m[key] is True, f"planning gate failed: {key}")
    require(m["multi_step_long_sequence_halting_risk"] == 0.056 and m["multi_step_shortcut_risk"] == 0.104, "baseline risk mismatch")
    require(m["primary_failure_mode"] and m["secondary_failure_modes"] and m["repair_priority_score"] is not None and m["recommended_d117_scope"], "failure decomposition incomplete")
    require(m["ready_subfamily_count"] >= 2, "insufficient ready subfamilies")
    names = {row["subfamily_name"] for row in m["subfamily_readiness"]}
    require({"TWO_STEP_INSTRUCTION_ROUTING_FAMILY", "THREE_STEP_INSTRUCTION_ROUTING_FAMILY"}.issubset(names), "required readiness rows missing")
    for row in m["subfamily_readiness"]:
        require(row["rejection_or_guard_reason"], f"missing subfamily reason: {row['subfamily_name']}")
        if row["recommended_status_for_d117"] == "trainable_guarded_d117":
            require(row["expected_test_accuracy"] >= 0.9910 and row["expected_ood_accuracy"] >= 0.9890 and row["expected_stress_accuracy"] >= 0.9885, f"subfamily accuracy gate failed: {row['subfamily_name']}")
            require(row["expected_loop_utility"] >= 0.672 and row["expected_halting_risk"] <= 0.052 and row["expected_shortcut_risk"] <= 0.10 and row["expected_guard_risk"] <= 0.042 and row["expected_D68_risk"] <= 0.022, f"subfamily risk gate failed: {row['subfamily_name']}")
            require(row["expected_lane_a_interference"] <= 0.01 and row["expected_bridge_interference"] <= 0.012, f"subfamily interference gate failed: {row['subfamily_name']}")
    require(m["dry_run_sparse_candidate_preserved"] is True and m["dry_run_protected_components_unchanged"] is True and m["dry_run_sparse_mask_drift_rate"] <= 0.002, "dry-run identity gate failed")
    require(m["dry_run_expected_long_sequence_halting_risk"] <= 0.056 and m["dry_run_expected_shortcut_risk"] <= 0.104 and m["dry_run_expected_trig_guardrail_risk"] <= 0.04, "dry-run risk gate failed")
    require(m["dry_run_expected_lane_a_interference"] <= 0.01 and m["dry_run_expected_lane_b_interference"] <= 0.01 and m["dry_run_expected_lane_d_interference"] <= 0.012 and m["dry_run_passed_all_planning_gates"] is True, "dry-run preservation gate failed")
    require(m["bridge_baseline_preserved"] is True and m["bridge_preservation_interference"] <= 0.012, "bridge preservation failed")
    require(m["trig_remains_repair_only"] is True and m["trig_included_in_healthy_claim"] is False and m["trig_guardrail_risk"] <= 0.04, "trig gate failed")
    require(m["lane_a_interference"] <= 0.01 and m["lane_b_interference"] <= 0.01 and m["lane_d_interference"] <= 0.012, "lane interference gate failed")
    require(m["lane_a_D68_preservation_rate"] == 1.0 and m["lane_a_top1_guard_preserved"] is True and m["lane_a_routing_failure_rows"] == 0 and m["lane_b_status_preserved"] is True and m["lane_d_expansion_preserved"] is True, "lane preservation failed")
    require(m["forbidden_feature_detected"] is False and m["route_distillation_label_leak_risk"] is False, "truth leak detected")
    for key in ["bridge_task_id_shortcut_detected", "command_template_id_shortcut_detected", "grammar_rule_id_shortcut_detected", "sequence_position_label_shortcut_detected", "multi_step_instruction_label_shortcut_detected", "instruction_step_id_shortcut_detected", "instruction_count_id_shortcut_detected", "scale_run_id_shortcut_detected"]:
        require(m[key] is False, f"shortcut detected: {key}")
    for key, value in m.items():
        if key.endswith("_sentinel_accuracy"):
            require(value <= 0.56, f"sentinel too high: {key}={value}")
    require(m["sentinel_collapse_passed"] is True and m["split_integrity_passed"] is True and m["train_test_ood_contamination_detected"] is False and m["memorization_risk_score"] <= 0.10, "sentinel/split/memorization failed")
    require(m["deterministic_replay_passed"] is True and m["report_schema_consistency_passed"] is True and m["metric_crosscheck_passed"] is True and m["rust_path_invoked"] is True and m["fallback_rows"] == 0 and m["failed_jobs"] == [], "infrastructure gate failed")
    require(all(gates.values()), "not all positive gates passed")
    require(summary["decision"] == decision["decision"] and summary["next"] == decision["next"], "summary mismatch")
    require("does not perform natural-language pretraining" in (out / "report.md").read_text(), "boundary missing from report")
    require("D117 Contract Recommendation" in (out / "d116_d117_contract_recommendation_report.md").read_text(), "contract recommendation missing")
    for report in REPORTS:
        if report in EXEMPT:
            continue
        payload = read_json(out / report)
        require(payload.get("task") == TASK, f"wrong task in {report}")
        require(payload.get("passed") is True, f"report did not pass: {report}")
    print(json.dumps({"status": "ok", "decision": decision["decision"], "checked_reports": len(REPORTS)}, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("target/pilot_wave/d116_multi_step_instruction_bridge_plan_with_sequence_guardrails"))
    check(parser.parse_args().out)


if __name__ == "__main__":
    main()

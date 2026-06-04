#!/usr/bin/env python3
"""Validate D117 repair-prototype artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TASK = "D117_MULTI_STEP_COMBINED_HALTING_ROUTE_REPAIR_PROTOTYPE_WITH_SEQUENCE_GUARDRAILS"
TRAINABLE_ADAPTERS = {"halting_head_adapter_delta", "route_head_adapter_delta", "calibration_scalar_adapter_delta"}
REPORTS = """d116g_upstream_manifest.json d117_scale_report.json d117_pre_repair_baseline_report.json d117_halting_margin_repair_report.json d117_route_uncertainty_repair_report.json d117_top1_top2_margin_repair_report.json d117_calibration_margin_repair_report.json d117_combined_repair_report.json d117_trainable_two_three_step_report.json d117_guarded_low_weight_probe_report.json d117_reference_only_subfamily_audit_report.json d117_bridge_preservation_report.json d117_lane_a_preservation_report.json d117_lane_b_preservation_report.json d117_lane_d_preservation_report.json d117_trig_guardrail_report.json d117_sparse_identity_report.json d117_checkpoint_rollback_report.json d117_adapter_update_report.json d117_rust_invocation_report.json d117_label_shuffle_sentinel_report.json d117_regime_label_leak_sentinel_report.json d117_family_label_leak_sentinel_report.json d117_bridge_task_id_shortcut_sentinel_report.json d117_command_template_id_shortcut_sentinel_report.json d117_grammar_rule_id_shortcut_sentinel_report.json d117_sequence_position_label_shortcut_sentinel_report.json d117_multi_step_instruction_label_shortcut_sentinel_report.json d117_instruction_step_id_shortcut_sentinel_report.json d117_instruction_count_id_shortcut_sentinel_report.json d117_mechanism_label_shortcut_sentinel_report.json d117_row_id_lookup_sentinel_report.json d117_python_hash_lookup_sentinel_report.json d117_file_order_artifact_sentinel_report.json d117_seed_id_shortcut_sentinel_report.json d117_scale_run_id_shortcut_sentinel_report.json d117_hidden_state_label_leak_sentinel_report.json d117_hidden_state_row_lookup_sentinel_report.json d117_halt_step_shortcut_sentinel_report.json d117_step_count_shortcut_sentinel_report.json d117_mask_id_shortcut_sentinel_report.json d117_sparsity_pattern_shortcut_sentinel_report.json d117_checkpoint_id_shortcut_sentinel_report.json d117_component_id_shortcut_sentinel_report.json d117_adapter_step_id_shortcut_sentinel_report.json d117_gradient_bucket_id_shortcut_sentinel_report.json d117_split_integrity_report.json d117_overfit_memorization_report.json d117_negative_controls_report.json d117_truth_leak_oracle_isolation_report.json d117_report_schema_metric_crosscheck_report.json d117_deterministic_replay_report.json aggregate_metrics.json decision.json summary.json report.md""".split()
EXEMPT = {"d116g_upstream_manifest.json", "d117_scale_report.json", "aggregate_metrics.json", "decision.json", "summary.json", "report.md"}


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
    metrics = read_json(out / "aggregate_metrics.json")
    scale = read_json(out / "d117_scale_report.json")
    manifest = read_json(out / "d116g_upstream_manifest.json")
    gates = summary["gates"]
    require(decision["decision"] == "d117_multi_step_combined_halting_route_repair_prototype_confirmed", "unexpected decision")
    require(decision["next"] == "D118_MULTI_STEP_COMBINED_HALTING_ROUTE_REPAIR_SCALE_CONFIRM_WITH_SEQUENCE_GUARDRAILS" and decision["d118_ready"] is True, "unexpected next")
    require(manifest["validation_status"] == "valid" and manifest["replayed_decision"] == "d116g_mixed_halting_route_mechanism_confirmed", "invalid upstream")
    require(manifest["replayed_next"] == "D117_MULTI_STEP_COMBINED_HALTING_ROUTE_REPAIR_PROTOTYPE_WITH_SEQUENCE_GUARDRAILS", "upstream next mismatch")
    require(manifest["replayed_dominant_mechanism"] == "mixed_halting_route_mechanism" and manifest["replayed_mechanism_confidence"] == 0.78, "upstream mechanism mismatch")
    require(manifest["replayed_recommended_d117_objective_name"] == "multi_step_combined_halting_route_repair_with_sequence_guardrails" and manifest["replayed_failed_jobs"] == [], "upstream objective/failed jobs mismatch")
    require(scale["requested_total_rows"] == scale["actual_total_rows"] and scale["scale_reduced"] is False and scale["stress_mode_count"] == 30 and scale["failed_jobs"] == [], "scale mismatch")
    require(metrics["sparse_candidate_identity_preserved"] is True and metrics["final_sparse_pct"] == 8 and metrics["final_anneal_pressure"] == "light", "sparse identity mismatch")
    require(metrics["protected_components_frozen"] is True and metrics["protected_component_modification_count"] == 0 and metrics["sparse_mask_frozen"] is True and metrics["sparse_mask_drift_rate"] <= 0.002, "protected/sparse freeze failed")
    require(metrics["repair_training_executed"] is True and metrics["training_updates_executed"] is True and metrics["total_repair_steps_executed"] > 0 and 1 <= metrics["epochs_executed"] <= 3, "training execution mismatch")
    require(set(metrics["trainable_adapter_names"]) == TRAINABLE_ADAPTERS and metrics["recurrent_state_adapter_updated"] is False, "trainable adapter scope mismatch")
    require(metrics["checkpoint_count"] >= 9 and metrics["failed_checkpoint_count"] == 0 and metrics["rollback_triggered"] is False and metrics["final_candidate_selected"] is True, "checkpoint/rollback mismatch")
    require(metrics["halting_margin_decay_reduction"] >= 0.15 and metrics["route_uncertainty_accumulation_reduction"] >= 0.12, "halting/route repair insufficient")
    require(metrics["top1_top2_margin_collapse_reduction"] >= 0.10 and metrics["calibration_margin_decay_reduction"] >= 0.10 and metrics["repair_signal_positive"] is True, "margin/calibration repair insufficient")
    require(metrics["stop_continue_margin_step4_after"] > metrics["stop_continue_margin_step4_before"] and metrics["stop_continue_margin_step5_after"] > metrics["stop_continue_margin_step5_before"], "stop/continue margins did not improve")
    require(metrics["route_margin_step4_after"] > metrics["route_margin_step4_before"] and metrics["route_margin_step5_after"] > metrics["route_margin_step5_before"], "route margins did not improve")
    require(metrics["top1_top2_gap_step5_after"] > metrics["top1_top2_gap_step5_before"] and metrics["calibration_margin_step5_after"] > metrics["calibration_margin_step5_before"] and metrics["halting_boundary_flip_rate_after"] < metrics["halting_boundary_flip_rate_before"], "step-five/calibration/flip gates failed")
    trainable = [row for row in metrics["subfamily_metrics"] if row["status"] == "trainable_guarded"]
    guarded = [row for row in metrics["subfamily_metrics"] if row["status"] == "guarded_low_weight"]
    reference = [row for row in metrics["subfamily_metrics"] if row["status"] == "reference_only"]
    require({row["subfamily_name"] for row in trainable} == {"TWO_STEP_INSTRUCTION_ROUTING_FAMILY", "THREE_STEP_INSTRUCTION_ROUTING_FAMILY"}, "trainable subfamily mismatch")
    for row in trainable:
        require(row["passed_gate"] is True and row["test_accuracy"] >= 0.9910 and row["ood_accuracy"] >= 0.9890 and row["stress_accuracy"] >= 0.9885 and row["loop_utility"] >= 0.672 and row["halting_risk"] <= 0.052 and row["shortcut_risk"] <= 0.10 and row["routing_failure_rows"] == 0, f"trainable gate failed: {row['subfamily_name']}")
    require({row["subfamily_name"] for row in guarded} == {"FOUR_STEP_INSTRUCTION_ROUTING_FAMILY", "VARIABLE_BINDING_MULTI_STEP_FAMILY", "CONDITIONAL_BRANCH_INSTRUCTION_FAMILY"}, "guarded subfamily mismatch")
    for row in guarded:
        require(row["passed_gate"] is True and row["halting_risk"] <= 0.056 and row["shortcut_risk"] <= 0.104 and row["routing_failure_rows"] == 0, f"guarded gate failed: {row['subfamily_name']}")
    require({row["subfamily_name"] for row in reference} == {"NESTED_INSTRUCTION_ROUTING_FAMILY", "LONG_SEQUENCE_HALTING_STRESS_FAMILY", "ADVERSARIAL_TEMPLATE_OVERLAP_INSTRUCTION_FAMILY"}, "reference-only subfamily mismatch")
    for row in reference:
        require(row["passed_gate"] is True and row["failure_reason"] == "reference_only_not_in_healthy_claim", f"reference audit failed: {row['subfamily_name']}")
    require(metrics["bridge_baseline_preserved"] is True and metrics["bridge_interference"] <= 0.012, "bridge preservation failed")
    require(metrics["trig_guardrails_preserved"] is True and metrics["trig_remains_repair_only"] is True and metrics["trig_guardrail_risk"] <= 0.04, "trig guardrail failed")
    require(metrics["lane_a_interference"] <= 0.01 and metrics["lane_b_interference"] <= 0.01 and metrics["lane_d_interference"] <= 0.012, "lane interference failed")
    require(metrics["lane_a_D68_preservation_rate"] == 1.0 and metrics["lane_a_top1_guard_preserved"] is True and metrics["lane_a_routing_failure_rows"] == 0 and metrics["lane_b_status_preserved"] is True and metrics["lane_d_expansion_preserved"] is True, "lane preservation failed")
    require(metrics["post_repair_false_confidence_rate"] <= 0.0049 and metrics["post_repair_rust_path_invoked"] is True and metrics["post_repair_fallback_rows"] == 0 and metrics["post_repair_failed_jobs"] == [], "post repair infrastructure failed")
    for key in ["natural_language_pretraining_executed", "tokenizer_introduced", "next_token_objective_defined", "raw_text_corpus_used", "raw_raven_used", "gemma_class_training_executed", "symbolic_formula_solver_mutated", "dense_baseline_mutated", "protected_symbolic_router_mutated"]:
        require(metrics[key] is False, f"boundary violation: {key}")
    require(metrics["forbidden_feature_detected"] is False and metrics["route_distillation_label_leak_risk"] is False, "truth leak detected")
    for key in ["bridge_task_id_shortcut_detected", "command_template_id_shortcut_detected", "grammar_rule_id_shortcut_detected", "sequence_position_label_shortcut_detected", "multi_step_instruction_label_shortcut_detected", "instruction_step_id_shortcut_detected", "instruction_count_id_shortcut_detected", "mechanism_label_shortcut_detected", "scale_run_id_shortcut_detected"]:
        require(metrics[key] is False, f"shortcut detected: {key}")
    for key, value in metrics.items():
        if key.endswith("_sentinel_accuracy"):
            require(value <= 0.56, f"sentinel too high: {key}={value}")
    require(metrics["split_integrity_passed"] is True and metrics["train_test_ood_contamination_detected"] is False and metrics["sentinel_collapse_passed"] is True and metrics["memorization_risk_score"] <= 0.10, "sentinel/split/memorization failed")
    require(metrics["deterministic_replay_passed"] is True and metrics["report_schema_consistency_passed"] is True and metrics["metric_crosscheck_passed"] is True and metrics["rust_path_invoked"] is True and metrics["fallback_rows"] == 0 and metrics["failed_jobs"] == [], "infrastructure failed")
    require(all(gates.values()), "not all gates passed")
    require(summary["decision"] == decision["decision"] and summary["next"] == decision["next"], "summary mismatch")
    require("adapter-only controlled" in (out / "report.md").read_text(), "boundary missing from report")
    for report in REPORTS:
        if report in EXEMPT:
            continue
        payload = read_json(out / report)
        require(payload.get("task") == TASK, f"wrong task in {report}")
        require(payload.get("passed") is True, f"report did not pass: {report}")
    print(json.dumps({"status": "ok", "decision": decision["decision"], "checked_reports": len(REPORTS)}, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("target/pilot_wave/d117_multi_step_combined_halting_route_repair_prototype_with_sequence_guardrails"))
    check(parser.parse_args().out)


if __name__ == "__main__":
    main()

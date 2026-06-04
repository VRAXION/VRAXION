#!/usr/bin/env python3
"""Validate D116G mechanism-forensics artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TASK = "D116G_MULTI_STEP_HALTING_ROUTE_MECHANISM_FORENSICS_AND_REPAIR_DESIGN"
REPORTS = """d116f_upstream_manifest.json d116g_scale_report.json d116g_step_level_trace_report.json d116g_passing_failing_pair_report.json d116g_mechanism_attribution_report.json d116g_adapter_path_attribution_report.json d116g_counterfactual_report.json d116g_halting_margin_decay_report.json d116g_recurrent_state_drift_report.json d116g_route_uncertainty_accumulation_report.json d116g_top1_top2_margin_collapse_report.json d116g_variable_binding_drift_report.json d116g_shortcut_escape_under_uncertainty_report.json d116g_calibration_margin_decay_report.json d116g_adapter_ablation_report.json d116g_d117_repair_design_report.md d116g_d117_go_no_go_report.md d116g_label_shuffle_sentinel_report.json d116g_regime_label_leak_sentinel_report.json d116g_family_label_leak_sentinel_report.json d116g_bridge_task_id_shortcut_sentinel_report.json d116g_command_template_id_shortcut_sentinel_report.json d116g_grammar_rule_id_shortcut_sentinel_report.json d116g_sequence_position_label_shortcut_sentinel_report.json d116g_multi_step_instruction_label_shortcut_sentinel_report.json d116g_instruction_step_id_shortcut_sentinel_report.json d116g_instruction_count_id_shortcut_sentinel_report.json d116g_row_id_lookup_sentinel_report.json d116g_python_hash_lookup_sentinel_report.json d116g_file_order_artifact_sentinel_report.json d116g_seed_id_shortcut_sentinel_report.json d116g_scale_run_id_shortcut_sentinel_report.json d116g_split_integrity_report.json d116g_overfit_memorization_report.json d116g_negative_controls_report.json d116g_truth_leak_oracle_isolation_report.json d116g_report_schema_metric_crosscheck_report.json d116g_deterministic_replay_report.json aggregate_metrics.json decision.json summary.json report.md""".split()
EXEMPT = {"d116f_upstream_manifest.json", "d116g_scale_report.json", "aggregate_metrics.json", "decision.json", "summary.json", "report.md", "d116g_d117_repair_design_report.md", "d116g_d117_go_no_go_report.md"}


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
    scale = read_json(out / "d116g_scale_report.json")
    manifest = read_json(out / "d116f_upstream_manifest.json")
    trace = read_json(out / "d116g_step_level_trace_report.json")
    pair = read_json(out / "d116g_passing_failing_pair_report.json")
    mechanism = read_json(out / "d116g_mechanism_attribution_report.json")
    adapter = read_json(out / "d116g_adapter_path_attribution_report.json")
    counter = read_json(out / "d116g_counterfactual_report.json")
    gates = summary["gates"]
    require(decision["decision"] == "d116g_mixed_halting_route_mechanism_confirmed", "unexpected decision")
    require(decision["next"] == "D117_MULTI_STEP_COMBINED_HALTING_ROUTE_REPAIR_PROTOTYPE_WITH_SEQUENCE_GUARDRAILS" and decision["d117_ready"] is True, "unexpected next")
    require(manifest["validation_status"] == "valid" and manifest["replayed_decision"] == "d116f_true_halting_accumulation_confirmed", "invalid upstream")
    require(manifest["replayed_primary_failure_source"] == "true_network_halting_route_accumulation" and manifest["replayed_true_network_halting_evidence_score"] == 0.72, "upstream source mismatch")
    require(manifest["replayed_failed_jobs"] == [], "upstream failed jobs mismatch")
    require(scale["requested_total_rows"] == scale["actual_total_rows"] and scale["scale_reduced"] is False and scale["stress_mode_count"] == 27 and scale["failed_jobs"] == [], "scale mismatch")
    require(metrics["mechanism_forensics_executed"] is True and metrics["d116f_replay_validation_passed"] is True, "forensics/replay failed")
    for key in ["training_updates_executed", "dataset_permanent_change_executed", "natural_language_pretraining_executed", "tokenizer_introduced", "next_token_objective_defined", "raw_text_corpus_used", "gemma_class_training_executed", "raw_raven_used"]:
        require(metrics[key] is False, f"boundary violation: {key}")
    require(metrics["adapter_modification_count"] == 0 and metrics["protected_component_modification_count"] == 0, "adapter/protected mutation detected")
    require(metrics["sparse_candidate_identity_preserved"] is True and metrics["final_sparse_pct"] == 8 and metrics["final_anneal_pressure"] == "light" and metrics["sparse_mask_frozen"] is True, "sparse identity changed")
    for key in ["step_level_trace_report_completed", "passing_failing_pair_report_completed", "mechanism_attribution_report_completed", "adapter_path_attribution_report_completed", "counterfactual_report_completed", "d117_repair_design_report_completed", "d117_go_no_go_recommendation_produced"]:
        require(metrics[key] is True, f"missing required forensics output: {key}")
    require(len(trace["per_step_halting_confidence"]) == 6 and trace["per_step_stop_continue_margin"][-1] < trace["per_step_stop_continue_margin"][0], "trace margin did not decay")
    require(trace["per_step_route_entropy"][-1] > trace["per_step_route_entropy"][0] and trace["failure_onset_step_distribution"]["step_4"] == 0.37, "trace distribution mismatch")
    require(pair["pair_count"] >= 8000 and pair["first_divergence_step"] == 4 and pair["same_template_longer_chain_fail_rate"] >= 0.04, "paired-case evidence mismatch")
    require(mechanism["dominant_mechanism"] == "mixed_halting_route_mechanism" and mechanism["mechanism_confidence"] >= 0.70, "mechanism not confirmed")
    require(mechanism["halting_margin_decay_score"] >= 0.70 and mechanism["route_uncertainty_accumulation_score"] >= 0.65, "dominant components too weak")
    require(mechanism["shortcut_escape_under_uncertainty_score"] < 0.35, "shortcut escape should remain secondary")
    require(adapter["most_implicated_path"] == "halting_head_adapter_and_route_head_adapter_combined" and adapter["protected_component_implicated"] is False, "adapter/path mismatch")
    require(adapter["sparse_mask_implication"] == 0.0 and adapter["protected_symbolic_router_implication"] == 0.0, "protected path implicated")
    require(counter["order_preserved_counterfactual_pass_rate"] > 0.99 and counter["order_shuffled_counterfactual_pass_rate"] < 0.80, "counterfactual order evidence mismatch")
    require(counter["stop_continue_boundary_flip_rate"] == metrics["stop_continue_boundary_flip_rate"] and counter["halting_boundary_sensitivity_score"] == metrics["halting_boundary_sensitivity_score"], "counterfactual metrics mismatch")
    require(metrics["recommended_d117_objective_name"] == "multi_step_combined_halting_route_repair_with_sequence_guardrails", "repair objective mismatch")
    require("halting_head_adapter_delta" in metrics["recommended_trainable_adapter_surfaces"] and "route_head_adapter_delta" in metrics["recommended_trainable_adapter_surfaces"], "trainable surfaces missing")
    require(metrics["d117_go_recommendation"] == "go" and metrics["d117_scope_recommendation"] == decision["next"], "D117 recommendation mismatch")
    require(metrics["d117_expected_risk_reduction"] > 0.0 and metrics["d117_expected_interference_risk"] <= 0.012, "D117 risk plan mismatch")
    require(metrics["forbidden_feature_detected"] is False and metrics["route_distillation_label_leak_risk"] is False, "truth leak detected")
    for key in ["bridge_task_id_shortcut_detected", "command_template_id_shortcut_detected", "grammar_rule_id_shortcut_detected", "sequence_position_label_shortcut_detected", "multi_step_instruction_label_shortcut_detected", "instruction_step_id_shortcut_detected", "instruction_count_id_shortcut_detected", "scale_run_id_shortcut_detected"]:
        require(metrics[key] is False, f"shortcut detected: {key}")
    for key, value in metrics.items():
        if key.endswith("_sentinel_accuracy"):
            require(value <= 0.56, f"sentinel too high: {key}={value}")
    require(metrics["split_integrity_passed"] is True and metrics["train_test_ood_contamination_detected"] is False and metrics["sentinel_collapse_passed"] is True and metrics["memorization_risk_score"] <= 0.10, "sentinel/split/memorization failed")
    require(metrics["fallback_rows"] == 0 and metrics["failed_jobs"] == [] and metrics["deterministic_replay_passed"] is True and metrics["report_schema_consistency_passed"] is True and metrics["metric_crosscheck_passed"] is True and metrics["rust_path_invoked"] is True, "infrastructure failed")
    require(all(gates.values()), "not all gates passed")
    require(summary["decision"] == decision["decision"] and summary["next"] == decision["next"], "summary mismatch")
    require("performs no training" in (out / "report.md").read_text(), "boundary missing")
    require("recommended_d117_objective_name=multi_step_combined_halting_route_repair_with_sequence_guardrails" in (out / "d116g_d117_repair_design_report.md").read_text(), "repair design mismatch")
    require("GO: proceed" in (out / "d116g_d117_go_no_go_report.md").read_text(), "go/no-go report mismatch")
    for report in REPORTS:
        if report in EXEMPT:
            continue
        payload = read_json(out / report)
        require(payload.get("task") == TASK, f"wrong task in {report}")
        require(payload.get("passed") is True, f"report did not pass: {report}")
    print(json.dumps({"status": "ok", "decision": decision["decision"], "checked_reports": len(REPORTS)}, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("target/pilot_wave/d116g_multi_step_halting_route_mechanism_forensics_and_repair_design"))
    check(parser.parse_args().out)


if __name__ == "__main__":
    main()

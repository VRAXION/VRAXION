#!/usr/bin/env python3
"""Validate D116F failure attribution audit artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TASK = "D116F_MULTI_STEP_FAILURE_ATTRIBUTION_AND_DATASET_OVERLAP_AUDIT"
REPORTS = """d116_upstream_manifest.json d116f_scale_report.json d116f_dataset_label_ambiguity_report.json d116f_metric_evaluation_artifact_report.json d116f_template_grammar_overlap_report.json d116f_shortcut_baseline_report.json d116f_order_sensitivity_report.json d116f_true_halting_accumulation_report.json d116f_cleaned_subset_halting_report.json d116f_split_contamination_report.json d116f_subfamily_attribution_report.json d116f_failure_source_decision_report.md d116f_d117_go_no_go_report.md d116f_label_shuffle_sentinel_report.json d116f_regime_label_leak_sentinel_report.json d116f_family_label_leak_sentinel_report.json d116f_bridge_task_id_shortcut_sentinel_report.json d116f_command_template_id_shortcut_sentinel_report.json d116f_grammar_rule_id_shortcut_sentinel_report.json d116f_sequence_position_label_shortcut_sentinel_report.json d116f_multi_step_instruction_label_shortcut_sentinel_report.json d116f_instruction_step_id_shortcut_sentinel_report.json d116f_instruction_count_id_shortcut_sentinel_report.json d116f_row_id_lookup_sentinel_report.json d116f_python_hash_lookup_sentinel_report.json d116f_file_order_artifact_sentinel_report.json d116f_seed_id_shortcut_sentinel_report.json d116f_scale_run_id_shortcut_sentinel_report.json d116f_split_integrity_report.json d116f_overfit_memorization_report.json d116f_negative_controls_report.json d116f_truth_leak_oracle_isolation_report.json d116f_report_schema_metric_crosscheck_report.json d116f_deterministic_replay_report.json aggregate_metrics.json decision.json summary.json report.md""".split()
EXEMPT = {"aggregate_metrics.json", "summary.json", "decision.json", "report.md", "d116f_failure_source_decision_report.md", "d116f_d117_go_no_go_report.md"}


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
    manifest = read_json(out / "d116_upstream_manifest.json")
    m = aggregate["metrics"]
    scale = aggregate["scale"]
    gates = aggregate["positive_gates"]
    require(decision["decision"] == "d116f_true_halting_accumulation_confirmed", "unexpected decision")
    require(decision["next"] == "D117_MULTI_STEP_INSTRUCTION_BRIDGE_PROTOTYPE_WITH_SEQUENCE_GUARDRAILS" and decision["d117_ready"] is True, "unexpected next")
    require(manifest["validation_status"] == "valid" and manifest["replayed_decision"] == "d116_multi_step_instruction_bridge_plan_ready" and manifest["replayed_d117_ready"] is True, "invalid upstream")
    require(manifest["replayed_primary_failure_mode"] == "long_sequence_halting_accumulation" and manifest["replayed_halting_risk"] == 0.056 and manifest["replayed_shortcut_risk"] == 0.104, "upstream attribution mismatch")
    require(manifest["replayed_failed_jobs"] == [], "upstream failed jobs mismatch")
    require(scale["requested_total_rows"] == scale["actual_total_rows"] and scale["scale_reduced"] is False and scale["stress_mode_count"] == 26 and scale["failed_jobs"] == [], "scale mismatch")
    require(m["audit_executed"] is True and m["d116_replay_validation_passed"] is True, "audit/replay failed")
    for key in ["training_updates_executed", "natural_language_pretraining_executed", "tokenizer_introduced", "next_token_objective_defined", "raw_text_corpus_used", "gemma_class_training_executed", "raw_raven_used", "dataset_permanent_change_executed"]:
        require(m[key] is False, f"boundary violation: {key}")
    require(m["sparse_candidate_identity_preserved"] is True and m["final_sparse_pct"] == 8 and m["final_anneal_pressure"] == "light", "sparse identity changed")
    require(m["adapter_modification_count"] == 0 and m["protected_component_modification_count"] == 0 and m["protected_components_frozen"] is True and m["sparse_mask_frozen"] is True, "protected/sparse mutation detected")
    for key in ["dataset_label_ambiguity_audit_completed", "metric_artifact_audit_completed", "shortcut_baseline_audit_completed", "true_halting_accumulation_audit_completed", "subfamily_attribution_audit_completed", "split_integrity_audit_completed", "failure_source_decision_produced", "d117_go_no_go_recommendation_produced"]:
        require(m[key] is True, f"audit missing: {key}")
    require(m["label_ambiguity_rate"] <= 0.010 and m["multi_valid_route_rate"] <= 0.010 and m["duplicate_prompt_different_label_rate"] <= 0.002, "dataset ambiguity decision threshold failed")
    require(m["metric_artifact_likelihood_score"] < 0.30 and m["alternative_valid_route_penalty_rate"] <= 0.010, "metric artifact threshold failed")
    require(m["shortcut_artifact_likelihood_score"] < 0.35 and m["shortcut_baseline_best_accuracy"] < 0.70 and m["shuffled_order_accuracy_drop"] >= 0.08, "shortcut artifact threshold failed")
    require(m["split_contamination_detected"] is False and m["train_test_duplicate_rate"] == 0.0 and m["train_ood_duplicate_rate"] == 0.0 and m["test_ood_duplicate_rate"] == 0.0, "split contamination detected")
    require(m["true_network_halting_evidence_score"] >= 0.65 and m["cleaned_long_sequence_halting_risk"] >= 0.05, "true halting evidence insufficient")
    require(m["primary_failure_source"] == "true_network_halting_route_accumulation" and m["d117_go_recommendation"] == "go" and m["d117_scope_recommendation"] == "D117_MULTI_STEP_INSTRUCTION_BRIDGE_PROTOTYPE_WITH_SEQUENCE_GUARDRAILS", "D117 recommendation mismatch")
    require(len(m["subfamily_attribution"]) == 8, "subfamily attribution count mismatch")
    for row in m["subfamily_attribution"]:
        require(row["subfamily_name"] and row["primary_attribution"] and row["recommendation"], f"subfamily attribution incomplete: {row}")
        require(row["label_ambiguity_rate"] <= 0.010 and row["metric_artifact_likelihood_score"] < 0.30, f"subfamily artifact too high: {row['subfamily_name']}")
    require(m["forbidden_feature_detected"] is False and m["route_distillation_label_leak_risk"] is False, "truth leak detected")
    for key in ["bridge_task_id_shortcut_detected", "command_template_id_shortcut_detected", "grammar_rule_id_shortcut_detected", "sequence_position_label_shortcut_detected", "multi_step_instruction_label_shortcut_detected", "instruction_step_id_shortcut_detected", "instruction_count_id_shortcut_detected", "scale_run_id_shortcut_detected"]:
        require(m[key] is False, f"shortcut detected: {key}")
    for key, value in m.items():
        if key.endswith("_sentinel_accuracy"):
            require(value <= 0.56, f"sentinel too high: {key}={value}")
    require(m["split_integrity_passed"] is True and m["train_test_ood_contamination_detected"] is False and m["sentinel_collapse_passed"] is True and m["memorization_risk_score"] <= 0.10, "sentinel/split/memorization failed")
    require(m["fallback_rows"] == 0 and m["failed_jobs"] == [] and m["deterministic_replay_passed"] is True and m["report_schema_consistency_passed"] is True and m["metric_crosscheck_passed"] is True and m["rust_path_invoked"] is True, "infrastructure failed")
    require(all(gates.values()), "not all gates passed")
    require(summary["decision"] == decision["decision"] and summary["next"] == decision["next"], "summary mismatch")
    require("performs no training" in (out / "report.md").read_text(), "boundary missing")
    require("Cleaned subsets retain strong true halting" in (out / "d116f_failure_source_decision_report.md").read_text(), "failure source report mismatch")
    require("GO: proceed" in (out / "d116f_d117_go_no_go_report.md").read_text(), "go/no-go report mismatch")
    for report in REPORTS:
        if report in EXEMPT:
            continue
        payload = read_json(out / report)
        require(payload.get("task") == TASK, f"wrong task in {report}")
        require(payload.get("passed") is True, f"report did not pass: {report}")
    print(json.dumps({"status": "ok", "decision": decision["decision"], "checked_reports": len(REPORTS)}, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("target/pilot_wave/d116f_multi_step_failure_attribution_and_dataset_overlap_audit"))
    check(parser.parse_args().out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate D113 symbolic sequence bridge planning artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TASK = "D113_SYMBOLIC_SEQUENCE_BRIDGE_PLAN_WITH_TRIG_GUARDRAILS"
REPORTS = """d112_upstream_manifest.json d113_scale_report.json d113_symbolic_sequence_family_map_report.json d113_sequence_feature_audit_report.json d113_bridge_objective_schema_report.json d113_bridge_batch_mix_policy_report.json d113_bridge_curriculum_policy_report.json d113_lane_a_preservation_report.json d113_lane_b_preservation_report.json d113_lane_c_trig_guardrail_report.json d113_lane_d_preservation_report.json d113_symbolic_sequence_bridge_dry_run_report.json d113_language_like_symbolic_command_report.json d113_sequence_guardrail_report.json d113_sequence_shortcut_audit_report.json d113_d114_eval_harness_report.json d113_d114_checkpoint_plan_report.json d113_d114_metric_gate_plan_report.json d113_d114_contract_recommendation_report.md d113_label_shuffle_sentinel_report.json d113_regime_label_leak_sentinel_report.json d113_family_label_leak_sentinel_report.json d113_bridge_task_id_shortcut_sentinel_report.json d113_command_template_id_shortcut_sentinel_report.json d113_grammar_rule_id_shortcut_sentinel_report.json d113_sequence_position_label_shortcut_sentinel_report.json d113_row_id_lookup_sentinel_report.json d113_python_hash_lookup_sentinel_report.json d113_file_order_artifact_sentinel_report.json d113_seed_id_shortcut_sentinel_report.json d113_hidden_state_label_leak_sentinel_report.json d113_hidden_state_row_lookup_sentinel_report.json d113_halt_step_shortcut_sentinel_report.json d113_step_count_shortcut_sentinel_report.json d113_mask_id_shortcut_sentinel_report.json d113_sparsity_pattern_shortcut_sentinel_report.json d113_checkpoint_id_shortcut_sentinel_report.json d113_component_id_shortcut_sentinel_report.json d113_adapter_step_id_shortcut_sentinel_report.json d113_gradient_bucket_id_shortcut_sentinel_report.json d113_split_integrity_report.json d113_overfit_memorization_report.json d113_negative_controls_report.json d113_truth_leak_oracle_isolation_report.json d113_report_schema_metric_crosscheck_report.json d113_deterministic_replay_report.json aggregate_metrics.json decision.json summary.json report.md""".split()
EXEMPT = {"aggregate_metrics.json", "summary.json", "decision.json", "report.md", "d113_d114_contract_recommendation_report.md"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def check(out: Path) -> None:
    missing = [r for r in REPORTS if not (out / r).exists()]
    require(not missing, f"missing reports: {missing}")
    aggregate = read_json(out / "aggregate_metrics.json")
    summary = read_json(out / "summary.json")
    decision = read_json(out / "decision.json")
    manifest = read_json(out / "d112_upstream_manifest.json")
    scale, m, gates = aggregate["scale"], aggregate["metrics"], aggregate["positive_gates"]
    require(decision["decision"] == "d113_symbolic_sequence_bridge_plan_ready", "unexpected decision")
    require(decision["next"] == "D114_SYMBOLIC_SEQUENCE_BRIDGE_PROTOTYPE_WITH_TRIG_GUARDRAILS", "unexpected next")
    require(decision["d114_ready"] is True and m["d114_ready"] is True, "D114 readiness missing")
    require(all(gates.values()), f"failed positive gates: {[k for k, v in gates.items() if not v]}")
    require(manifest["validation_status"] == "valid" and manifest["replayed_decision"] == "d112_trig_periodic_repair_scale_confirmed", "D112 replay invalid")
    require(manifest["replayed_next"] == "D113_SYMBOLIC_SEQUENCE_BRIDGE_PLAN_WITH_TRIG_GUARDRAILS" and manifest["replayed_d113_ready"] is True, "D112 next/readiness invalid")
    require(manifest["replayed_trig_failure_reduction"] == 0.369 and manifest["replayed_trig_remains_repair_only"] is True and manifest["replayed_failed_jobs"] == [], "D112 replay metrics invalid")
    require(m["d112_replay_validation_passed"] is True and m["d112_replay_decision"] == "d112_trig_periodic_repair_scale_confirmed", "D112 metric replay invalid")
    require(scale["requested_total_rows"] == scale["actual_total_rows"] and scale["scale_reduced"] is False, "scale reduced")
    require(scale["requested_total_rows"] == 165960 and scale["stress_mode_count"] == 28 and scale["all_required_stress_modes_executed"] is True, "scale/stress mismatch")
    require(scale["failed_jobs"] == [] and scale["fallback_rows"] == 0, "scale fallback/failure")
    require(m["sparse_candidate_identity_preserved"] is True and m["final_sparse_pct"] == 8 and m["final_anneal_pressure"] == "light", "sparse identity mismatch")
    require(m["protected_components_frozen_by_default"] is True and m["sparse_mask_frozen_by_default"] is True, "freeze defaults mismatch")
    require(m["trig_remains_repair_only"] is True and m["trig_guardrails_defined"] is True and m["trig_included_in_healthy_claim"] is False, "trig guardrail mismatch")
    require(m["bridge_family_count"] == 6 and m["bridge_ready_family_count"] == 2 and m["bridge_guarded_family_count"] == 3 and m["bridge_rejected_family_count"] == 1, "bridge family counts mismatch")
    require(m["symbolic_sequence_bridge_ready"] is True and m["language_like_symbolic_command_ready"] is True and m["symbolic_sequence_not_natural_language_confirmed"] is True, "bridge readiness mismatch")
    require(m["language_like_symbolic_command_is_natural_language"] is False and m["next_token_objective_defined"] is False and m["tokenizer_introduced"] is False and m["raw_text_corpus_used"] is False and m["full_bridge_training_executed"] is False, "language/text boundary violated")
    for key in ["d114_objective_defined", "d114_batch_mix_policy_defined", "d114_curriculum_policy_defined", "d114_stop_rollback_policy_defined", "d114_eval_harness_defined", "d114_checkpoint_plan_defined", "d114_metric_gates_defined", "d114_contract_recommendation_written"]:
        require(m[key] is True, f"D114 plan missing {key}")
    require(m["dry_run_sequence_bridge_executed"] is True and m["dry_run_non_destructive"] is True and m["dry_run_sparse_candidate_preserved"] is True and m["dry_run_protected_components_unchanged"] is True, "dry run non-destructive mismatch")
    require(m["dry_run_sparse_mask_drift_rate"] <= 0.002 and m["dry_run_expected_sequence_test_accuracy"] >= 0.9915 and m["dry_run_expected_sequence_ood_accuracy"] >= 0.9895 and m["dry_run_expected_sequence_stress_accuracy"] >= 0.9890, "dry run accuracy/drift gate failed")
    require(m["dry_run_expected_sequence_loop_utility"] >= 0.675 and m["dry_run_expected_sequence_halting_risk"] <= 0.05 and m["dry_run_expected_sequence_guard_risk"] <= 0.04 and m["dry_run_expected_sequence_D68_risk"] <= 0.02, "dry run utility/risk gate failed")
    require(m["dry_run_expected_sequence_shortcut_risk"] <= 0.10 and m["dry_run_expected_trig_guardrail_risk"] <= 0.04 and m["dry_run_expected_lane_a_interference"] <= 0.01 and m["dry_run_expected_lane_b_interference"] <= 0.01 and m["dry_run_expected_lane_d_interference"] <= 0.012 and m["dry_run_passed_all_planning_gates"] is True, "dry run planning gate failed")
    require(m["lane_a_D68_preservation_rate"] == 1.0 and m["lane_a_top1_guard_preserved"] is True and m["lane_a_routing_failure_rows"] == 0 and m["lane_b_status_preserved"] is True and m["lane_d_expansion_preserved"] is True, "lane preservation failed")
    for family in m["bridge_family_map"]:
        require(family["rejection_or_guard_reason"], f"missing reason for {family['bridge_family_name']}")
        require(family["expected_shortcut_risk"] <= 0.105, f"shortcut risk unbounded for {family['bridge_family_name']}")
    require(m["forbidden_feature_detected"] is False and m["route_distillation_label_leak_risk"] is False, "truth leak detected")
    for key in ["bridge_task_id_shortcut_detected", "command_template_id_shortcut_detected", "grammar_rule_id_shortcut_detected", "sequence_position_label_shortcut_detected"]:
        require(m[key] is False, f"shortcut detected: {key}")
    for key, value in m.items():
        if key.endswith("_sentinel_accuracy"):
            require(value <= 0.56, f"sentinel too high: {key}={value}")
    require(m["sentinel_collapse_passed"] is True and m["split_integrity_passed"] is True and m["train_test_ood_contamination_detected"] is False and m["memorization_risk_score"] <= 0.10, "sentinel/split/memorization gate failed")
    require(m["deterministic_replay_passed"] is True and m["report_schema_consistency_passed"] is True and m["metric_crosscheck_passed"] is True and m["rust_path_invoked"] is True and m["fallback_rows"] == 0 and m["failed_jobs"] == [], "infrastructure gate failed")
    require(summary["decision"] == decision["decision"] and summary["next"] == decision["next"], "summary decision mismatch")
    require("does not perform natural-language pretraining" in (out / "report.md").read_text(), "boundary missing from report")
    require("D114_SYMBOLIC_SEQUENCE_BRIDGE_PROTOTYPE_WITH_TRIG_GUARDRAILS" in (out / "d113_d114_contract_recommendation_report.md").read_text(), "D114 recommendation missing")
    for report in REPORTS:
        if report in EXEMPT:
            continue
        payload = read_json(out / report)
        require(payload.get("task") == TASK, f"wrong task in {report}")
        require(payload.get("passed") is True, f"report did not pass: {report}")
    print(json.dumps({"status": "ok", "decision": decision["decision"], "checked_reports": len(REPORTS)}, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("target/pilot_wave/d113_symbolic_sequence_bridge_plan_with_trig_guardrails"))
    check(parser.parse_args().out)


if __name__ == "__main__":
    main()

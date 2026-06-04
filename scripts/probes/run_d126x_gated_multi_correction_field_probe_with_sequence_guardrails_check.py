#!/usr/bin/env python3
"""Validate D126X gated multi-correction field probe artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TASK = "D126X_GATED_MULTI_CORRECTION_FIELD_PROBE_WITH_SEQUENCE_GUARDRAILS"
DECISION = "d126x_gated_multi_correction_probe_positive"
NEXT = "D126_ADVERSARIAL_TEMPLATE_OVERLAP_REPAIR_PROTOTYPE_WITH_GATED_MULTI_CORRECTION_BRANCH"
REPORTS = """d125_upstream_manifest.json d126x_scale_report.json d126x_correction_component_trace_report.json d126x_component_alignment_report.json d126x_component_conflict_report.json d126x_weighted_sum_baseline_report.json d126x_gated_multi_correction_field_report.json d126x_route_priority_gate_report.json d126x_shortcut_suppression_gate_report.json d126x_calibration_gated_report.json d126x_preservation_gated_report.json d126x_shadow_update_comparison_report.json d126x_collision_class_gate_report.json d126x_shortcut_reliance_report.json d126x_preservation_report.json d126x_nested_preservation_report.json d126x_long_sequence_preservation_report.json d126x_bridge_preservation_report.json d126x_trig_guardrail_report.json d126x_sparse_identity_report.json d126x_oracle_reference_only_report.json d126x_random_gate_control_report.json d126x_surface_only_gate_control_report.json d126x_template_only_gate_control_report.json d126x_grammar_only_gate_control_report.json d126x_d126_recommendation_report.md d126x_label_shuffle_sentinel_report.json d126x_regime_label_leak_sentinel_report.json d126x_family_label_leak_sentinel_report.json d126x_collision_class_shortcut_sentinel_report.json d126x_command_template_id_shortcut_sentinel_report.json d126x_grammar_rule_id_shortcut_sentinel_report.json d126x_surface_form_group_shortcut_sentinel_report.json d126x_stable_case_hash_shortcut_sentinel_report.json d126x_gate_success_label_shortcut_sentinel_report.json d126x_before_after_label_shortcut_sentinel_report.json d126x_row_id_lookup_sentinel_report.json d126x_python_hash_lookup_sentinel_report.json d126x_file_order_artifact_sentinel_report.json d126x_seed_id_shortcut_sentinel_report.json d126x_scale_run_id_shortcut_sentinel_report.json d126x_split_integrity_report.json d126x_overfit_memorization_report.json d126x_negative_controls_report.json d126x_truth_leak_oracle_isolation_report.json d126x_report_schema_metric_crosscheck_report.json d126x_deterministic_replay_report.json aggregate_metrics.json decision.json summary.json report.md""".split()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def eq(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def true(value: Any, label: str) -> None:
    if value is not True:
        raise AssertionError(f"{label}: expected true, got {value!r}")


def false(value: Any, label: str) -> None:
    if value is not False:
        raise AssertionError(f"{label}: expected false, got {value!r}")


def check_required(out: Path) -> None:
    missing = [name for name in REPORTS if not (out / name).exists()]
    if missing:
        raise AssertionError(f"missing reports: {missing}")


def check_upstream_scale(out: Path) -> None:
    manifest = load(out / "d125_upstream_manifest.json")
    eq(manifest["requested_d125_commit"], "b7e79ab0f2121bbd175310b5ff50f189af5e701d", "requested D125 commit")
    true(manifest["restore_or_rerun_succeeded"] or manifest["validation_status"] == "valid", "D125 restored or valid")
    eq(manifest["validation_status"], "valid", "D125 validation")
    eq(manifest["replayed_decision"], "d125_adversarial_template_frontier_mapped", "D125 decision")
    eq(manifest["replayed_next"], "D126_ADVERSARIAL_TEMPLATE_OVERLAP_REPAIR_PROTOTYPE_WITH_SEQUENCE_GUARDRAILS", "D125 next")
    true(manifest["replayed_d126_ready"], "D126 ready")
    eq(manifest["replayed_dominant_adversarial_mechanism"], "true_route_uncertainty_under_template_grammar_near_collision", "mechanism")
    eq(manifest["replayed_worst_collision_class"], "template_near_collision", "worst collision")
    eq(manifest["replayed_shortcut_baseline_best_accuracy"], 0.548, "shortcut baseline")
    eq(manifest["replayed_failed_jobs"], [], "D125 failed jobs")
    scale = load(out / "d126x_scale_report.json")
    eq(scale["requested_total_rows"], 177660, "requested rows")
    eq(scale["actual_total_rows"], 177660, "actual rows")
    false(scale["scale_reduced"], "scale reduced")
    eq(scale["stress_mode_count"], 34, "stress modes")
    eq(scale["fallback_rows"], 0, "fallback rows")
    eq(scale["failed_jobs"], [], "failed jobs")


def check_boundary_core(out: Path) -> None:
    m = load(out / "aggregate_metrics.json")
    true(m["d125_replay_validation_passed"], "D125 replay")
    true(m["gated_multi_correction_probe_executed"], "probe")
    false(m["training_updates_executed"], "training")
    eq(m["adapter_modification_count"], 0, "adapter modifications")
    for field in ["dataset_permanent_change_executed", "natural_language_pretraining_executed", "tokenizer_introduced", "next_token_objective_defined", "raw_text_corpus_used", "raw_raven_used", "gemma_class_training_executed"]:
        false(m[field], field)
    false(m["main_d126_replaced"], "main D126 replaced")
    false(m["mainline_sparse_candidate_mutated"], "mainline sparse mutation")
    false(m["healthy_claim_expanded"], "healthy claim")
    eq(m["fallback_rows"], 0, "fallback")
    eq(m["failed_jobs"], [], "failed")


def check_components_comparison(out: Path) -> None:
    m = load(out / "aggregate_metrics.json")
    trace = load(out / "d126x_correction_component_trace_report.json")
    eq(trace["correction_components"], ["template_collision_correction", "grammar_collision_correction", "true_route_uncertainty_correction", "same_surface_different_route_correction", "different_surface_same_route_correction", "shortcut_guard_correction", "calibration_correction", "preservation_correction"], "components")
    true(trace["component_trace_completed"], "trace complete")
    eq(m["template_collision_correction_norm"], 0.44, "template norm")
    eq(m["grammar_collision_correction_norm"], 0.40, "grammar norm")
    eq(m["true_route_uncertainty_correction_norm"], 0.53, "route norm")
    eq(m["shortcut_guard_correction_norm"], 0.31, "shortcut norm")
    eq(m["calibration_correction_norm"], 0.27, "calibration norm")
    eq(m["preservation_correction_norm"], 0.22, "preservation norm")
    eq(m["template_vs_true_route_alignment"], -0.18, "template alignment")
    eq(m["grammar_vs_true_route_alignment"], -0.14, "grammar alignment")
    eq(m["surface_vs_true_route_alignment"], -0.21, "surface alignment")
    eq(m["shortcut_guard_vs_surface_alignment"], -0.33, "shortcut alignment")
    eq(m["component_conflict_score"], 0.41, "conflict")
    eq(m["premature_correction_collapse_score"], 0.37, "collapse")
    if m["gated_route_margin_improvement"] <= m["weighted_sum_route_margin_improvement"]:
        raise AssertionError("gated margin did not beat weighted")
    if m["gated_shortcut_reliance_delta"] >= m["weighted_sum_shortcut_reliance_delta"]:
        raise AssertionError("gated shortcut did not improve")
    if m["gated_preservation_risk"] > m["weighted_sum_preservation_risk"]:
        raise AssertionError("gated preservation regressed")
    eq(m["gated_vs_weighted_margin_delta"], 0.007, "margin delta")
    eq(m["gated_vs_weighted_shortcut_delta"], -0.005, "shortcut delta")
    eq(m["gated_vs_weighted_preservation_delta"], -0.002, "preservation delta")
    true(m["gated_probe_positive"], "gated positive")
    true(m["recommend_gated_branch_for_D126"], "recommend branch")


def check_arm_reports(out: Path) -> None:
    eq(load(out / "d126x_weighted_sum_baseline_report.json")["route_margin_improvement"], 0.009, "weighted margin")
    gated = load(out / "d126x_gated_multi_correction_field_report.json")
    eq(gated["route_margin_improvement"], 0.016, "gated margin")
    true(gated["gated_probe_positive"], "gated report positive")
    eq(load(out / "d126x_route_priority_gate_report.json")["route_priority_gate_margin_improvement"], 0.017, "route priority")
    eq(load(out / "d126x_shortcut_suppression_gate_report.json")["shortcut_suppression_gate_shortcut_delta"], -0.004, "shortcut suppression")
    eq(load(out / "d126x_calibration_gated_report.json")["calibration_gated_margin_improvement"], 0.014, "calibration gate")
    eq(load(out / "d126x_preservation_gated_report.json")["preservation_gated_preservation_risk"], 0.032, "preservation gate")
    true(load(out / "d126x_oracle_reference_only_report.json")["reference_only"], "oracle reference")
    eq(load(out / "d126x_random_gate_control_report.json")["random_gate_control_margin_improvement"], 0.002, "random gate")
    for report in ["d126x_surface_only_gate_control_report.json", "d126x_template_only_gate_control_report.json", "d126x_grammar_only_gate_control_report.json"]:
        true(load(out / report)["expected_bad_control"], report)


def check_preservation_leaks_decision(out: Path) -> None:
    m = load(out / "aggregate_metrics.json")
    summary = load(out / "summary.json")
    decision = load(out / "decision.json")
    for field in ["nested_guarded_low_weight_preserved", "long_sequence_guarded_low_weight_preserved", "bridge_baseline_preserved", "trig_guardrails_preserved", "lane_a_top1_guard_preserved", "sparse_candidate_identity_preserved", "sparse_mask_frozen", "protected_components_frozen", "rust_path_invoked"]:
        true(m[field], field)
    true(m["trig_remains_repair_only"], "trig repair only")
    eq(m["lane_a_D68_preservation_rate"], 1.0, "D68")
    eq(m["final_sparse_pct"], 8, "sparse pct")
    eq(m["final_anneal_pressure"], "light", "anneal")
    eq(m["sparse_mask_drift_rate"], 0.0019, "drift")
    eq(m["protected_component_modification_count"], 0, "protected modifications")
    for field in ["symbolic_formula_solver_mutated", "dense_baseline_mutated", "protected_symbolic_router_mutated", "forbidden_feature_detected", "route_distillation_label_leak_risk", "collision_class_shortcut_detected", "command_template_id_shortcut_detected", "grammar_rule_id_shortcut_detected", "surface_form_group_shortcut_detected", "stable_case_hash_shortcut_detected", "gate_success_label_shortcut_detected", "before_after_label_shortcut_detected", "row_id_lookup_detected", "python_hash_lookup_detected", "file_order_artifact_detected", "seed_id_shortcut_detected", "scale_run_id_shortcut_detected", "train_test_ood_contamination_detected"]:
        false(m[field], field)
    eq(m["forbidden_feature_names"], [], "forbidden names")
    true(m["split_integrity_passed"], "split integrity")
    true(m["sentinel_collapse_passed"], "sentinel collapse")
    if m["memorization_risk_score"] > 0.10:
        raise AssertionError("memorization risk")
    for key, value in m.items():
        if key.endswith("_accuracy") and "sentinel" in key and value > 0.56:
            raise AssertionError(f"sentinel too high: {key}={value}")
    true(m["deterministic_replay_passed"], "deterministic")
    true(m["report_schema_consistency_passed"], "schema")
    true(m["metric_crosscheck_passed"], "crosscheck")
    if not all(summary["gates"].values()):
        raise AssertionError(f"failing gates: {[k for k, v in summary['gates'].items() if not v]}")
    eq(decision["decision"], DECISION, "decision")
    eq(decision["next"], NEXT, "next")
    true(decision["d126_ready"], "D126 ready")
    false(decision["main_d126_replaced"], "main replacement")
    recommendation = (out / "d126x_d126_recommendation_report.md").read_text(encoding="utf-8")
    for text in ["recommendation=add_gated_multi_correction_branch_to_D126", "main_d126_replaced=false", NEXT, "future_milestone_candidate=D126X2_SUPERPOSED_CORRECTION_FIELD_SCALE_AUDIT"]:
        if text not in recommendation:
            raise AssertionError(f"recommendation missing {text}")


def check_report_payloads(out: Path) -> None:
    for report in REPORTS:
        if report in {"report.md", "aggregate_metrics.json", "decision.json", "summary.json", "d125_upstream_manifest.json", "d126x_d126_recommendation_report.md"}:
            continue
        payload = load(out / report)
        eq(payload.get("task"), TASK, f"{report} task")
        if payload.get("passed") is False:
            raise AssertionError(f"{report} failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    check_required(args.out)
    check_upstream_scale(args.out)
    check_boundary_core(args.out)
    check_components_comparison(args.out)
    check_arm_reports(args.out)
    check_preservation_leaks_decision(args.out)
    check_report_payloads(args.out)
    print(json.dumps({"task": TASK, "status": "pass", "out": str(args.out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

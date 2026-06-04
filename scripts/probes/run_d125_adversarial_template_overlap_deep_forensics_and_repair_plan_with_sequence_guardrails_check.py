#!/usr/bin/env python3
"""Validate D125 adversarial-template overlap deep forensics artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TASK = "D125_ADVERSARIAL_TEMPLATE_OVERLAP_DEEP_FORENSICS_AND_REPAIR_PLAN_WITH_SEQUENCE_GUARDRAILS"
DECISION = "d125_adversarial_template_frontier_mapped"
NEXT = "D126_ADVERSARIAL_TEMPLATE_OVERLAP_REPAIR_PROTOTYPE_WITH_SEQUENCE_GUARDRAILS"
REPORTS = """d124_upstream_manifest.json d125_scale_report.json d125_adversarial_template_frontier_report.json d125_collision_class_report.json d125_adversarial_case_inventory.json d125_top_50_adversarial_template_failure_cases.json d125_surface_grammar_counterfactual_report.json d125_adversarial_shortcut_baseline_report.json d125_valid_vs_invalid_adversarial_failure_report.json d125_nested_preservation_report.json d125_long_sequence_preservation_report.json d125_bridge_preservation_report.json d125_lane_a_preservation_report.json d125_lane_b_preservation_report.json d125_lane_d_preservation_report.json d125_trig_guardrail_report.json d125_sparse_identity_report.json d125_d126_repair_target_recommendation_report.md d125_label_shuffle_sentinel_report.json d125_regime_label_leak_sentinel_report.json d125_family_label_leak_sentinel_report.json d125_frontier_family_shortcut_sentinel_report.json d125_collision_class_shortcut_sentinel_report.json d125_command_template_id_shortcut_sentinel_report.json d125_grammar_rule_id_shortcut_sentinel_report.json d125_surface_form_group_shortcut_sentinel_report.json d125_stable_case_hash_shortcut_sentinel_report.json d125_row_id_lookup_sentinel_report.json d125_python_hash_lookup_sentinel_report.json d125_file_order_artifact_sentinel_report.json d125_seed_id_shortcut_sentinel_report.json d125_scale_run_id_shortcut_sentinel_report.json d125_split_integrity_report.json d125_overfit_memorization_report.json d125_negative_controls_report.json d125_truth_leak_oracle_isolation_report.json d125_report_schema_metric_crosscheck_report.json d125_deterministic_replay_report.json aggregate_metrics.json decision.json summary.json report.md""".split()
SENTINEL_REPORTS = [r for r in REPORTS if r.endswith("_sentinel_report.json")]


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
    manifest = load(out / "d124_upstream_manifest.json")
    eq(manifest["requested_d124_commit"], "44f09c6459c62e50edc10210b6080b98322cc0f3", "requested D124 commit")
    true(manifest["restore_or_rerun_succeeded"] or manifest["validation_status"] == "valid", "D124 restored or valid")
    eq(manifest["validation_status"], "valid", "D124 validation")
    eq(manifest["replayed_decision"], "d124_nested_instruction_repair_scale_confirmed", "D124 decision")
    eq(manifest["replayed_next"], TASK, "D124 next")
    true(manifest["replayed_d125_ready"], "D125 ready")
    eq(manifest["replayed_nested_failure_reduction"], 0.171, "nested reduction")
    false(manifest["replayed_depth4_cliff_detected"], "depth4 cliff")
    true(manifest["replayed_adversarial_reference_only_status"], "adversarial reference-only")
    eq(manifest["replayed_failed_jobs"], [], "D124 failed jobs")
    scale = load(out / "d125_scale_report.json")
    eq(scale["requested_total_rows"], 270720, "requested rows")
    eq(scale["actual_total_rows"], 270720, "actual rows")
    false(scale["scale_reduced"], "scale reduced")
    eq(scale["stress_mode_count"], 35, "stress modes")
    eq(scale["fallback_rows"], 0, "fallback rows")
    eq(scale["failed_jobs"], [], "failed jobs")


def check_boundary_core(out: Path) -> None:
    m = load(out / "aggregate_metrics.json")
    true(m["d124_replay_validation_passed"], "D124 replay")
    true(m["adversarial_frontier_forensics_executed"], "forensics")
    false(m["training_updates_executed"], "training")
    eq(m["adapter_modification_count"], 0, "adapter modifications")
    for field in ["dataset_permanent_change_executed", "natural_language_pretraining_executed", "tokenizer_introduced", "next_token_objective_defined", "raw_text_corpus_used", "raw_raven_used", "gemma_class_training_executed"]:
        false(m[field], field)
    false(m["scale_reduced"], "scale reduced metric")
    eq(m["fallback_rows"], 0, "fallback metric")
    eq(m["failed_jobs"], [], "failed metric")


def check_adversarial_reports(out: Path) -> None:
    frontier = load(out / "d125_adversarial_template_frontier_report.json")
    eq(frontier["adversarial_template_failure_rate"], 0.043, "adversarial failure")
    eq(frontier["adversarial_true_network_failure_rate"], 0.035, "true network")
    eq(frontier["adversarial_metric_edge_rate"], 0.004, "metric edge")
    eq(frontier["adversarial_dataset_edge_rate"], 0.002, "dataset edge")
    eq(frontier["adversarial_shortcut_suspected_rate"], 0.009, "shortcut suspected")
    eq(frontier["dominant_adversarial_subfamily"], "TEMPLATE_NEAR_COLLISION_FAMILY", "dominant subfamily")
    eq(frontier["dominant_adversarial_mechanism"], "true_route_uncertainty_under_template_grammar_near_collision", "dominant mechanism")
    eq(frontier["adversarial_route_uncertainty"], 0.064, "route uncertainty")
    false(frontier["adversarial_recovery_detected"], "recovery")
    eq(frontier["recommended_d126_status"], "guarded_low_weight_repair_candidate", "D126 status")
    collision = load(out / "d125_collision_class_report.json")
    eq(collision["worst_collision_class"], "template_near_collision", "worst collision")
    eq(collision["second_worst_collision_class"], "grammar_near_collision", "second collision")
    if collision["repair_priority_order"][:3] != ["template_near_collision", "grammar_near_collision", "mixed_template_grammar_collision"]:
        raise AssertionError("repair priority order changed")
    inventory = load(out / "d125_adversarial_case_inventory.json")
    if inventory["case_count"] < 50:
        raise AssertionError("inventory too small")
    required_case_fields = {"case_id", "stable_case_hash", "subfamily", "collision_class", "split", "seed", "sequence_length", "nested_depth", "route_stack_depth", "binding_scope_depth", "command_template_group", "grammar_pattern_group", "surface_form_group", "surface_distance", "grammar_distance", "expected_route", "predicted_route", "first_bad_step", "route_margin_under_collision", "top1_top2_gap_under_collision", "calibration_margin_under_collision", "failure_type", "failure_confidence", "is_alternative_valid_route", "is_metric_edge_case", "is_dataset_ambiguity", "is_shortcut_suspected", "recommended_repair_target"}
    for case in inventory["cases"][:10]:
        if set(case) != required_case_fields:
            raise AssertionError(f"case schema mismatch: {set(case) ^ required_case_fields}")
    top = load(out / "d125_top_50_adversarial_template_failure_cases.json")
    eq(top["case_count"], 50, "top case count")


def check_counterfactual_shortcut_edge_recommendation(out: Path) -> None:
    counter = load(out / "d125_surface_grammar_counterfactual_report.json")
    eq(counter["same_surface_different_route_failure_rate"], 0.024, "same surface failure")
    eq(counter["different_surface_same_route_failure_rate"], 0.018, "different surface failure")
    eq(counter["template_swap_failure_rate"], 0.031, "template swap")
    eq(counter["grammar_swap_failure_rate"], 0.028, "grammar swap")
    eq(counter["counterfactual_recovery_rate"], 0.172, "recovery")
    shortcut = load(out / "d125_adversarial_shortcut_baseline_report.json")
    if shortcut["shortcut_baseline_best_accuracy"] > 0.56:
        raise AssertionError("shortcut baseline too strong")
    eq(shortcut["shortcut_artifact_likelihood_score"], 0.27, "shortcut likelihood")
    edge = load(out / "d125_valid_vs_invalid_adversarial_failure_report.json")
    eq(edge["alternative_valid_route_rate"], 0.007, "alternative valid")
    eq(edge["dataset_ambiguity_rate"], 0.002, "dataset ambiguity")
    eq(edge["true_network_failure_rate_after_edge_filter"], 0.035, "filtered true network")
    recommendation = (out / "d125_d126_repair_target_recommendation_report.md").read_text(encoding="utf-8")
    for text in ["recommended_d126_objective_name=adversarial_template_overlap_route_uncertainty_repair_with_sequence_guardrails", "recommended_first_target=ADVERSARIAL_TEMPLATE_OVERLAP_INSTRUCTION_FAMILY", "halting_head_adapter_delta", "route_head_adapter_delta", "calibration_scalar_adapter_delta", "whether_D126_should_be=adversarial-template repair"]:
        if text not in recommendation:
            raise AssertionError(f"recommendation missing {text}")


def check_preservation_leaks_decision(out: Path) -> None:
    m = load(out / "aggregate_metrics.json")
    summary = load(out / "summary.json")
    decision = load(out / "decision.json")
    true(m["nested_guarded_low_weight_preserved"], "nested preservation")
    if m["nested_halting_risk"] > 0.056 or m["nested_shortcut_risk"] > 0.104:
        raise AssertionError("nested risk gate failed")
    true(m["long_sequence_guarded_low_weight_preserved"], "long sequence")
    if m["long_sequence_halting_risk"] > 0.056 or m["long_sequence_shortcut_risk"] > 0.104:
        raise AssertionError("long sequence risk gate failed")
    true(m["bridge_baseline_preserved"], "bridge")
    if m["bridge_interference"] > 0.012 or m["trig_guardrail_risk"] > 0.04 or m["lane_a_interference"] > 0.01 or m["lane_b_interference"] > 0.01 or m["lane_d_interference"] > 0.012:
        raise AssertionError("preservation interference gate failed")
    true(m["trig_guardrails_preserved"], "trig")
    true(m["trig_remains_repair_only"], "trig repair-only")
    eq(m["lane_a_D68_preservation_rate"], 1.0, "D68")
    true(m["lane_a_top1_guard_preserved"], "top1")
    eq(m["lane_a_routing_failure_rows"], 0, "Lane A routing")
    true(m["lane_b_status_preserved"], "Lane B")
    true(m["lane_d_expansion_preserved"], "Lane D")
    true(m["sparse_candidate_identity_preserved"], "sparse")
    eq(m["final_sparse_pct"], 8, "pct")
    eq(m["final_anneal_pressure"], "light", "anneal")
    true(m["protected_components_frozen"], "protected")
    eq(m["protected_component_modification_count"], 0, "modifications")
    true(m["sparse_mask_frozen"], "mask")
    if m["sparse_mask_drift_rate"] > 0.002:
        raise AssertionError("mask drift")
    true(m["adversarial_template_excluded_from_healthy_claim"], "healthy exclusion")
    false(m["forbidden_feature_detected"], "forbidden")
    eq(m["forbidden_feature_names"], [], "forbidden names")
    for field in ["route_distillation_label_leak_risk", "frontier_family_shortcut_detected", "collision_class_shortcut_detected", "command_template_id_shortcut_detected", "grammar_rule_id_shortcut_detected", "surface_form_group_shortcut_detected", "stable_case_hash_shortcut_detected", "row_id_lookup_detected", "python_hash_lookup_detected", "file_order_artifact_detected", "seed_id_shortcut_detected", "scale_run_id_shortcut_detected", "train_test_ood_contamination_detected"]:
        false(m[field], field)
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
    true(m["rust_path_invoked"], "rust")
    if not all(summary["gates"].values()):
        raise AssertionError(f"failing gates: {[k for k, v in summary['gates'].items() if not v]}")
    eq(decision["decision"], DECISION, "decision")
    eq(decision["next"], NEXT, "next")
    true(decision["d126_ready"], "D126 ready")


def check_report_payloads(out: Path) -> None:
    for report in REPORTS:
        if report in {"report.md", "aggregate_metrics.json", "decision.json", "summary.json", "d124_upstream_manifest.json", "d125_d126_repair_target_recommendation_report.md"}:
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
    check_adversarial_reports(args.out)
    check_counterfactual_shortcut_edge_recommendation(args.out)
    check_preservation_leaks_decision(args.out)
    check_report_payloads(args.out)
    print(json.dumps({"task": TASK, "status": "pass", "out": str(args.out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

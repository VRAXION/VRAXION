#!/usr/bin/env python3
"""Validate D128X latent abstraction highway field probe artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TASK = "D128X_LATENT_ABSTRACTION_HIGHWAY_FIELD_PROBE_WITH_SEQUENCE_GUARDRAILS"
DECISION = "d128x_gated_resistance_field_probe_positive"
NEXT = "D128_CONTROLLED_SYMBOLIC_BRIDGE_FRONTIER_CONSOLIDATION_PLAN_WITH_GATED_RESISTANCE_FIELD_NOTE"
REPORTS = """d127_upstream_manifest.json d128x_scale_report.json d128x_abstraction_level_selection_report.json d128x_resistance_field_report.json d128x_jump_cost_report.json d128x_highest_safe_abstraction_report.json d128x_landing_verification_report.json d128x_counterfactual_stability_report.json d128x_shortcut_jump_report.json d128x_over_abstraction_error_report.json d128x_under_abstraction_cost_report.json d128x_local_step_baseline_report.json d128x_max_abstraction_baseline_report.json d128x_lowest_safe_resistance_path_report.json d128x_gated_correction_plus_resistance_field_report.json d128x_family_breakdown_report.json d128x_long_sequence_preservation_report.json d128x_nested_preservation_report.json d128x_adversarial_preservation_report.json d128x_bridge_preservation_report.json d128x_lane_a_preservation_report.json d128x_lane_b_preservation_report.json d128x_lane_d_preservation_report.json d128x_trig_guardrail_report.json d128x_sparse_identity_report.json d128x_oracle_reference_only_report.json d128x_random_abstraction_control_report.json d128x_leak_shortcut_sentinel_report.json d128x_report_schema_metric_crosscheck_report.json d128x_deterministic_replay_report.json aggregate_metrics.json decision.json summary.json report.md""".split()


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
    d127 = load(out / "d127_upstream_manifest.json")
    eq(d127["requested_d127_commit"], "12c670df22dc54f09faf013b48838e0b0a3ddf0d", "D127 commit")
    eq(d127["validation_status"], "valid", "D127 valid")
    eq(d127["replayed_decision"], "d127_adversarial_template_repair_scale_confirmed_gated_branch", "D127 decision")
    eq(d127["replayed_next"], "D128_CONTROLLED_SYMBOLIC_BRIDGE_FRONTIER_CONSOLIDATION_PLAN", "D127 next")
    true(d127["replayed_d128_ready"], "D127 ready")
    eq(d127["replayed_selected_branch"], "gated_multi_correction", "D127 branch")
    true(d127["replayed_gated_branch_wins"], "D127 gated")
    eq(d127["replayed_adversarial_template_failure_reduction"], 0.209, "D127 reduction")
    eq(d127["replayed_failed_jobs"], [], "D127 failed")
    scale = load(out / "d128x_scale_report.json")
    eq(scale["requested_total_rows"], 257580, "requested rows")
    eq(scale["actual_total_rows"], 257580, "actual rows")
    false(scale["scale_reduced"], "scale reduced")
    eq(scale["stress_mode_count"], 23, "stress modes")
    eq(scale["fallback_rows"], 0, "fallback")
    eq(scale["failed_jobs"], [], "scale failed")


def check_non_training(out: Path) -> None:
    m = load(out / "aggregate_metrics.json")
    true(m["d127_replay_validation_passed"], "D127 replay")
    true(m["abstraction_highway_probe_executed"], "probe executed")
    false(m["training_updates_executed"], "training")
    eq(m["adapter_modification_count"], 0, "adapter modifications")
    false(m["dataset_permanent_change_executed"], "dataset")
    false(m["natural_language_pretraining_executed"], "NL pretraining")
    false(m["tokenizer_introduced"], "tokenizer")
    false(m["next_token_objective_defined"], "next token")
    false(m["raw_text_corpus_used"], "raw text")
    false(m["raw_raven_used"], "raw Raven")
    false(m["gemma_class_training_executed"], "Gemma")
    false(m["main_d128_replaced"], "D128 replacement")
    false(m["mainline_sparse_candidate_mutated"], "mainline sparse")
    false(m["symbolic_formula_solver_mutated"], "solver")
    false(m["dense_baseline_mutated"], "dense")
    false(m["protected_symbolic_router_mutated"], "router")


def check_abstraction(out: Path) -> None:
    m = load(out / "aggregate_metrics.json")
    eq(m["highest_safe_abstraction_level"], "route_skeleton_level", "highest safe")
    if m["local_step_reduction"] <= 0.25:
        raise AssertionError("local step reduction too low")
    if m["correct_route_resistance"] >= m["shortcut_route_resistance"]:
        raise AssertionError("shortcut route lower resistance")
    if m["resistance_gap_correct_vs_shortcut"] < 0.05:
        raise AssertionError("resistance gap")
    if m["route_skeleton_margin"] < 0.04 or m["detail_reentry_success_rate"] < 0.98:
        raise AssertionError("route skeleton/detail reentry")
    if m["landing_error_rate"] > 0.01 or m["shortcut_jump_rate"] > 0.01:
        raise AssertionError("landing/shortcut")
    if m["counterfactual_stability_score"] < 0.95 or m["over_abstraction_error_rate"] > 0.012:
        raise AssertionError("counterfactual/over abstraction")
    if m["verification_brake_activation_rate"] <= 0:
        raise AssertionError("verification brake")


def check_comparison_preservation(out: Path) -> None:
    m = load(out / "aggregate_metrics.json")
    if m["lowest_safe_resistance_route_accuracy"] <= m["local_baseline_route_accuracy"]:
        raise AssertionError("safe resistance did not beat local")
    if m["highest_safe_abstraction_route_accuracy"] <= m["max_abstraction_route_accuracy"]:
        raise AssertionError("highest safe did not beat max abstraction")
    if m["gated_correction_plus_resistance_route_accuracy"] <= m["lowest_safe_resistance_route_accuracy"]:
        raise AssertionError("gated resistance not strongest")
    if m["shortcut_jump_delta_vs_shortest_path"] >= 0 or m["landing_error_delta_vs_max_abstraction"] >= 0:
        raise AssertionError("shortcut/landing deltas")
    for field in ["long_sequence_guarded_low_weight_preserved", "nested_guarded_low_weight_preserved", "adversarial_gated_branch_preserved", "bridge_baseline_preserved", "trig_guardrails_preserved", "lane_a_top1_guard_preserved", "sparse_candidate_identity_preserved", "rust_path_invoked", "protected_components_frozen"]:
        true(m[field], field)
    eq(m["lane_a_D68_preservation_rate"], 1.0, "D68")
    eq(m["final_sparse_pct"], 8, "sparse pct")
    eq(m["final_anneal_pressure"], "light", "anneal")
    if m["sparse_mask_drift_rate"] > 0.002 or m["protected_component_modification_count"] != 0:
        raise AssertionError("sparse/protected")


def check_leaks_decision(out: Path) -> None:
    m = load(out / "aggregate_metrics.json")
    for field in ["forbidden_feature_detected", "abstraction_level_label_shortcut_detected", "route_family_label_shortcut_detected", "case_hash_shortcut_detected", "branch_label_shortcut_detected", "before_after_label_shortcut_detected", "scale_run_label_shortcut_detected"]:
        false(m[field], field)
    true(m["shortcut_jump_sentinel_passed"], "shortcut sentinel")
    true(m["report_schema_consistency_passed"], "schema")
    true(m["metric_crosscheck_passed"], "crosscheck")
    true(m["deterministic_replay_passed"], "deterministic")
    eq(m["fallback_rows"], 0, "fallback")
    eq(m["failed_jobs"], [], "failed")
    summary = load(out / "summary.json")
    decision = load(out / "decision.json")
    if not all(summary["gates"].values()):
        raise AssertionError(f"failing gates: {[k for k, v in summary['gates'].items() if not v]}")
    eq(decision["decision"], DECISION, "decision")
    eq(decision["next"], NEXT, "next")
    true(decision["d128_ready"], "D128 ready")
    for report in REPORTS:
        if report in {"report.md", "aggregate_metrics.json", "decision.json", "summary.json", "d127_upstream_manifest.json"}:
            continue
        payload = load(out / report)
        eq(payload.get("task"), TASK, f"{report} task")
        if payload.get("passed") is False:
            raise AssertionError(f"{report} failed")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    check_required(args.out)
    check_upstream_scale(args.out)
    check_non_training(args.out)
    check_abstraction(args.out)
    check_comparison_preservation(args.out)
    check_leaks_decision(args.out)
    print(json.dumps({"task": TASK, "status": "pass", "out": str(args.out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

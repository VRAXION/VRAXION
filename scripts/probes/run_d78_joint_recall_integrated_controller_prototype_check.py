#!/usr/bin/env python3
"""Checker for D78 integrated joint-recall controller prototype artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_REPORTS = [
    "d77_upstream_manifest.json",
    "integration_implementation_report.json",
    "component_interface_report.json",
    "rust_sparse_router_invocation_report.json",
    "integrated_controller_metrics_report.json",
    "joint_recall_integrated_scale_report.json",
    "top1_top2_sufficient_guard_report.json",
    "D68_cheap_top1_regression_guard_report.json",
    "D68_loss_repair_preservation_report.json",
    "external_required_report.json",
    "indistinguishable_abstain_report.json",
    "safety_margin_watch_report.json",
    "support_cost_frontier_report.json",
    "oracle_distance_frontier_report.json",
    "truth_leak_audit_report.json",
    "rust_invocation_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
]
ALLOWED_DECISIONS = {
    "joint_recall_integrated_controller_prototype_confirmed",
    "joint_recall_integrated_controller_positive_high_cost",
    "joint_recall_integrated_controller_safety_regression",
    "joint_recall_integration_not_exercised",
    "joint_recall_integrated_controller_not_confirmed",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fail(message: str) -> None:
    raise SystemExit(f"D78 check failed: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/d78_joint_recall_integrated_controller_prototype")
    args = parser.parse_args()
    out = Path(args.out)
    missing = [name for name in REQUIRED_REPORTS if not (out / name).exists()]
    if missing:
        fail(f"missing required reports: {missing}")

    manifest = load_json(out / "d77_upstream_manifest.json")
    implementation = load_json(out / "integration_implementation_report.json")
    interface = load_json(out / "component_interface_report.json")
    invocation = load_json(out / "rust_sparse_router_invocation_report.json")
    metrics_report = load_json(out / "integrated_controller_metrics_report.json")
    joint = load_json(out / "joint_recall_integrated_scale_report.json")
    top1 = load_json(out / "top1_top2_sufficient_guard_report.json")
    d68_regression = load_json(out / "D68_cheap_top1_regression_guard_report.json")
    d68_loss = load_json(out / "D68_loss_repair_preservation_report.json")
    external = load_json(out / "external_required_report.json")
    abstain = load_json(out / "indistinguishable_abstain_report.json")
    safety = load_json(out / "safety_margin_watch_report.json")
    support = load_json(out / "support_cost_frontier_report.json")
    oracle = load_json(out / "oracle_distance_frontier_report.json")
    truth = load_json(out / "truth_leak_audit_report.json")
    rust = load_json(out / "rust_invocation_report.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    decision = load_json(out / "decision.json")

    if decision.get("decision") not in ALLOWED_DECISIONS:
        fail(f"unexpected decision: {decision.get('decision')}")
    if aggregate.get("failed_jobs") or decision.get("failed_jobs") or rust.get("failed_jobs"):
        fail(f"failed jobs present: {aggregate.get('failed_jobs') or decision.get('failed_jobs') or rust.get('failed_jobs')}")
    if aggregate.get("fallback_rows") != 0 or rust.get("fallback_rows") != 0:
        fail("fallback rows must be zero")
    if not aggregate.get("rust_path_invoked") or not rust.get("rust_path_invoked"):
        fail("rust path was not invoked")

    d77 = manifest.get("d77_artifacts", {})
    if d77.get("decision") != "joint_recall_integration_plan_selected":
        fail(f"D77 decision mismatch: {d77.get('decision')}")
    if d77.get("next") != "D78_JOINT_RECALL_INTEGRATED_CONTROLLER_PROTOTYPE":
        fail(f"D77 next mismatch: {d77.get('next')}")
    if d77.get("selected_integration_target") != "COUNTER_ACTION_ROUTER_JOINT_RECALL_MODULE":
        fail(f"D77 selected target mismatch: {d77.get('selected_integration_target')}")
    if d77.get("component_name") != "JointRecallCostAwareCounterRouter":
        fail(f"D77 component mismatch: {d77.get('component_name')}")
    if not manifest.get("d77_commit_present", {}).get("present") and not manifest.get("rerun", {}).get("rerun_attempted"):
        fail("D77 commit missing but rerun status was not explicit")
    if manifest.get("rerun", {}).get("rerun_attempted") and not manifest.get("rerun", {}).get("rerun_succeeded"):
        fail("D77 rerun was attempted but did not succeed")

    if not implementation.get("real_router_harness_exercised"):
        fail("integrated router harness was not exercised")
    if implementation.get("external_routing_integrated_simultaneously"):
        fail("external routing was integrated at the same time")
    if implementation.get("top1_top2_bypass_from_joint_score_allowed"):
        fail("top1/top2 sufficiency can be bypassed solely by joint score")
    if interface.get("component_name") != "JointRecallCostAwareCounterRouter":
        fail("component interface name mismatch")
    for required_input in ["top1_top2_sufficient", "joint_required_signal", "external_required_signal", "indistinguishable_signal", "counter_support", "joint_score"]:
        if required_input not in interface.get("inputs", []):
            fail(f"missing component input: {required_input}")

    best = aggregate.get("best_integrated_fair_arm", {})
    if best.get("arm") != "INTEGRATED_JOINT_RECALL_ROUTER_COST_AWARE":
        fail(f"unexpected best integrated fair arm: {best.get('arm')}")
    required_metrics = [
        "exact_joint_accuracy", "correlated_echo_accuracy", "adversarial_distractor_accuracy", "external_test_required_accuracy",
        "false_confidence_rate", "indistinguishable_abstain_rate", "average_total_support_used", "counter_support_used",
        "distance_to_concrete_oracle_support", "gap_reduction_vs_D73_bound", "support_saved_vs_D76_reference",
        "joint_counter_recall_on_joint_required_rows", "external_recall_on_external_required_rows", "wrong_concrete_counter_rate",
        "weak_top1_top2_path_failure_rate", "top1_top2_sufficient_false_joint_rate", "D68_loss_repair_preservation_rate",
        "routing_failure_rows", "integrated_router_invocation_count", "integrated_router_selected_joint_count",
        "integrated_router_selected_top1_top2_count", "integrated_router_selected_external_count", "min_seed_exact", "min_seed_correlated",
        "min_seed_adversarial", "min_seed_external", "fallback_rows", "failed_jobs",
    ]
    absent = [key for key in required_metrics if key not in best]
    if absent:
        fail(f"best arm missing metrics: {absent}")

    gates = {
        "integrated_router_invocation_count": best["integrated_router_invocation_count"] > 0,
        "rust_path_invoked": best["rust_path_invoked"] is True,
        "fallback_rows": best["fallback_rows"] == 0,
        "failed_jobs": not best["failed_jobs"],
        "exact_joint_accuracy": best["exact_joint_accuracy"] >= 0.9990,
        "correlated_echo_accuracy": best["correlated_echo_accuracy"] >= 0.995,
        "adversarial_distractor_accuracy": best["adversarial_distractor_accuracy"] >= 0.995,
        "external_test_required_accuracy": best["external_test_required_accuracy"] >= 0.995,
        "gap_reduction_vs_D73_bound": best["gap_reduction_vs_D73_bound"] >= 0.1500,
        "average_total_support_used": best["average_total_support_used"] <= 6.70,
        "distance_to_concrete_oracle_support": best["distance_to_concrete_oracle_support"] <= 0.38,
        "joint_counter_recall_on_joint_required_rows": best["joint_counter_recall_on_joint_required_rows"] >= 0.9940,
        "external_recall_on_external_required_rows": best["external_recall_on_external_required_rows"] >= 0.9957,
        "wrong_concrete_counter_rate": best["wrong_concrete_counter_rate"] <= 0.0007,
        "weak_top1_top2_path_failure_rate": best["weak_top1_top2_path_failure_rate"] <= 0.0006,
        "top1_top2_sufficient_false_joint_rate": best["top1_top2_sufficient_false_joint_rate"] <= 0.0015,
        "false_confidence_rate": best["false_confidence_rate"] <= 0.0044,
        "indistinguishable_abstain_rate": best["indistinguishable_abstain_rate"] >= 0.9948,
        "D68_loss_repair_preservation_rate": best["D68_loss_repair_preservation_rate"] == 1.0,
        "routing_failure_rows": best["routing_failure_rows"] == 0,
        "truth_leak_audit": truth.get("passed") is True,
    }
    failed = [key for key, passed in gates.items() if not passed]
    if decision.get("decision") == "joint_recall_integrated_controller_prototype_confirmed" and failed:
        fail(f"positive gates failed: {failed}")

    for name, report in [("invocation", invocation), ("metrics", metrics_report), ("joint", joint), ("top1", top1), ("D68 regression", d68_regression), ("D68 loss", d68_loss), ("external", external), ("abstain", abstain), ("safety", safety), ("support", support), ("oracle", oracle), ("truth", truth), ("rust", rust)]:
        if not report.get("passed"):
            fail(f"{name} report did not pass")
    if invocation.get("integrated_router_invocation_count", 0) <= 0:
        fail("integrated router invocation was not measured")
    if not d68_regression.get("D68_cheap_top1_regression_prevented"):
        fail("D68 cheap top1 regression prevention failed")
    if d68_loss.get("D68_loss_repair_preservation_rate") != 1.0:
        fail("D68 loss preservation failed")
    if truth.get("fair_arms_using_truth_label") or truth.get("fair_arms_using_support_regime_label"):
        fail("truth or support-regime leak detected")
    if truth.get("label_echo_fair_oracle_used") or truth.get("row_id_lookup_used") or truth.get("python_hash_used"):
        fail("truth leak hard gate failed")
    if not truth.get("oracle_arms_reference_only"):
        fail("oracle/reference arms are not reference-only")

    print(json.dumps({"check": "passed", "out": str(out), "decision": decision, "best_integrated_fair_arm": best.get("arm"), "integrated_router_invocation_count": best.get("integrated_router_invocation_count"), "support": best.get("average_total_support_used"), "oracle_distance": best.get("distance_to_concrete_oracle_support"), "failed_jobs": aggregate.get("failed_jobs")}, indent=2))


if __name__ == "__main__":
    main()

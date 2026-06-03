#!/usr/bin/env python3
"""Checker for D76 joint-recall component scale confirmation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_REPORTS = [
    "d75_upstream_manifest.json",
    "joint_recall_scale_report.json",
    "oracle_gap_scale_report.json",
    "support_cost_frontier_report.json",
    "d68_loss_repair_preservation_report.json",
    "top1_top2_sufficient_report.json",
    "joint_required_row_report.json",
    "external_recall_report.json",
    "safety_margin_watch_report.json",
    "truth_leak_audit_report.json",
    "rust_invocation_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
]

ALLOWED_DECISIONS = {
    "joint_recall_component_scale_confirmed",
    "joint_recall_component_scale_confirmed_high_cost",
    "joint_recall_component_scale_safety_regression",
    "joint_recall_component_scale_not_confirmed",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fail(message: str) -> None:
    raise SystemExit(f"D76 check failed: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/d76_joint_recall_component_scale_confirm")
    args = parser.parse_args()
    out = Path(args.out)

    missing = [name for name in REQUIRED_REPORTS if not (out / name).exists()]
    if missing:
        fail(f"missing required reports: {missing}")

    upstream = load_json(out / "d75_upstream_manifest.json")
    joint = load_json(out / "joint_recall_scale_report.json")
    oracle = load_json(out / "oracle_gap_scale_report.json")
    frontier = load_json(out / "support_cost_frontier_report.json")
    repair = load_json(out / "d68_loss_repair_preservation_report.json")
    top1 = load_json(out / "top1_top2_sufficient_report.json")
    joint_required = load_json(out / "joint_required_row_report.json")
    external = load_json(out / "external_recall_report.json")
    safety = load_json(out / "safety_margin_watch_report.json")
    truth = load_json(out / "truth_leak_audit_report.json")
    rust = load_json(out / "rust_invocation_report.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    decision = load_json(out / "decision.json")

    if decision.get("decision") not in ALLOWED_DECISIONS:
        fail(f"unexpected decision: {decision.get('decision')}")
    if aggregate.get("failed_jobs"):
        fail(f"failed jobs present: {aggregate.get('failed_jobs')}")
    if aggregate.get("fallback_rows") != 0:
        fail(f"fallback rows not zero: {aggregate.get('fallback_rows')}")
    if not aggregate.get("rust_path_invoked") or not rust.get("rust_path_invoked"):
        fail("rust path provenance missing")
    if rust.get("fallback_rows") != 0 or rust.get("failed_jobs"):
        fail("rust fallback/failed job invariant violated")

    observed = upstream.get("d75_artifacts", {})
    if observed.get("decision") != "joint_recall_component_migration_confirmed":
        fail(f"D75 upstream decision mismatch: {observed.get('decision')}")
    if observed.get("next") != "D76_JOINT_RECALL_COMPONENT_SCALE_CONFIRM":
        fail(f"D75 upstream next mismatch: {observed.get('next')}")
    if observed.get("best_arm") != "JOINT_RECALL_COMPONENT_COST_AWARE":
        fail(f"D75 best arm mismatch: {observed.get('best_arm')}")
    if not upstream.get("d75_commit_present", {}).get("present") and not upstream.get("restore", {}).get("restore_attempted"):
        fail("D75 commit missing but restore status was not explicit")

    best = aggregate.get("best_fair_arm", {})
    required = [
        "exact_joint_accuracy",
        "correlated_echo_accuracy",
        "adversarial_distractor_accuracy",
        "external_test_required_accuracy",
        "false_confidence_rate",
        "indistinguishable_abstain_rate",
        "average_total_support_used",
        "counter_support_used",
        "distance_to_concrete_oracle_support",
        "gap_reduction_vs_D73_bound",
        "support_saved_vs_D71",
        "support_saved_vs_D75_reference",
        "joint_counter_recall_on_joint_required_rows",
        "external_recall_on_external_required_rows",
        "wrong_concrete_counter_rate",
        "weak_top1_top2_path_failure_rate",
        "top1_top2_sufficient_false_joint_rate",
        "D68_loss_repair_preservation_rate",
        "routing_failure_rows",
        "min_seed_exact",
        "min_seed_correlated",
        "min_seed_adversarial",
        "min_seed_external",
        "fallback_rows",
        "failed_jobs",
    ]
    absent = [key for key in required if key not in best]
    if absent:
        fail(f"best fair arm missing metrics: {absent}")

    positive_decisions = {"joint_recall_component_scale_confirmed", "joint_recall_component_scale_confirmed_high_cost"}
    if decision.get("decision") in positive_decisions:
        gates = {
            "scaled_D75_component": best.get("arm") == "D75_JOINT_RECALL_COST_AWARE_REPLAY",
            "gap_reduction_vs_D73_bound": best["gap_reduction_vs_D73_bound"] >= 0.1500,
            "average_total_support_used": best["average_total_support_used"] <= 6.70,
            "distance_to_concrete_oracle_support": best["distance_to_concrete_oracle_support"] <= 0.38,
            "exact_joint_accuracy": best["exact_joint_accuracy"] >= 0.9990,
            "correlated_echo_accuracy": best["correlated_echo_accuracy"] >= 0.995,
            "adversarial_distractor_accuracy": best["adversarial_distractor_accuracy"] >= 0.995,
            "external_test_required_accuracy": best["external_test_required_accuracy"] >= 0.995,
            "joint_counter_recall_on_joint_required_rows": best["joint_counter_recall_on_joint_required_rows"] >= 0.9940,
            "external_recall_on_external_required_rows": best["external_recall_on_external_required_rows"] >= 0.9957,
            "wrong_concrete_counter_rate": best["wrong_concrete_counter_rate"] <= 0.0007,
            "weak_top1_top2_path_failure_rate": best["weak_top1_top2_path_failure_rate"] <= 0.0006,
            "top1_top2_sufficient_false_joint_rate": best["top1_top2_sufficient_false_joint_rate"] <= 0.0015,
            "false_confidence_rate": best["false_confidence_rate"] <= 0.0044,
            "indistinguishable_abstain_rate": best["indistinguishable_abstain_rate"] >= 0.9948,
            "D68_loss_repair_preservation_rate": best["D68_loss_repair_preservation_rate"] == 1.0,
            "routing_failure_rows": best["routing_failure_rows"] == 0,
            "fallback_rows": best["fallback_rows"] == 0,
            "failed_jobs": not best["failed_jobs"],
            "min_seed_exact": best["min_seed_exact"] >= 0.997,
            "min_seed_correlated": best["min_seed_correlated"] >= 0.995,
            "min_seed_adversarial": best["min_seed_adversarial"] >= 0.995,
            "min_seed_external": best["min_seed_external"] >= 0.995,
        }
        failed = [key for key, passed in gates.items() if not passed]
        if failed:
            fail(f"D76 positive gates failed: {failed}")

    for name, report in [
        ("joint recall scale", joint),
        ("oracle gap scale", oracle),
        ("support cost frontier", frontier),
        ("D68 repair", repair),
        ("top1/top2 sufficient", top1),
        ("joint required", joint_required),
        ("external recall", external),
        ("safety margin", safety),
        ("truth leak", truth),
        ("rust invocation", rust),
    ]:
        if not report.get("passed") and name not in {"support cost frontier"}:
            fail(f"{name} report did not pass")
    if not frontier.get("passed"):
        fail("support cost frontier report did not pass")

    if truth.get("fair_arms_using_truth_label"):
        fail("fair truth-label leak detected")
    if truth.get("fair_arms_using_support_regime_label"):
        fail("fair support-regime leak detected")
    if truth.get("row_id_lookup_used") or truth.get("python_hash_used"):
        fail("row lookup or Python hash hard gate failed")
    if truth.get("label_echo_fair_oracle_used"):
        fail("label echo fair oracle hard gate failed")
    if not truth.get("oracle_arms_reference_only"):
        fail("oracle/reference arms are not reference-only")
    if not top1.get("D68_cheap_top1_regression_prevented"):
        fail("D68 cheap top1 regression prevention failed")
    if repair.get("D68_loss_repair_preservation_rate") != 1.0:
        fail("D68 loss repair preservation failed")

    print(
        json.dumps(
            {
                "check": "passed",
                "out": str(out),
                "decision": decision,
                "scaled_arm": decision.get("scaled_arm"),
                "best_fair_arm": best.get("arm"),
                "average_total_support_used": best.get("average_total_support_used"),
                "gap_reduction_vs_D73_bound": best.get("gap_reduction_vs_D73_bound"),
                "distance_to_concrete_oracle_support": best.get("distance_to_concrete_oracle_support"),
                "joint_recall": best.get("joint_counter_recall_on_joint_required_rows"),
                "external_recall": best.get("external_recall_on_external_required_rows"),
                "fallback_rows": aggregate.get("fallback_rows"),
                "failed_jobs": aggregate.get("failed_jobs"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

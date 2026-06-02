#!/usr/bin/env python3
"""Artifact checker for D71 support-cost oracle-gap scale confirm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_REPORTS = [
    "d70_upstream_manifest.json",
    "support_cost_scale_report.json",
    "oracle_distance_frontier_report.json",
    "routing_preservation_report.json",
    "d68_loss_repair_preservation_report.json",
    "joint_recall_scale_report.json",
    "external_recall_scale_report.json",
    "safety_margin_watch_report.json",
    "truth_leak_audit_report.json",
    "rust_invocation_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
]

ALLOWED_DECISIONS = {
    "support_cost_oracle_gap_scale_confirmed",
    "oracle_gap_scale_confirmed_safety_margin_watch",
    "oracle_gap_reduction_not_scale_stable",
    "oracle_gap_routing_regression",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fail(message: str) -> None:
    raise SystemExit(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    if not out.exists():
        fail(f"out path missing: {out}")
    missing = [name for name in REQUIRED_REPORTS if not (out / name).exists()]
    if missing:
        fail(f"missing required reports: {missing}")

    decision = load_json(out / "decision.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    upstream = load_json(out / "d70_upstream_manifest.json")
    scale = load_json(out / "support_cost_scale_report.json")
    frontier = load_json(out / "oracle_distance_frontier_report.json")
    routing = load_json(out / "routing_preservation_report.json")
    repair = load_json(out / "d68_loss_repair_preservation_report.json")
    joint = load_json(out / "joint_recall_scale_report.json")
    external = load_json(out / "external_recall_scale_report.json")
    safety = load_json(out / "safety_margin_watch_report.json")
    truth = load_json(out / "truth_leak_audit_report.json")
    rust = load_json(out / "rust_invocation_report.json")

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

    observed = upstream.get("d70_artifacts", {})
    if observed.get("decision") != "support_cost_oracle_gap_reduction_confirmed":
        fail(f"D70 upstream decision mismatch: {observed.get('decision')}")
    if observed.get("next") != "D71_SUPPORT_COST_ORACLE_GAP_SCALE_CONFIRM":
        fail(f"D70 upstream next mismatch: {observed.get('next')}")
    if observed.get("best_fair_arm") != "ORACLE_GAP_TARGETED_ROUTER":
        fail(f"D70 best arm mismatch: {observed.get('best_fair_arm')}")

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
        "support_saved_vs_D69",
        "support_saved_vs_D68R",
        "distance_to_concrete_oracle_support",
        "wrong_concrete_counter_rate",
        "weak_top1_top2_path_failure_rate",
        "joint_counter_recall_on_joint_required_rows",
        "external_recall_on_external_required_rows",
        "d68_loss_repair_preservation_rate",
        "min_seed_exact",
        "min_seed_correlated",
        "min_seed_adversarial",
        "min_seed_external",
    ]
    absent = [key for key in required if key not in best]
    if absent:
        fail(f"best fair arm missing metrics: {absent}")

    if decision.get("decision") == "support_cost_oracle_gap_scale_confirmed":
        gates = {
            "scaled_arm": best.get("arm") == "D70_ORACLE_GAP_TARGETED_REPLAY",
            "exact_joint_accuracy": best["exact_joint_accuracy"] >= 0.9990,
            "correlated_echo_accuracy": best["correlated_echo_accuracy"] >= 0.995,
            "adversarial_distractor_accuracy": best["adversarial_distractor_accuracy"] >= 0.995,
            "external_test_required_accuracy": best["external_test_required_accuracy"] >= 0.995,
            "false_confidence_rate": best["false_confidence_rate"] <= 0.01,
            "indistinguishable_abstain_rate": best["indistinguishable_abstain_rate"] >= 0.99,
            "wrong_concrete_counter_rate": best["wrong_concrete_counter_rate"] <= 0.001,
            "weak_top1_top2_path_failure_rate": best["weak_top1_top2_path_failure_rate"] <= 0.001,
            "joint_counter_recall_on_joint_required_rows": best["joint_counter_recall_on_joint_required_rows"] >= 0.99,
            "external_recall_on_external_required_rows": best["external_recall_on_external_required_rows"] >= 0.995,
            "d68_loss_repair_preservation_rate": best["d68_loss_repair_preservation_rate"] == 1.0,
            "support_saved_vs_D69": best["support_saved_vs_D69"] >= 0.15,
            "distance_to_concrete_oracle_support": best["distance_to_concrete_oracle_support"] <= 0.55,
            "min_seed_exact": best["min_seed_exact"] >= 0.997,
            "fallback_rows": aggregate.get("fallback_rows") == 0,
            "failed_jobs": not aggregate.get("failed_jobs"),
        }
        failed = [key for key, passed in gates.items() if not passed]
        if failed:
            fail(f"D71 positive gates failed: {failed}")

    if not scale.get("stable_vs_d70"):
        fail("D70 replay was not scale stable")
    if frontier.get("remaining_gap", 999) > 0.55:
        fail("oracle frontier gap too high")
    if not routing.get("does_not_repeat_d68_failure"):
        fail("D68 cheap-top1/concrete routing failure repeated")
    if repair.get("d68_loss_repair_preservation_rate") != 1.0:
        fail("D68 loss repair preservation failed")
    if not joint.get("passed"):
        fail("joint recall scale gate failed")
    if not external.get("passed"):
        fail("external recall scale gate failed")
    if not safety.get("passed"):
        fail("safety margin watch failed")
    if truth.get("fair_arms_using_truth_label"):
        fail("fair truth-label leak detected")
    if truth.get("fair_arms_using_support_regime_label"):
        fail("fair support-regime leak detected")
    if truth.get("row_id_lookup_used") or truth.get("python_hash_used"):
        fail("row lookup or Python hash hard gate failed")
    if truth.get("label_echo_fair_oracle_used"):
        fail("label echo fair oracle hard gate failed")
    if not truth.get("oracle_arms_reference_only"):
        fail("oracle arms are not reference-only")

    print(
        json.dumps(
            {
                "check": "passed",
                "out": str(out),
                "decision": decision,
                "scaled_arm": decision.get("scaled_arm"),
                "best_fair_arm": best.get("arm"),
                "support_saved_vs_D69": best.get("support_saved_vs_D69"),
                "distance_to_concrete_oracle_support": best.get("distance_to_concrete_oracle_support"),
                "failed_jobs": aggregate.get("failed_jobs"),
                "fallback_rows": aggregate.get("fallback_rows"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

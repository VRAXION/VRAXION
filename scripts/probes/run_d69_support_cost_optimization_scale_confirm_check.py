#!/usr/bin/env python3
"""Checker for D69 support cost optimization scale confirm."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED_REPORTS = [
    "queue.json",
    "progress.jsonl",
    "d68c_upstream_manifest.json",
    "d68c_bootstrap_report.json",
    "support_cost_scale_report.json",
    "routing_preservation_report.json",
    "d68_loss_repair_preservation_report.json",
    "joint_recall_scale_report.json",
    "external_recall_scale_report.json",
    "safety_regression_report.json",
    "oracle_distance_frontier_report.json",
    "support_cost_frontier_report.json",
    "truth_leak_audit_report.json",
    "rust_invocation_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
]

ALLOWED_DECISIONS = {
    "support_cost_optimization_scale_confirmed",
    "support_cost_scale_confirmed_safety_margin_watch",
    "support_cost_optimization_not_scale_stable",
    "support_cost_optimization_routing_regression",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fail(message: str) -> None:
    print(json.dumps({"check": "failed", "error": message}, indent=2))
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--out", default="target/pilot_wave/d69_support_cost_optimization_scale_confirm/smoke")
    args = parser.parse_args()

    out = Path(args.out)
    if not out.exists():
        fail(f"missing output directory: {out}")
    missing = [name for name in REQUIRED_REPORTS if not (out / name).exists()]
    if missing:
        fail(f"missing required reports: {missing}")

    decision = load_json(out / "decision.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    upstream = load_json(out / "d68c_upstream_manifest.json")
    routing = load_json(out / "routing_preservation_report.json")
    repair = load_json(out / "d68_loss_repair_preservation_report.json")
    joint = load_json(out / "joint_recall_scale_report.json")
    external = load_json(out / "external_recall_scale_report.json")
    safety = load_json(out / "safety_regression_report.json")
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

    scaled = aggregate.get("scaled_arm", {})
    required = [
        "exact_joint_accuracy",
        "correlated_echo_accuracy",
        "adversarial_distractor_accuracy",
        "external_test_required_accuracy",
        "false_confidence_rate",
        "indistinguishable_abstain_rate",
        "average_total_support_used",
        "support_saved_vs_D68R",
        "distance_to_concrete_oracle_support",
        "wrong_concrete_counter_rate",
        "weak_top1_top2_path_failure_rate",
        "joint_counter_recall_on_joint_required_rows",
        "d68_loss_repair_preservation_rate",
        "min_seed_exact",
    ]
    absent = [key for key in required if key not in scaled]
    if absent:
        fail(f"scaled arm missing metrics: {absent}")

    if decision.get("decision") in {
        "support_cost_optimization_scale_confirmed",
        "support_cost_scale_confirmed_safety_margin_watch",
    }:
        gates = {
            "exact_joint_accuracy": scaled["exact_joint_accuracy"] >= 0.9990,
            "correlated_echo_accuracy": scaled["correlated_echo_accuracy"] >= 0.995,
            "adversarial_distractor_accuracy": scaled["adversarial_distractor_accuracy"] >= 0.995,
            "external_test_required_accuracy": scaled["external_test_required_accuracy"] >= 0.995,
            "false_confidence_rate": scaled["false_confidence_rate"] <= 0.01,
            "indistinguishable_abstain_rate": scaled["indistinguishable_abstain_rate"] >= 0.99,
            "wrong_concrete_counter_rate": scaled["wrong_concrete_counter_rate"] <= 0.001,
            "weak_top1_top2_path_failure_rate": scaled["weak_top1_top2_path_failure_rate"] <= 0.001,
            "joint_counter_recall_on_joint_required_rows": scaled["joint_counter_recall_on_joint_required_rows"] >= 0.99,
            "d68_loss_repair_preservation_rate": scaled["d68_loss_repair_preservation_rate"] == 1.0,
            "support_saved_vs_D68R": scaled["support_saved_vs_D68R"] >= 0.30,
            "distance_to_concrete_oracle_support": scaled["distance_to_concrete_oracle_support"] <= 0.80,
            "min_seed_exact": scaled["min_seed_exact"] >= 0.997,
        }
        failed = [key for key, passed in gates.items() if not passed]
        if failed:
            fail(f"scaled gate failed: {failed}")

    observed_upstream = upstream.get("d68c_artifacts", {})
    if observed_upstream.get("decision") != "support_cost_optimization_confirmed":
        fail(f"D68C upstream decision mismatch: {observed_upstream.get('decision')}")
    if observed_upstream.get("next") != "D69_SUPPORT_COST_OPTIMIZATION_SCALE_CONFIRM":
        fail(f"D68C upstream next mismatch: {observed_upstream.get('next')}")
    if not routing.get("preserved"):
        fail("routing preservation failed")
    if repair.get("d68_loss_repair_preservation_rate") != 1.0:
        fail("D68 loss repair preservation failed")
    if not joint.get("passed"):
        fail("joint recall scale gate failed")
    if not external.get("passed"):
        fail("external recall scale gate failed")
    if not safety.get("passed_gate"):
        fail("safety gate failed")
    if truth.get("fair_arms_using_truth_label"):
        fail("fair truth-label leak detected")
    if truth.get("fair_arms_using_support_regime_label"):
        fail("fair support-regime leak detected")
    if truth.get("row_id_lookup_used") or truth.get("python_hash_used"):
        fail("row lookup or Python hash hard gate failed")

    print(
        json.dumps(
            {
                "check": "passed",
                "out": str(out),
                "decision": decision,
                "scale_mode": aggregate.get("scale_mode"),
                "support_saved_vs_D68R": scaled.get("support_saved_vs_D68R"),
                "failed_jobs": aggregate.get("failed_jobs"),
                "fallback_rows": aggregate.get("fallback_rows"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

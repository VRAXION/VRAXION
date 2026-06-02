#!/usr/bin/env python3
"""Checker for D68C support cost optimization artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED_REPORTS = [
    "queue.json",
    "progress.jsonl",
    "d68r_upstream_manifest.json",
    "artifact_restore_report.json",
    "support_cost_optimization_report.json",
    "concrete_action_routing_preservation_report.json",
    "d68_loss_repair_preservation_report.json",
    "top1_top2_vs_joint_routing_report.json",
    "joint_counter_recall_report.json",
    "external_escalation_report.json",
    "support_over_cheapest_report.json",
    "oracle_distance_frontier_report.json",
    "action_confusion_matrix_report.json",
    "clean_mixed_cost_saving_report.json",
    "hard_regime_recall_report.json",
    "safety_report.json",
    "truth_leak_audit_report.json",
    "rust_invocation_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
]

ALLOWED_DECISIONS = {
    "support_cost_optimization_confirmed",
    "counter_action_routing_stable_high_cost",
    "support_cost_optimization_recall_failure",
    "support_cost_optimization_safety_failure",
    "support_cost_optimization_not_confirmed",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fail(message: str) -> None:
    print(json.dumps({"check": "failed", "error": message}, indent=2))
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--out", default="target/pilot_wave/d68c_support_cost_optimization/smoke")
    args = parser.parse_args()

    out = Path(args.out)
    if not out.exists():
        fail(f"missing output directory: {out}")
    missing = [name for name in REQUIRED_REPORTS if not (out / name).exists()]
    if missing:
        fail(f"missing required reports: {missing}")

    decision = load_json(out / "decision.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    routing = load_json(out / "concrete_action_routing_preservation_report.json")
    repair = load_json(out / "d68_loss_repair_preservation_report.json")
    joint = load_json(out / "joint_counter_recall_report.json")
    external = load_json(out / "external_escalation_report.json")
    safety = load_json(out / "safety_report.json")
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

    best = aggregate.get("best_fair_arm", {})
    required_keys = [
        "exact_joint_accuracy",
        "correlated_echo_accuracy",
        "adversarial_distractor_accuracy",
        "external_test_required_accuracy",
        "average_total_support_used",
        "support_saved_vs_D68R",
        "wrong_concrete_counter_rate",
        "weak_top1_top2_path_failure_rate",
        "joint_counter_recall_on_joint_required_rows",
        "d68_loss_repair_preservation_rate",
    ]
    absent = [key for key in required_keys if key not in best]
    if absent:
        fail(f"best fair arm missing metrics: {absent}")

    if decision.get("decision") == "support_cost_optimization_confirmed":
        checks = {
            "exact_joint_accuracy": best["exact_joint_accuracy"] >= 0.9990,
            "correlated_echo_accuracy": best["correlated_echo_accuracy"] >= 0.995,
            "adversarial_distractor_accuracy": best["adversarial_distractor_accuracy"] >= 0.995,
            "external_test_required_accuracy": best["external_test_required_accuracy"] >= 0.995,
            "false_confidence_rate": best["false_confidence_rate"] <= 0.01,
            "indistinguishable_abstain_rate": best["indistinguishable_abstain_rate"] >= 0.99,
            "wrong_concrete_counter_rate": best["wrong_concrete_counter_rate"] <= 0.001,
            "weak_top1_top2_path_failure_rate": best["weak_top1_top2_path_failure_rate"] <= 0.001,
            "joint_counter_recall_on_joint_required_rows": best["joint_counter_recall_on_joint_required_rows"] >= 0.99,
            "d68_loss_repair_preservation_rate": best["d68_loss_repair_preservation_rate"] == 1.0,
            "average_total_support_used": best["average_total_support_used"] < 7.6795,
            "support_saved_vs_D68R": best["support_saved_vs_D68R"] >= 0.30,
        }
        failed = [key for key, passed in checks.items() if not passed]
        if failed:
            fail(f"positive decision gates failed: {failed}")

    if truth.get("fair_arms_using_truth_label"):
        fail("fair arm truth-label leak detected")
    if truth.get("fair_arms_using_support_regime_label"):
        fail("fair arm support-regime leak detected")
    if truth.get("row_id_lookup_used"):
        fail("row id lookup used")
    if truth.get("python_hash_used"):
        fail("Python hash used")
    if not routing.get("preserved_by_best_arm"):
        fail("D68 concrete routing repair was not preserved")
    if repair.get("d68_loss_repair_preservation_rate") != 1.0:
        fail("D68 loss repair preservation failed")
    if not joint.get("passed"):
        fail("joint recall gate failed")
    if external.get("external_test_required_accuracy", 0.0) < 0.995:
        fail("external escalation gate failed")
    if not safety.get("passed"):
        fail("safety gate failed")

    print(
        json.dumps(
            {
                "check": "passed",
                "out": str(out),
                "decision": decision,
                "best_fair_arm": best.get("arm"),
                "support_saved_vs_D68R": best.get("support_saved_vs_D68R"),
                "failed_jobs": aggregate.get("failed_jobs"),
                "fallback_rows": aggregate.get("fallback_rows"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

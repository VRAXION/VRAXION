#!/usr/bin/env python3
"""Checker for D68R concrete counter-action routing repair."""

import argparse
import json
import sys
from pathlib import Path

REQUIRED_REPORTS = [
    "d68a_upstream_manifest.json",
    "concrete_router_training_report.json",
    "concrete_action_routing_report.json",
    "top1_vs_joint_counter_report.json",
    "support_cost_frontier_report.json",
    "d68r_harm_repair_report.json",
    "control_report.json",
    "regime_breakdown_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
]

ALLOWED_DECISIONS = {
    "concrete_counter_action_routing_repair_positive",
    "concrete_counter_action_routing_repair_positive_high_cost",
    "concrete_counter_action_routing_repair_partial",
    "concrete_counter_action_routing_repair_not_confirmed",
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fail(message):
    print(json.dumps({"check": "failed", "error": message}, indent=2))
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--out", default="target/pilot_wave/d68r_concrete_counter_action_routing_repair/smoke")
    args = parser.parse_args()

    out = Path(args.out)
    if not out.exists():
        fail(f"missing output directory: {out}")

    missing = [name for name in REQUIRED_REPORTS if not (out / name).exists()]
    if missing:
        fail(f"missing required reports: {missing}")

    decision = load_json(out / "decision.json")
    summary = load_json(out / "summary.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    routing = load_json(out / "concrete_action_routing_report.json")
    training = load_json(out / "concrete_router_training_report.json")
    control = load_json(out / "control_report.json")

    if decision.get("decision") not in ALLOWED_DECISIONS:
        fail(f"unexpected decision: {decision.get('decision')}")
    if aggregate.get("failed_jobs"):
        fail(f"failed jobs present: {aggregate.get('failed_jobs')}")
    if summary.get("fallback_rows") != 0:
        fail(f"fallback rows not zero: {summary.get('fallback_rows')}")
    if not aggregate.get("rust_path_invoked"):
        fail("rust path was not invoked")
    if not training.get("selected_accuracy_first"):
        fail("missing selected accuracy-first router config")

    fair = routing.get("D68R_CONCRETE_ROUTER")
    if not fair:
        fail("missing D68R_CONCRETE_ROUTER routing metrics")
    for key in [
        "exact_joint_accuracy",
        "average_total_support_used",
        "wrong_concrete_counter_rate",
        "weak_top1_top2_path_failure_rate",
        "concrete_selected_counter_missed_rate",
    ]:
        if key not in fair:
            fail(f"missing D68R_CONCRETE_ROUTER.{key}")

    if control.get("fair_arms_using_truth_label"):
        fail("fair arms used truth labels")
    if control.get("fair_arms_using_regime_label"):
        fail("fair arms used regime labels")

    print(
        json.dumps(
            {
                "check": "passed",
                "out": str(out),
                "decision": decision,
                "summary": {
                    "rust_path_invoked": aggregate.get("rust_path_invoked"),
                    "rust_aggregation_rows": aggregate.get("rust_aggregation_rows"),
                    "rust_controller_rows": aggregate.get("rust_controller_rows"),
                    "fallback_rows": summary.get("fallback_rows"),
                    "failed_jobs": aggregate.get("failed_jobs"),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

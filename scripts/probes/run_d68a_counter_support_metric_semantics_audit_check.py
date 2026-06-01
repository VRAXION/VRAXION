#!/usr/bin/env python3
"""Checker for D68A counter-support metric semantics audit."""

import argparse
import json
import sys
from pathlib import Path

REQUIRED_REPORTS = [
    "d68_upstream_manifest.json",
    "artifact_completeness_and_rebuild_parity_report.json",
    "counter_metric_definition_report.json",
    "concrete_counter_action_report.json",
    "causal_counter_removal_report.json",
    "cheapest_correct_support_report.json",
    "d68_harm_classification_report.json",
    "diagnostic_margin_stability_report.json",
    "support_accounting_report.json",
    "regime_blind_audit_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
]

ALLOWED_DECISIONS = {
    "counter_support_metrics_confirmed",
    "counter_metrics_valid_but_need_rename",
    "counter_support_metric_pipeline_not_confirmed",
    "d68a_artifact_insufficient_for_metric_audit",
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fail(msg):
    print(json.dumps({"check": "failed", "error": msg}, indent=2))
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--out", default="target/pilot_wave/d68a_counter_support_metric_semantics_audit/smoke")
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
    concrete = load_json(out / "concrete_counter_action_report.json")
    artifact = load_json(out / "artifact_completeness_and_rebuild_parity_report.json")
    regime_blind = load_json(out / "regime_blind_audit_report.json")

    if decision.get("decision") not in ALLOWED_DECISIONS:
        fail(f"unexpected decision: {decision.get('decision')}")
    if aggregate.get("failed_jobs"):
        fail(f"failed jobs present: {aggregate.get('failed_jobs')}")
    if summary.get("fallback_rows") != 0:
        fail(f"fallback rows not zero: {summary.get('fallback_rows')}")
    if not aggregate.get("rust_path_invoked"):
        fail("rust path was not invoked")
    if not artifact.get("rebuild_performed"):
        fail("D68A did not rebuild concrete action alternatives")
    if regime_blind.get("fair_arms_using_truth_label"):
        fail("fair arms used truth labels")
    if regime_blind.get("fair_arms_using_regime_label"):
        fail("fair arms used regime labels")

    for arm in ["D67_BEST_REPLAY", "D68_TRAINED_THRESHOLD_REPLAY"]:
        if arm not in concrete:
            fail(f"missing concrete audit for {arm}")
        required_keys = [
            "reported_unnecessary_counter_request_rate",
            "causal_unnecessary_counter_support_rate",
            "concrete_selected_counter_missed_rate",
            "wrong_concrete_counter_rate",
            "support_over_cheapest_effective_mean",
        ]
        for key in required_keys:
            if key not in concrete[arm]:
                fail(f"missing {arm}.{key}")

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

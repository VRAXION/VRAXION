#!/usr/bin/env python3
"""Checker for D68 counter-support triage repair."""

import argparse
import json
from pathlib import Path

REQUIRED = [
    "queue.json",
    "progress.jsonl",
    "compute_probe.json",
    "d67_upstream_manifest.json",
    "dataset_manifest.json",
    "row_outputs_test.jsonl",
    "row_outputs_ood.jsonl",
    "triage_summary_report.json",
    "trained_threshold_triage_report.json",
    "support_scoring_report.json",
    "support_triage_report.json",
    "counter_support_triage_report.json",
    "counter_precision_recall_report.json",
    "external_test_triage_report.json",
    "support_cost_frontier_report.json",
    "fixed_budget_sweep_report.json",
    "clean_unnecessary_counter_audit_report.json",
    "mixed_unnecessary_counter_audit_report.json",
    "unnecessary_counter_support_report.json",
    "missed_counter_support_report.json",
    "regime_breakdown_report.json",
    "ood_cost_frontier_report.json",
    "support_budget_report.json",
    "ablation_and_control_report.json",
    "content_corruption_report.json",
    "truth_leak_audit_report.json",
    "rust_invocation_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
]

ALLOWED_DECISIONS = {
    "counter_support_triage_repair_confirmed",
    "counter_triage_high_recall_high_cost",
    "counter_triage_recall_failure",
    "counter_support_triage_repair_not_confirmed",
}

ROW_FIELDS = {
    "row_id",
    "seed",
    "split",
    "arm",
    "support_regime",
    "selected_action",
    "gate_selected_policy",
    "gate_basis",
    "rust_aggregation_used",
    "rust_aggregation_input_is_support_set",
    "python_precomputed_final_aggregate_label_used",
    "support_scoring_used",
    "support_budget_cap",
    "cost_adjusted_accuracy",
    "unnecessary_counter_support",
    "unnecessary_internal_counter",
    "unnecessary_external_test",
    "missed_counter_support",
    "missed_internal_counter",
    "missed_external_test",
    "counter_needed",
    "external_test_needed",
    "counter_requested",
    "external_test_requested",
    "counter_true_positive",
    "counter_false_positive",
    "counter_false_negative",
    "external_test_true_positive",
    "external_test_false_positive",
    "external_test_false_negative",
    "support_over_cheapest_correct",
    "rust_network_path_invoked",
    "python_fallback_used",
    "fair_arm",
    "control_arm",
    "reference_only_arm",
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def first_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/d68_counter_support_triage_repair/smoke")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    errors = []
    for name in REQUIRED:
        if not (out / name).exists():
            errors.append(f"missing required artifact: {name}")
    if errors:
        raise SystemExit("\n".join(errors))

    decision = load_json(out / "decision.json")
    summary = load_json(out / "summary.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    dataset = load_json(out / "dataset_manifest.json")
    truth = load_json(out / "truth_leak_audit_report.json")
    support_cost = load_json(out / "support_cost_frontier_report.json")
    precision_recall = load_json(out / "counter_precision_recall_report.json")

    if decision.get("decision") not in ALLOWED_DECISIONS:
        errors.append(f"unexpected decision: {decision.get('decision')}")
    if aggregate.get("failed_jobs"):
        errors.append(f"failed_jobs not empty: {aggregate.get('failed_jobs')}")
    if not aggregate.get("rust_path_invoked"):
        errors.append("rust_path_invoked is false")
    if aggregate.get("fallback_rows") != 0:
        errors.append(f"fallback_rows must be 0, got {aggregate.get('fallback_rows')}")
    if aggregate.get("python_precomputed_final_aggregate_label_rows") != 0:
        errors.append("Python precomputed final aggregate labels were used")
    if not dataset.get("healthy_milestone_not_micro"):
        errors.append("dataset manifest does not mark healthy milestone run")
    if not dataset.get("scale_confirm_milestone"):
        errors.append("dataset manifest does not mark scale confirm milestone")
    if not dataset.get("unnecessary_counter_frontier_visible"):
        errors.append("dataset manifest does not mark unnecessary counter frontier")
    if not dataset.get("counter_triage_metric_definitions", {}).get("external_test_measured_separately"):
        errors.append("dataset manifest does not define separated external-test metrics")
    if not dataset.get("rust_arms_receive_support_evidence_set_representation"):
        errors.append("Rust support/evidence set input not declared")
    if truth.get("fair_arms_using_truth_label"):
        errors.append("truth leak audit found fair arms using truth label")
    if truth.get("fair_arms_using_forbidden_metadata"):
        errors.append(f"truth leak audit found forbidden metadata usage: {truth.get('fair_arms_using_forbidden_metadata')}")
    if truth.get("python_precomputed_final_aggregate_label_used_by_fair_arms"):
        errors.append("truth leak audit found precomputed aggregate label usage")

    for arm in [
        "D67_BEST_REPLAY",
        "COUNTER_TRIAGE_MULTI_SIGNAL_GATE",
        "TRAINED_THRESHOLD_TRIAGE_GATE",
        "COUNTER_TRIAGE_CONSERVATIVE_HIGH_RECALL",
        "COUNTER_TRIAGE_COST_OPTIMIZED",
        "CAP_7_CONTROL",
        "CAP_9_CONTROL",
        "AGGREGATION_ABLATION_CONTROL",
        "RANDOM_COUNTER_CONTROL",
        "SHUFFLED_TRIAGE_SIGNAL_CONTROL",
        "BAD_TRIAGE_SIGNAL_CONTROL",
    ]:
        if arm not in support_cost:
            errors.append(f"support cost report missing arm: {arm}")
        if arm not in precision_recall:
            errors.append(f"precision/recall report missing arm: {arm}")

    row = first_jsonl(out / "row_outputs_test.jsonl")
    if not row:
        errors.append("row_outputs_test.jsonl is empty")
    else:
        missing = sorted(ROW_FIELDS - set(row))
        if missing:
            errors.append(f"row output missing fields: {missing}")
        if row.get("python_precomputed_final_aggregate_label_used"):
            errors.append("row says Python precomputed final aggregate label was used")
        if row.get("fair_arm") and row.get("gate_used_truth_label"):
            errors.append("fair row says gate used truth label")

    if errors:
        raise SystemExit("\n".join(errors))
    print(
        json.dumps(
            {
                "check": "passed",
                "out": str(out),
                "decision": decision,
                "summary": {
                    "rust_path_invoked": summary.get("rust_path_invoked"),
                    "rust_aggregation_rows": summary.get("rust_aggregation_rows"),
                    "fallback_rows": summary.get("fallback_rows"),
                    "failed_jobs": summary.get("failed_jobs"),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

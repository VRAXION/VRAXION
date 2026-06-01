#!/usr/bin/env python3
"""Checker for D65 set-invariant IPF aggregation prototype."""

import argparse
import json
from pathlib import Path

REQUIRED = [
    "queue.json",
    "progress.jsonl",
    "compute_probe.json",
    "d64u_upstream_manifest.json",
    "dataset_manifest.json",
    "row_outputs_test.jsonl",
    "row_outputs_ood.jsonl",
    "set_aggregation_definition_report.json",
    "order_invariance_report.json",
    "score_shape_aggregation_report.json",
    "rust_aggregation_mapping_report.json",
    "aggregation_quality_report.json",
    "controller_with_set_aggregation_report.json",
    "support_content_corruption_report.json",
    "truth_leak_audit_report.json",
    "rust_invocation_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
]

ALLOWED_DECISIONS = {
    "set_invariant_ipf_aggregation_confirmed",
    "hybrid_set_aggregation_positive",
    "order_dependence_reappeared",
    "set_invariant_ipf_aggregation_not_confirmed",
}

ROW_FIELDS = {
    "row_id",
    "seed",
    "split",
    "arm",
    "support_regime",
    "selected_action",
    "gate_selected_policy",
    "rust_aggregation_used",
    "rust_aggregation_input_is_support_set",
    "python_precomputed_final_aggregate_label_used",
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
    parser.add_argument("--out", default="target/pilot_wave/d65_set_invariant_ipf_aggregation_prototype/smoke")
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
    definition = load_json(out / "set_aggregation_definition_report.json")
    truth = load_json(out / "truth_leak_audit_report.json")

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
    if not dataset.get("rust_arms_receive_support_evidence_set_representation"):
        errors.append("dataset manifest does not confirm support/evidence set input to Rust arms")
    if dataset.get("python_precomputed_final_aggregate_label_used"):
        errors.append("dataset manifest says Python final aggregate label was used")
    if "Python-precomputed final aggregate labels" not in json.dumps(definition):
        errors.append("definition report does not document forbidden precomputed aggregate labels")
    if truth.get("fair_arms_using_truth_label"):
        errors.append("truth leak audit found fair arms using truth label")
    if truth.get("python_precomputed_final_aggregate_label_used_by_fair_arms"):
        errors.append("truth leak audit found precomputed aggregate label usage")

    row = first_jsonl(out / "row_outputs_test.jsonl")
    if not row:
        errors.append("row_outputs_test.jsonl is empty")
    else:
        missing = sorted(ROW_FIELDS - set(row))
        if missing:
            errors.append(f"row output missing fields: {missing}")
        if row.get("python_precomputed_final_aggregate_label_used"):
            errors.append("row says Python precomputed final aggregate label was used")

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

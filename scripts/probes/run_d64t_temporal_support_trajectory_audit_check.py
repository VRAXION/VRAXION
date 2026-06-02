#!/usr/bin/env python3
"""Checker for D64T temporal/support trajectory audit."""

import argparse
import json
from pathlib import Path

REQUIRED = [
    "queue.json",
    "progress.jsonl",
    "compute_probe.json",
    "d64s_upstream_manifest.json",
    "dataset_manifest.json",
    "row_outputs_test.jsonl",
    "row_outputs_ood.jsonl",
    "noncommutativity_report.json",
    "trajectory_vs_set_report.json",
    "stage_preserving_shuffle_report.json",
    "stage_destroying_shuffle_report.json",
    "order_artifact_report.json",
    "early_vs_late_counter_report.json",
    "open_vs_closed_loop_report.json",
    "final_state_vs_trajectory_readout_report.json",
    "truth_leak_audit_report.json",
    "rust_invocation_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
]

ALLOWED_DECISIONS = {
    "support_trajectory_signal_confirmed",
    "support_order_not_required_set_aggregation_sufficient",
    "arbitrary_order_artifact_detected",
    "temporal_interference_claim_not_confirmed",
    "d64t_instrumentation_incomplete",
}

ROW_FIELDS = {
    "row_id",
    "seed",
    "split",
    "arm",
    "support_regime",
    "trajectory_variant",
    "selected_action",
    "gate_selected_policy",
    "correct",
    "exact_joint_correct",
    "stage_sequence",
    "order_disagreement_norm",
    "path_flip_norm",
    "raw_support_sequence_manipulated_before_feature_generation",
    "diagnostic_bits_shuffled_after_feature_generation",
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
    parser.add_argument("--out", default="target/pilot_wave/d64t_temporal_support_trajectory_audit/smoke")
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
    aggregate = load_json(out / "aggregate_metrics.json")
    dataset = load_json(out / "dataset_manifest.json")
    summary = load_json(out / "summary.json")
    if decision.get("decision") not in ALLOWED_DECISIONS:
        errors.append(f"unexpected decision: {decision.get('decision')}")
    if aggregate.get("failed_jobs"):
        errors.append(f"failed_jobs not empty: {aggregate.get('failed_jobs')}")
    if not aggregate.get("rust_path_invoked"):
        errors.append("rust_path_invoked is false")
    if aggregate.get("fallback_rows") != 0:
        errors.append(f"fallback_rows must be 0, got {aggregate.get('fallback_rows')}")
    if not aggregate.get("raw_support_sequence_manipulated_before_feature_generation"):
        errors.append("raw support sequence manipulation flag is false")
    if aggregate.get("diagnostic_bits_shuffled_after_feature_generation"):
        errors.append("diagnostic bit shuffle flag must be false")
    if not dataset.get("truth_hidden_from_controller_inputs"):
        errors.append("truth_hidden_from_controller_inputs flag is false")

    row = first_jsonl(out / "row_outputs_test.jsonl")
    if not row:
        errors.append("row_outputs_test.jsonl is empty")
    else:
        missing = sorted(ROW_FIELDS - set(row))
        if missing:
            errors.append(f"row output missing fields: {missing}")
        if not row.get("raw_support_sequence_manipulated_before_feature_generation"):
            errors.append("row does not confirm raw support manipulation before feature generation")
        if row.get("diagnostic_bits_shuffled_after_feature_generation"):
            errors.append("row incorrectly says diagnostic bits were shuffled after feature generation")

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

#!/usr/bin/env python3
"""Artifact checker for D44F soft-field directionality probe."""
import argparse
import json
from pathlib import Path

REQUIRED = [
    "d44e_upstream_manifest.json",
    "dataset_manifest.json",
    "soft_field_geometry_report.json",
    "directional_alignment_report.json",
    "monotonic_support_update_report.json",
    "vector_additivity_report.json",
    "ambiguity_prediction_report.json",
    "negative_elimination_vector_report.json",
    "shuffled_field_control_report.json",
    "random_projection_control_report.json",
    "adversarial_collision_stress_report.json",
    "primitive_space_breakdown_report.json",
    "support_count_breakdown_report.json",
    "failure_taxonomy_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
]
REQUIRED_METRICS = [
    "directional_alignment_accuracy",
    "true_family_mean_rank",
    "true_equivalence_mean_rank",
    "entropy_drop_1_to_5",
    "margin_gain_1_to_5",
    "monotonic_alignment_rate",
    "vector_additivity_error",
    "ambiguity_prediction_auc_or_accuracy",
    "eliminated_correct_candidate_rate",
    "shuffled_field_accuracy",
    "random_projection_accuracy",
    "collision_stress_accuracy",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    missing = [name for name in REQUIRED if not (out / name).exists()]
    if missing:
        raise SystemExit(f"missing artifacts: {missing}")
    aggregate = json.loads((out / "aggregate_metrics.json").read_text())
    missing_metrics = [name for name in REQUIRED_METRICS if name not in aggregate]
    if missing_metrics:
        raise SystemExit(f"missing aggregate metrics: {missing_metrics}")
    directional = json.loads((out / "directional_alignment_report.json").read_text())
    for space in ("CURRENT5", "ALL28_UNORDERED", "ORDERED56_CONTROL"):
        if space not in directional:
            raise SystemExit(f"missing directional space: {space}")
        for count in ("1", "2", "3", "4", "5"):
            if count not in directional[space]["by_support_count"]:
                raise SystemExit(f"missing support count {count} for {space}")
    failure = json.loads((out / "failure_taxonomy_report.json").read_text())
    if not failure.get("breaks_or_weakens"):
        raise SystemExit("failure taxonomy must state where soft-field directionality breaks or weakens")
    decision = json.loads((out / "decision.json").read_text())
    print(json.dumps({"status": "ok", "decision": decision["decision"], "next": decision["next"]}, indent=2))


if __name__ == "__main__":
    main()

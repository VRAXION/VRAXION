#!/usr/bin/env python3
"""Artifact checker for D44G IPF breakpoint stress map."""
import argparse
import json
from pathlib import Path

REQUIRED = [
    "upstream_manifest.json",
    "dataset_manifest.json",
    "stress_axis_summary_report.json",
    "candidate_space_size_report.json",
    "aliasing_stress_report.json",
    "correlated_noise_support_report.json",
    "adversarial_distractor_report.json",
    "support_budget_limit_report.json",
    "factorisation_removal_report.json",
    "operator_space_expansion_light_report.json",
    "ipf_predictive_power_report.json",
    "ipf_causal_perturbation_report.json",
    "breakpoint_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
]
AXES = [
    "CANDIDATE_SPACE_SIZE",
    "ALIASING_STRESS",
    "CORRELATED_NOISE_SUPPORT",
    "ADVERSARIAL_DISTRACTORS",
    "SUPPORT_BUDGET_LIMIT",
    "FACTORISATION_REMOVAL",
    "OPERATOR_SPACE_EXPANSION_LIGHT",
]
CONTROLS = ["shuffled_field_accuracy", "random_projection_accuracy", "candidate_order_shuffle_accuracy"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    missing = [name for name in REQUIRED if not (out / name).exists()]
    if missing:
        raise SystemExit(f"missing artifacts: {missing}")
    summary = json.loads((out / "stress_axis_summary_report.json").read_text())
    missing_axes = [axis for axis in AXES if axis not in summary]
    if missing_axes:
        raise SystemExit(f"missing stress axes: {missing_axes}")
    aggregate = json.loads((out / "aggregate_metrics.json").read_text())
    missing_controls = [control for control in CONTROLS if control not in aggregate]
    if missing_controls:
        raise SystemExit(f"missing controls: {missing_controls}")
    support = json.loads((out / "support_budget_limit_report.json").read_text())
    for count in ("1", "2", "3", "4", "5"):
        if count not in support:
            raise SystemExit(f"missing support budget {count}")
    candidate = json.loads((out / "candidate_space_size_report.json").read_text())
    sample = candidate.get("ALL28", {})
    for key in ("candidate_accuracy", "family_accuracy", "equivalence_accuracy"):
        if key not in sample:
            raise SystemExit(f"candidate/family/equivalence metric missing: {key}")
    breakpoint = json.loads((out / "breakpoint_report.json").read_text())
    if not breakpoint.get("named_breakpoints"):
        raise SystemExit("breakpoints must be named")
    decision = json.loads((out / "decision.json").read_text())
    print(json.dumps({"status": "ok", "decision": decision["decision"], "next": decision["next"]}, indent=2))


if __name__ == "__main__":
    main()

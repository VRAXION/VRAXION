#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

REQUIRED = [
    "d44d_upstream_manifest.json",
    "d44d_inconsistency_audit.json",
    "canonical_evaluator_report.json",
    "fixed_support_count_report.json",
    "support4_audit_report.json",
    "staged_policy_comparison_report.json",
    "oracle_minimal_support_report.json",
    "primitive_space_current5_repaired_report.json",
    "primitive_space_all28_repaired_report.json",
    "primitive_space_ordered_pair_control_report.json",
    "primitive_space_distractor_sweep_report.json",
    "primitive_space_collision_report.json",
    "family_vs_candidate_accuracy_report.json",
    "candidate_order_sensitivity_report.json",
    "support_policy_recommendation_report.json",
    "primitive_space_recommendation_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
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
    fixed = json.loads((out / "fixed_support_count_report.json").read_text())
    current5 = json.loads((out / "primitive_space_current5_repaired_report.json").read_text())
    if abs(current5["support_upper_bounds"]["5"]["family_level"] - fixed["5"]["test_accuracy"]) > 1e-12:
        raise SystemExit("current5 still contradicts fixed support 5")
    decision = json.loads((out / "decision.json").read_text())
    print(json.dumps({"status": "ok", "decision": decision["decision"], "next": decision["next"]}, indent=2))


if __name__ == "__main__":
    main()

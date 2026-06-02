#!/usr/bin/env python3
"""Artifact checker for D46 robust support policy scale confirm."""

import argparse
import json
from pathlib import Path

REQUIRED = [
    "queue.json",
    "progress.jsonl",
    "d45b_upstream_manifest.json",
    "dataset_manifest.json",
    "policy_comparison_report.json",
    "support_cost_report.json",
    "control_report.json",
    "seed_variance_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "row_outputs_test.jsonl",
    "row_outputs_ood.jsonl",
]


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def load(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    missing = [name for name in REQUIRED if not (out / name).exists()]
    require(not missing, f"missing artifacts: {missing}")
    aggregate = load(out / "aggregate_metrics.json")
    decision = load(out / "decision.json")
    robust = aggregate["primary_policy_metrics"]["FULL_ROBUST_COMBINED_REPLAY"]
    naive = aggregate["primary_policy_metrics"]["NAIVE_IPF_BASELINE"]
    require(robust["clean_accuracy"] >= 0.995, "clean below scale gate")
    require(robust["correlated_accuracy"] >= 0.95, "correlated below scale gate")
    require(robust["adversarial_accuracy"] >= 0.95, "adversarial below scale gate")
    require(robust["mixed_accuracy"] >= 0.95, "mixed below scale gate")
    require(robust["min_seed_correlated"] >= 0.90, "min seed correlated below scale gate")
    require(robust["min_seed_adversarial"] >= 0.90, "min seed adversarial below scale gate")
    require(naive["clean_accuracy"] - robust["clean_accuracy"] <= 0.005, "clean regression too high")
    require(aggregate["controls_worse"], "controls not worse")
    require(aggregate["failed_jobs"] == [], "failed jobs not empty")
    require(aggregate["support_cost_reported"], "support cost missing")
    require(decision["decision"] in {"robust_support_policy_scale_confirmed", "robust_support_scale_confirmed_high_cost"}, f"unexpected decision {decision['decision']}")
    print(json.dumps({"status": "ok", "decision": decision["decision"], "next": decision["next"]}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Checker for D64U claim repair."""

import argparse
import json
from pathlib import Path

REQUIRED = [
    "queue.json",
    "progress.jsonl",
    "compute_probe.json",
    "d64_upstream_manifest.json",
    "d64b_upstream_manifest.json",
    "d64s_upstream_manifest.json",
    "d64t_upstream_manifest.json",
    "claim_boundary_report.json",
    "supported_claims_report.json",
    "rejected_or_unconfirmed_claims_report.json",
    "d65_set_aggregation_plan.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
]


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/d64u_redefine_ipf_diagnostic_claim_and_set_aggregation_plan/smoke")
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
    supported = load_json(out / "supported_claims_report.json")["claims"]
    rejected = load_json(out / "rejected_or_unconfirmed_claims_report.json")["claims"]
    d65_plan = load_json(out / "d65_set_aggregation_plan.json")
    boundary = load_json(out / "claim_boundary_report.json")

    if decision.get("decision") != "ipf_diagnostic_claim_redefined_set_aggregation_ready":
        errors.append(f"unexpected/non-positive decision: {decision.get('decision')}")
    if summary.get("failed_jobs"):
        errors.append(f"failed_jobs not empty: {summary.get('failed_jobs')}")
    if aggregate.get("failed_jobs"):
        errors.append(f"aggregate failed_jobs not empty: {aggregate.get('failed_jobs')}")
    if len(supported) < 5:
        errors.append("supported claims report is too small")
    if len(rejected) < 5:
        errors.append("rejected/unconfirmed claims report is too small")
    rejected_text = "\n".join(item["claim"] for item in rejected).lower()
    supported_text = "\n".join(item["claim"] for item in supported).lower()
    if "temporal" not in rejected_text:
        errors.append("temporal/order rejected claim missing")
    if "set-invariant" not in supported_text and "set-invariant" not in json.dumps(d65_plan).lower():
        errors.append("set-invariant D65 direction missing")
    if d65_plan.get("next_task") != "D65_SET_INVARIANT_IPF_AGGREGATION_PROTOTYPE":
        errors.append(f"unexpected D65 next task: {d65_plan.get('next_task')}")
    for key, value in boundary.get("upstream_decisions", {}).items():
        if not value.get("decision_matches_expected"):
            errors.append(f"{key} upstream decision does not match expected")
        if value.get("missing_reports"):
            errors.append(f"{key} upstream missing reports: {value.get('missing_reports')}")

    if errors:
        raise SystemExit("\n".join(errors))
    print(
        json.dumps(
            {
                "check": "passed",
                "out": str(out),
                "decision": decision,
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

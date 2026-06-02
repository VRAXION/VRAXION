#!/usr/bin/env python3
"""Artifact checker for D72 support-cost oracle-gap bound analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_REPORTS = [
    "d71_upstream_manifest.json",
    "oracle_gap_decomposition_report.json",
    "irreducible_cost_bound_report.json",
    "reducible_cost_report.json",
    "joint_recall_cost_report.json",
    "external_recall_cost_report.json",
    "false_confidence_cost_report.json",
    "abstain_cost_report.json",
    "low_cost_variant_harm_report.json",
    "safe_deescalation_frontier_report.json",
    "min_seed_bound_report.json",
    "truth_leak_audit_report.json",
    "rust_invocation_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
]

ALLOWED_DECISIONS = {
    "oracle_gap_reducible_cost_identified",
    "oracle_gap_safety_bound_identified",
    "oracle_gap_bound_inconclusive",
    "oracle_gap_bound_analysis_safety_failure",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fail(message: str) -> None:
    raise SystemExit(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    if not out.exists():
        fail(f"out path missing: {out}")
    missing = [name for name in REQUIRED_REPORTS if not (out / name).exists()]
    if missing:
        fail(f"missing required reports: {missing}")

    decision = load_json(out / "decision.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    upstream = load_json(out / "d71_upstream_manifest.json")
    decomposition = load_json(out / "oracle_gap_decomposition_report.json")
    irreducible = load_json(out / "irreducible_cost_bound_report.json")
    reducible = load_json(out / "reducible_cost_report.json")
    joint = load_json(out / "joint_recall_cost_report.json")
    external = load_json(out / "external_recall_cost_report.json")
    false_conf = load_json(out / "false_confidence_cost_report.json")
    abstain = load_json(out / "abstain_cost_report.json")
    low_cost = load_json(out / "low_cost_variant_harm_report.json")
    frontier = load_json(out / "safe_deescalation_frontier_report.json")
    min_seed = load_json(out / "min_seed_bound_report.json")
    truth = load_json(out / "truth_leak_audit_report.json")
    rust = load_json(out / "rust_invocation_report.json")

    if decision.get("decision") not in ALLOWED_DECISIONS:
        fail(f"unexpected decision: {decision.get('decision')}")
    if aggregate.get("failed_jobs"):
        fail(f"failed jobs present: {aggregate.get('failed_jobs')}")
    if aggregate.get("fallback_rows") != 0:
        fail(f"fallback rows not zero: {aggregate.get('fallback_rows')}")
    if not aggregate.get("rust_path_invoked") or not rust.get("rust_path_invoked"):
        fail("rust path provenance missing")
    if rust.get("fallback_rows") != 0 or rust.get("failed_jobs"):
        fail("rust fallback/failed job invariant violated")

    observed = upstream.get("d71_artifacts", {})
    if observed.get("decision") != "support_cost_oracle_gap_scale_confirmed":
        fail(f"D71 upstream decision mismatch: {observed.get('decision')}")
    if observed.get("next") != "D72_SUPPORT_COST_ORACLE_GAP_BOUND_ANALYSIS":
        fail(f"D71 upstream next mismatch: {observed.get('next')}")
    if observed.get("scaled_arm") != "D70_ORACLE_GAP_TARGETED_REPLAY":
        fail(f"D71 scaled arm mismatch: {observed.get('scaled_arm')}")

    replay = aggregate.get("d71_replay", {})
    required = [
        "exact_joint_accuracy",
        "correlated_echo_accuracy",
        "adversarial_distractor_accuracy",
        "external_test_required_accuracy",
        "false_confidence_rate",
        "indistinguishable_abstain_rate",
        "average_total_support_used",
        "counter_support_used",
        "support_gap_vs_oracle",
        "wrong_concrete_counter_rate",
        "weak_top1_top2_path_failure_rate",
        "joint_counter_recall_on_joint_required_rows",
        "external_recall_on_external_required_rows",
        "d68_loss_repair_preservation_rate",
        "min_seed_exact",
        "min_seed_correlated",
        "min_seed_adversarial",
        "min_seed_external",
    ]
    absent = [key for key in required if key not in replay]
    if absent:
        fail(f"D71 replay missing metrics: {absent}")

    total = decomposition.get("total_remaining_gap")
    irreducible_cost = decomposition.get("estimated_irreducible_cost")
    reducible_cost = decomposition.get("estimated_reducible_cost")
    if total is None or irreducible_cost is None or reducible_cost is None:
        fail("decomposition missing total/irreducible/reducible fields")
    if abs((irreducible_cost + reducible_cost) - total) > 1e-9:
        fail("irreducible + reducible does not equal total remaining gap")
    if decomposition.get("classification") not in {"safety_routing_bound_with_small_reducible_tail", "reducible_cost_identified", "inconclusive"}:
        fail(f"unexpected classification: {decomposition.get('classification')}")
    if not decomposition.get("blocking_gates"):
        fail("blocking gates were not named")

    if decision.get("decision") == "oracle_gap_safety_bound_identified":
        if irreducible_cost < 0.35 or reducible_cost >= 0.20:
            fail("safety-bound decision does not match irreducible/reducible estimates")
    if decision.get("decision") == "oracle_gap_reducible_cost_identified" and reducible_cost < 0.20:
        fail("reducible-cost decision without >=0.20 reducible estimate")

    if irreducible.get("estimated_irreducible_cost") != irreducible_cost:
        fail("irreducible report mismatch")
    if reducible.get("estimated_reducible_cost") != reducible_cost:
        fail("reducible report mismatch")
    if not joint.get("gate_blocks_more_cut"):
        fail("joint recall bound did not block more cut")
    if not external.get("gate_blocks_more_cut"):
        fail("external recall bound did not block more cut")
    if not false_conf.get("gate_blocks_more_cut"):
        fail("false-confidence bound did not block more cut")
    if not abstain.get("gate_blocks_more_cut"):
        fail("abstain bound did not block more cut")
    if not low_cost.get("harm_confirmed"):
        fail("low-cost harm was not confirmed")
    if low_cost.get("routing_failure_rows", 0) <= 0:
        fail("low-cost harm report has no routing failures")
    if frontier.get("safe_deescalation_below_d71_found"):
        fail("frontier unexpectedly found safe deescalation below D71")
    if not min_seed.get("min_seed_bound_blocks_blind_cut"):
        fail("min-seed bound did not block blind cut")

    if truth.get("fair_arms_using_truth_label"):
        fail("fair truth-label leak detected")
    if truth.get("fair_arms_using_support_regime_label"):
        fail("fair support-regime leak detected")
    if truth.get("row_id_lookup_used") or truth.get("python_hash_used"):
        fail("row lookup or Python hash hard gate failed")
    if truth.get("label_echo_fair_oracle_used"):
        fail("label echo fair oracle hard gate failed")
    if not truth.get("oracle_arms_reference_only"):
        fail("oracle arms are not reference-only")

    print(
        json.dumps(
            {
                "check": "passed",
                "out": str(out),
                "decision": decision,
                "estimated_irreducible_cost": irreducible_cost,
                "estimated_reducible_cost": reducible_cost,
                "classification": decomposition.get("classification"),
                "failed_jobs": aggregate.get("failed_jobs"),
                "fallback_rows": aggregate.get("fallback_rows"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

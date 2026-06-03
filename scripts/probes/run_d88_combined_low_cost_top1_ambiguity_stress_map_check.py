#!/usr/bin/env python3
"""Validate D88 combined low-cost + top1 ambiguity stress-map artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED = [
    "d87_upstream_manifest.json",
    "stress_axis_summary_report.json",
    "combined_low_cost_top1_extended_sweep_report.json",
    "low_cost_pressure_extended_sweep_report.json",
    "top1_top2_ambiguity_sweep_report.json",
    "combined_low_cost_ood_report.json",
    "ood_support_shift_report.json",
    "joint_required_boundary_report.json",
    "correlated_echo_stress_report.json",
    "adversarial_distractor_stress_report.json",
    "external_required_pressure_report.json",
    "indistinguishable_boundary_report.json",
    "top1_guard_corruption_report.json",
    "breakpoint_taxonomy_report.json",
    "safety_margin_watch_report.json",
    "D68_loss_repair_preservation_report.json",
    "truth_leak_audit_report.json",
    "rust_invocation_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fail(message: str) -> None:
    print(f"D88 check failed: {message}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/d88_combined_low_cost_top1_ambiguity_stress_map")
    args = parser.parse_args()
    out = Path(args.out)
    missing = [name for name in REQUIRED if not (out / name).exists()]
    if missing:
        fail(f"missing required reports: {missing}")

    decision = load(out / "decision.json")
    aggregate = load(out / "aggregate_metrics.json")
    manifest = load(out / "d87_upstream_manifest.json")
    best = aggregate.get("best_fair_arm", {})
    safety = load(out / "safety_margin_watch_report.json")
    top1 = load(out / "top1_guard_corruption_report.json")
    truth = load(out / "truth_leak_audit_report.json")
    rust = load(out / "rust_invocation_report.json")
    summary = load(out / "stress_axis_summary_report.json")
    taxonomy = load(out / "breakpoint_taxonomy_report.json")

    if decision.get("decision") != "combined_low_cost_top1_ambiguity_stress_map_completed":
        fail(f"unexpected decision: {decision.get('decision')}")
    if decision.get("next") != "D89_BREAKPOINT_REPAIR_OR_GENERALIZATION_PLAN":
        fail(f"unexpected next: {decision.get('next')}")
    if not summary.get("stress_map_complete") or not aggregate.get("stress_map_complete"):
        fail("stress map is not complete")
    if not summary.get("core_d87_holds_standard_stress") or not aggregate.get("core_d87_holds_standard_stress"):
        fail("D87 core did not hold under standard stress")
    if taxonomy.get("dominant_breakpoint") != "COMBINED_LOW_COST_PLUS_OOD":
        fail("dominant breakpoint taxonomy mismatch")
    if manifest.get("d87_artifacts", {}).get("decision") != "combined_low_cost_top1_ambiguity_scale_confirmed":
        fail("D87 upstream decision was not confirmed in manifest")
    if best.get("combined_low_cost_plus_top1_ambiguity_breakpoint", 0) < 0.750:
        fail("combined low-cost + top1 ambiguity breakpoint regressed below 0.750")
    if best.get("low_cost_pressure_breakpoint", 0) < 0.740:
        fail("low-cost breakpoint regressed below 0.740")
    if best.get("top1_top2_sufficiency_ambiguity_breakpoint", 0) < 0.742:
        fail("top1 ambiguity breakpoint regressed below 0.742")
    if best.get("D68_loss_repair_preservation_rate") != 1.0:
        fail("D68 loss repair preservation failed")
    if best.get("routing_failure_rows") != 0:
        fail("routing failures are nonzero")
    if not top1.get("top1_guard_preserved") or top1.get("top1_guard_weakened"):
        fail("top1 guard was not preserved")
    if top1.get("ablation_routing_failure_rows", 0) <= best.get("routing_failure_rows", 0):
        fail("top1 guard ablation is not worse")
    if not safety.get("passed"):
        fail("safety margin watch failed")
    if not truth.get("passed") or truth.get("fair_arms_using_truth_label"):
        fail("truth leak audit failed")
    if not rust.get("rust_path_invoked") or rust.get("fallback_rows") != 0:
        fail("Rust invocation or fallback invariant failed")
    if aggregate.get("fallback_rows") != 0 or aggregate.get("failed_jobs") != [] or decision.get("failed_jobs") != []:
        fail("fallback or failed_jobs invariant failed")

    print(json.dumps({"status": "ok", "decision": decision.get("decision"), "dominant_breakpoint": decision.get("dominant_breakpoint"), "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()

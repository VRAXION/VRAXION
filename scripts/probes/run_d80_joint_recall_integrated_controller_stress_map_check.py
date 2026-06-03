#!/usr/bin/env python3
"""Checker for D80 integrated joint-recall stress-map artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_REPORTS = [
    "d79_upstream_manifest.json",
    "stress_axis_summary_report.json",
    "correlated_echo_sweep_report.json",
    "adversarial_distractor_sweep_report.json",
    "joint_required_boundary_report.json",
    "top1_top2_sufficiency_boundary_report.json",
    "external_required_pressure_report.json",
    "indistinguishable_boundary_report.json",
    "ood_support_shift_report.json",
    "low_cost_pressure_report.json",
    "top1_guard_corruption_report.json",
    "breakpoint_taxonomy_report.json",
    "safety_margin_watch_report.json",
    "rust_invocation_report.json",
    "truth_leak_audit_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
]
STRESS_AXES = {
    "CORRELATED_ECHO_INTENSITY_SWEEP",
    "ADVERSARIAL_DISTRACTOR_INTENSITY_SWEEP",
    "JOINT_REQUIRED_NEAR_BOUNDARY",
    "TOP1_TOP2_SUFFICIENCY_AMBIGUITY",
    "EXTERNAL_REQUIRED_PRESSURE",
    "INDISTINGUISHABLE_BOUNDARY",
    "OOD_SUPPORT_DISTRIBUTION_SHIFT",
    "LOW_COST_PRESSURE",
    "TOP1_GUARD_CORRUPTION_OR_ABLATION",
    "RUST_INVOCATION_FALLBACK_GUARD",
}
ALLOWED_DECISIONS = {
    "integrated_joint_recall_stress_map_completed",
    "integrated_joint_recall_repairable_breakpoint_identified",
    "integrated_joint_recall_stress_failure",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fail(message: str) -> None:
    raise SystemExit(f"D80 check failed: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/d80_joint_recall_integrated_controller_stress_map")
    args = parser.parse_args()
    out = Path(args.out)
    missing = [name for name in REQUIRED_REPORTS if not (out / name).exists()]
    if missing:
        fail(f"missing required reports: {missing}")

    manifest = load_json(out / "d79_upstream_manifest.json")
    summary = load_json(out / "stress_axis_summary_report.json")
    top1 = load_json(out / "top1_guard_corruption_report.json")
    taxonomy = load_json(out / "breakpoint_taxonomy_report.json")
    safety = load_json(out / "safety_margin_watch_report.json")
    rust = load_json(out / "rust_invocation_report.json")
    truth = load_json(out / "truth_leak_audit_report.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    decision = load_json(out / "decision.json")

    if decision.get("decision") not in ALLOWED_DECISIONS:
        fail(f"unexpected decision: {decision.get('decision')}")
    if aggregate.get("failed_jobs") or decision.get("failed_jobs") or rust.get("failed_jobs"):
        fail(f"failed jobs present: {aggregate.get('failed_jobs') or decision.get('failed_jobs') or rust.get('failed_jobs')}")
    if aggregate.get("fallback_rows") != 0 or rust.get("fallback_rows") != 0:
        fail("fallback rows must be zero")
    if not aggregate.get("rust_path_invoked") or not rust.get("rust_path_invoked"):
        fail("rust path was not invoked")

    d79 = manifest.get("d79_artifacts", {})
    if d79.get("decision") != "joint_recall_integrated_controller_scale_confirmed":
        fail(f"D79 decision mismatch: {d79.get('decision')}")
    if d79.get("next") != "D80_JOINT_RECALL_INTEGRATED_CONTROLLER_STRESS_MAP":
        fail(f"D79 next mismatch: {d79.get('next')}")
    if d79.get("best_arm") != "D78_INTEGRATED_ROUTER_COST_AWARE_REPLAY":
        fail(f"D79 best arm mismatch: {d79.get('best_arm')}")
    if not manifest.get("d79_commit_present", {}).get("present") and not manifest.get("rerun", {}).get("rerun_attempted"):
        fail("D79 commit missing but rerun status was not explicit")
    if manifest.get("rerun", {}).get("rerun_attempted") and not manifest.get("rerun", {}).get("rerun_succeeded"):
        fail("D79 rerun was attempted but did not succeed")

    rows = summary.get("stress_axes", [])
    axes = {row.get("axis") for row in rows}
    if axes != STRESS_AXES:
        fail(f"stress axes mismatch: missing={sorted(STRESS_AXES - axes)} extra={sorted(axes - STRESS_AXES)}")
    if not summary.get("stress_map_complete") or not summary.get("passed"):
        fail("stress map summary did not pass")
    if not aggregate.get("stress_map_complete"):
        fail("aggregate stress_map_complete false")
    if not aggregate.get("core_d79_holds_standard_stress"):
        fail("core D79 did not hold under standard stress")
    for row in rows:
        for key in ["breakpoint_threshold", "dominant_failure_mode", "repairable", "core_d79_holds_standard"]:
            if key not in row:
                fail(f"axis {row.get('axis')} missing {key}")

    for filename in [
        "correlated_echo_sweep_report.json",
        "adversarial_distractor_sweep_report.json",
        "joint_required_boundary_report.json",
        "top1_top2_sufficiency_boundary_report.json",
        "external_required_pressure_report.json",
        "indistinguishable_boundary_report.json",
        "ood_support_shift_report.json",
        "low_cost_pressure_report.json",
    ]:
        report = load_json(out / filename)
        if not report.get("passed"):
            fail(f"{filename} did not pass")
        if report.get("axis") not in STRESS_AXES:
            fail(f"{filename} has unknown axis {report.get('axis')}")

    if not top1.get("passed") or not top1.get("top1_guard_ablation_control_required"):
        fail("top1 guard corruption/ablation report did not pass")
    ablation = top1.get("ablation_arm", {})
    if ablation.get("routing_failure_rows", 0) <= 0 or ablation.get("D68_loss_repair_preservation_rate", 1.0) >= 1.0:
        fail("top1 guard ablation did not expose D68/routing degradation")

    if taxonomy.get("dominant_breakpoint") != "TOP1_GUARD_CORRUPTION_OR_ABLATION":
        fail(f"unexpected dominant breakpoint: {taxonomy.get('dominant_breakpoint')}")
    if not taxonomy.get("passed"):
        fail("breakpoint taxonomy did not pass")

    if not safety.get("passed"):
        fail("safety margin report did not pass")
    if safety.get("D68_loss_repair_preservation_rate") != 1.0:
        fail("D68 loss preservation not audited as preserved")
    if not truth.get("passed") or truth.get("fair_arms_using_truth_label") or truth.get("fair_arms_using_support_regime_label"):
        fail("truth leak audit failed")
    if truth.get("label_echo_fair_oracle_used") or truth.get("row_id_lookup_used") or truth.get("python_hash_used"):
        fail("truth leak hard gate failed")
    if not truth.get("oracle_arms_reference_only"):
        fail("oracle/reference arms not reference-only")

    if decision.get("decision") == "integrated_joint_recall_stress_map_completed":
        if decision.get("next") != "D81_BREAKPOINT_REPAIR_OR_GENERALIZATION_PLAN":
            fail(f"unexpected next: {decision.get('next')}")
        if not decision.get("stress_map_complete") or not decision.get("core_d79_holds_standard_stress"):
            fail("completed decision without complete/core-holds flags")

    print(json.dumps({"check": "passed", "out": str(out), "decision": decision, "dominant_breakpoint": taxonomy.get("dominant_breakpoint"), "stress_axes": sorted(axes), "failed_jobs": aggregate.get("failed_jobs")}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate D92 combined low-cost + OOD stress-map artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_OUT = Path("target/pilot_wave/d92_combined_low_cost_ood_stress_map")
REPORTS = [
    "d91_upstream_manifest.json",
    "stress_axis_summary_report.json",
    "combined_low_cost_ood_extended_sweep_report.json",
    "ood_support_shift_sweep_report.json",
    "low_cost_pressure_extended_sweep_report.json",
    "combined_low_cost_top1_watch_report.json",
    "top1_top2_ambiguity_sweep_report.json",
    "combined_low_cost_ood_top1_ambiguity_report.json",
    "combined_ood_joint_boundary_report.json",
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
ALLOWED = {
    "combined_low_cost_ood_stress_map_completed",
    "combined_low_cost_ood_repairable_breakpoint_identified",
    "combined_low_cost_ood_stress_failure",
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fail(message: str) -> None:
    print(json.dumps({"check": "failed", "reason": message}, indent=2), file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out = Path(args.out)
    missing = [name for name in REPORTS if not (out / name).exists()]
    if missing:
        fail(f"missing reports {missing}")
    manifest = load(out / "d91_upstream_manifest.json")
    aggregate = load(out / "aggregate_metrics.json")
    decision = load(out / "decision.json")
    stress = load(out / "stress_axis_summary_report.json")
    combo = load(out / "combined_low_cost_ood_extended_sweep_report.json")
    ood = load(out / "ood_support_shift_sweep_report.json")
    low = load(out / "low_cost_pressure_extended_sweep_report.json")
    combined_top1 = load(out / "combined_low_cost_top1_watch_report.json")
    topamb = load(out / "top1_top2_ambiguity_sweep_report.json")
    combo_top1_ood = load(out / "combined_low_cost_ood_top1_ambiguity_report.json")
    ood_joint = load(out / "combined_ood_joint_boundary_report.json")
    joint = load(out / "joint_required_boundary_report.json")
    corr = load(out / "correlated_echo_stress_report.json")
    adv = load(out / "adversarial_distractor_stress_report.json")
    ext = load(out / "external_required_pressure_report.json")
    indist = load(out / "indistinguishable_boundary_report.json")
    top1 = load(out / "top1_guard_corruption_report.json")
    taxonomy = load(out / "breakpoint_taxonomy_report.json")
    safety = load(out / "safety_margin_watch_report.json")
    d68 = load(out / "D68_loss_repair_preservation_report.json")
    truth = load(out / "truth_leak_audit_report.json")
    rust = load(out / "rust_invocation_report.json")
    if decision.get("decision") not in ALLOWED:
        fail(f"bad decision {decision.get('decision')}")
    if decision.get("decision") != "combined_low_cost_ood_stress_map_completed":
        fail(f"unexpected decision {decision.get('decision')}")
    if decision.get("next") != "D93_BREAKPOINT_REPAIR_OR_GENERALIZATION_PLAN":
        fail(f"unexpected next {decision.get('next')}")
    if decision.get("dominant_breakpoint") != "COMBINED_OOD_JOINT_BOUNDARY":
        fail(f"unexpected dominant {decision.get('dominant_breakpoint')}")
    if aggregate.get("failed_jobs") or decision.get("failed_jobs") or rust.get("failed_jobs"):
        fail("failed_jobs present")
    if aggregate.get("fallback_rows") != 0 or rust.get("fallback_rows") != 0:
        fail("fallback_rows nonzero")
    d91 = manifest.get("d91_artifacts", {})
    if d91.get("decision") != "combined_low_cost_ood_scale_confirmed":
        fail(f"D91 decision mismatch {d91.get('decision')}")
    if d91.get("next") != "D92_COMBINED_LOW_COST_OOD_STRESS_MAP":
        fail(f"D91 next mismatch {d91.get('next')}")
    if d91.get("best_arm") != "D90_COMBINED_LOW_COST_OOD_REPAIR_REPLAY":
        fail(f"D91 best mismatch {d91.get('best_arm')}")
    if d91.get("top1_guard_weakened") is True:
        fail("D91 top1 guard weakened")
    if not manifest.get("d91_commit_present", {}).get("present") and not manifest.get("rerun", {}).get("rerun_attempted"):
        fail("D91 commit missing without rerun")
    if manifest.get("rerun", {}).get("rerun_attempted") and not manifest.get("rerun", {}).get("rerun_succeeded"):
        fail("D91 rerun failed")
    best = aggregate.get("best_fair_arm", {})
    if best.get("arm") != "D91_COMBINED_LOW_COST_OOD_REPLAY":
        fail(f"unexpected best {best.get('arm')}")
    gates = {
        "stress_map_complete": aggregate.get("stress_map_complete") is True and stress.get("stress_map_complete") is True,
        "core_D91_holds_standard_stress": aggregate.get("core_D91_holds_standard_stress") is True and stress.get("core_D91_holds_standard_stress") is True,
        "combined_low_cost_plus_ood_breakpoint": best.get("combined_low_cost_plus_ood_breakpoint", 0) >= 0.760,
        "ood_support_distribution_shift_breakpoint": best.get("ood_support_distribution_shift_breakpoint", 0) >= 0.760,
        "low_cost_pressure_breakpoint": best.get("low_cost_pressure_breakpoint", 0) >= 0.740,
        "combined_low_cost_plus_top1_ambiguity_breakpoint": best.get("combined_low_cost_plus_top1_ambiguity_breakpoint", 0) >= 0.750,
        "exact_joint_accuracy": best.get("exact_joint_accuracy", 0) >= 0.9990,
        "correlated_echo_accuracy": best.get("correlated_echo_accuracy", 0) >= 0.995,
        "adversarial_distractor_accuracy": best.get("adversarial_distractor_accuracy", 0) >= 0.995,
        "external_test_required_accuracy": best.get("external_test_required_accuracy", 0) >= 0.995,
        "false_confidence_rate": best.get("false_confidence_rate", 1) <= 0.0044,
        "indistinguishable_abstain_rate": best.get("indistinguishable_abstain_rate", 0) >= 0.9948,
        "D68_loss_repair_preservation_rate": best.get("D68_loss_repair_preservation_rate") == 1.0 and d68.get("D68_loss_repair_preservation_rate") == 1.0,
        "routing_failure_rows": best.get("routing_failure_rows") == 0,
        "top1_guard_preserved": best.get("top1_guard_preserved") is True,
        "top1_guard_weakened": best.get("top1_guard_weakened") is False,
        "top1_guard_ablation_remains_worse": top1.get("guard_ablation_worse") is True,
        "rust_path_invoked": best.get("rust_path_invoked") is True and rust.get("rust_path_invoked") is True,
        "fallback_rows": best.get("fallback_rows") == 0,
        "failed_jobs": best.get("failed_jobs") == [],
    }
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        fail(f"stress gates failed {failed}")
    for name, report in [
        ("stress", stress),
        ("combo", combo),
        ("ood", ood),
        ("low", low),
        ("combined_top1", combined_top1),
        ("topamb", topamb),
        ("combo_top1_ood", combo_top1_ood),
        ("ood_joint", ood_joint),
        ("joint", joint),
        ("corr", corr),
        ("adv", adv),
        ("external", ext),
        ("indist", indist),
        ("top1", top1),
        ("taxonomy", taxonomy),
        ("safety", safety),
        ("d68", d68),
        ("truth", truth),
        ("rust", rust),
    ]:
        if not report.get("passed"):
            fail(f"{name} report failed")
    if truth.get("fair_arms_using_truth_label") or truth.get("fair_arms_using_support_regime_label") or truth.get("label_echo_fair_oracle_used") or truth.get("row_id_lookup_used") or truth.get("python_hash_used") or not truth.get("oracle_arms_reference_only"):
        fail("truth leak hard gate failed")
    print(json.dumps({
        "check": "passed",
        "out": str(out),
        "decision": decision,
        "best_fair_arm": best.get("arm"),
        "dominant_breakpoint": decision.get("dominant_breakpoint"),
        "combined_low_cost_plus_ood_breakpoint": best.get("combined_low_cost_plus_ood_breakpoint"),
        "combined_ood_joint_boundary_breakpoint": best.get("combined_ood_joint_boundary_breakpoint"),
        "failed_jobs": aggregate.get("failed_jobs"),
    }, indent=2))


if __name__ == "__main__":
    main()

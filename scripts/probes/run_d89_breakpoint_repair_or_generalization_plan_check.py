#!/usr/bin/env python3
"""Validate D89 breakpoint repair/generalization planning artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_OUT = Path("target/pilot_wave/d89_breakpoint_repair_or_generalization_plan")
REQUIRED = [
    "d88_upstream_manifest.json",
    "breakpoint_ranking_report.json",
    "combined_low_cost_ood_analysis_report.json",
    "ood_generalization_candidate_report.json",
    "top1_guard_invariant_report.json",
    "repair_candidate_roi_report.json",
    "generalization_candidate_report.json",
    "D90_proof_gate_report.json",
    "risk_register.json",
    "truth_leak_audit_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
]
ALLOWED_DECISIONS = {
    "combined_low_cost_ood_plan_selected",
    "ood_support_shift_generalization_plan_selected",
    "combined_low_cost_top1_ood_plan_selected",
    "joint_boundary_repair_plan_selected",
    "breakpoint_repair_plan_not_ready",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fail(message: str) -> None:
    print(json.dumps({"check": "failed", "reason": message}, indent=2), file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out = Path(args.out)
    missing = [name for name in REQUIRED if not (out / name).exists()]
    if missing:
        fail(f"missing reports {missing}")

    manifest = load(out / "d88_upstream_manifest.json")
    ranking = load(out / "breakpoint_ranking_report.json")
    analysis = load(out / "combined_low_cost_ood_analysis_report.json")
    ood = load(out / "ood_generalization_candidate_report.json")
    top1 = load(out / "top1_guard_invariant_report.json")
    roi = load(out / "repair_candidate_roi_report.json")
    generalization = load(out / "generalization_candidate_report.json")
    gates = load(out / "D90_proof_gate_report.json")
    risks = load(out / "risk_register.json")
    truth = load(out / "truth_leak_audit_report.json")
    aggregate = load(out / "aggregate_metrics.json")
    decision = load(out / "decision.json")

    if decision.get("decision") not in ALLOWED_DECISIONS:
        fail(f"bad decision {decision.get('decision')}")
    if decision.get("decision") != "combined_low_cost_ood_plan_selected":
        fail(f"unexpected selected decision {decision.get('decision')}")
    if decision.get("next") != "D90_COMBINED_LOW_COST_OOD_REPAIR_PROTOTYPE":
        fail(f"unexpected next {decision.get('next')}")
    if decision.get("selected_repair_path") != "COMBINED_LOW_COST_OOD_REPAIR_PLAN":
        fail("wrong selected repair path")
    if aggregate.get("failed_jobs") or decision.get("failed_jobs"):
        fail("failed_jobs must be empty")

    d88 = manifest.get("d88_artifacts", {})
    if d88.get("decision") != "combined_low_cost_top1_ambiguity_stress_map_completed":
        fail(f"D88 decision mismatch {d88.get('decision')}")
    if d88.get("next") != "D89_BREAKPOINT_REPAIR_OR_GENERALIZATION_PLAN":
        fail(f"D88 next mismatch {d88.get('next')}")
    if not d88.get("stress_map_complete") or not d88.get("core_D87_holds_standard_stress"):
        fail("D88 stress/core handoff not complete")
    if d88.get("dominant_breakpoint") != "COMBINED_LOW_COST_PLUS_OOD":
        fail("D88 dominant breakpoint mismatch")
    if d88.get("hard_invariant_breakpoint") != "TOP1_GUARD_CORRUPTION_OR_ABLATION":
        fail("D88 hard invariant mismatch")
    if d88.get("top1_guard_weakened") is True:
        fail("D88 top1 guard weakened")
    if not manifest.get("d88_commit_present", {}).get("present") and not manifest.get("rerun", {}).get("rerun_attempted"):
        fail("D88 commit missing without rerun/restore attempt")
    if manifest.get("rerun", {}).get("rerun_attempted") and not manifest.get("rerun", {}).get("rerun_succeeded"):
        fail("D88 rerun/restore failed")

    rows = ranking.get("ranking", [])
    if not rows or rows[0].get("candidate") != "COMBINED_LOW_COST_OOD_REPAIR_PLAN":
        fail("breakpoint ranking did not select combined low-cost OOD first")
    if rows[0].get("target_breakpoint") != "COMBINED_LOW_COST_PLUS_OOD" or rows[0].get("breakpoint_threshold") != 0.744:
        fail("top breakpoint threshold mismatch")
    if not analysis.get("passed") or analysis.get("selected_repair_path") != decision.get("selected_repair_path"):
        fail("combined low-cost OOD analysis failed")
    if ood.get("best_generalization_candidate") != "OOD_SUPPORT_SHIFT_GENERALIZATION_PLAN" or not ood.get("passed"):
        fail("OOD generalization candidate report failed")
    if not roi.get("passed") or roi.get("selected_repair_path") != decision.get("selected_repair_path"):
        fail("ROI report failed")
    if not generalization.get("passed") or generalization.get("selected_primary_plan") != decision.get("selected_repair_path"):
        fail("generalization report failed")
    if top1.get("top1_guard_must_not_be_weakened") is not True or top1.get("is_disposable_cost_knob") is True or not top1.get("passed"):
        fail("top1 guard invariant failed")
    if top1.get("ablation_routing_failure_rows") != 45 or top1.get("ablation_D68_loss_repair_preservation_rate") >= 1.0:
        fail("top1 ablation evidence invalid")
    required_gates = gates.get("measurable_gates", [])
    for required in [
        "combined_low_cost_plus_ood_breakpoint >= 0.760",
        "top1 guard preserved=true and weakened=false",
        "D68_loss_repair_preservation_rate = 1.0",
        "routing_failure_rows = 0",
        "rust_path_invoked=true",
        "fallback_rows=0",
        "failed_jobs=[]",
    ]:
        if required not in required_gates:
            fail(f"D90 proof gate missing {required}")
    if not risks.get("passed"):
        fail("risk register failed")
    if truth.get("fair_arms_using_truth_label") or truth.get("fair_arms_using_support_regime_label") or truth.get("label_echo_fair_oracle_used") or truth.get("row_id_lookup_used") or truth.get("python_hash_used") or not truth.get("oracle_arms_reference_only") or not truth.get("passed"):
        fail("truth leak hard gate failed")

    print(json.dumps({"check": "passed", "out": str(out), "decision": decision, "selected_repair_path": decision.get("selected_repair_path"), "top_breakpoint": rows[0], "failed_jobs": aggregate.get("failed_jobs")}, indent=2))


if __name__ == "__main__":
    main()

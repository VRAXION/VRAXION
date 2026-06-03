#!/usr/bin/env python3
"""Validate D93 breakpoint repair/generalization planning artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_OUT = Path("target/pilot_wave/d93_breakpoint_repair_or_generalization_plan")
REQUIRED = [
    "d92_upstream_manifest.json",
    "breakpoint_ranking_report.json",
    "combined_ood_joint_boundary_analysis_report.json",
    "joint_boundary_candidate_report.json",
    "ood_generalization_candidate_report.json",
    "low_cost_ood_joint_combo_report.json",
    "top1_guard_invariant_report.json",
    "repair_candidate_roi_report.json",
    "generalization_candidate_report.json",
    "D94_proof_gate_report.json",
    "risk_register.json",
    "truth_leak_audit_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
]
ALLOWED_DECISIONS = {
    "combined_ood_joint_boundary_plan_selected",
    "joint_boundary_repair_plan_selected",
    "ood_support_shift_generalization_plan_selected",
    "combined_low_cost_ood_joint_plan_selected",
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

    manifest = load(out / "d92_upstream_manifest.json")
    ranking = load(out / "breakpoint_ranking_report.json")
    analysis = load(out / "combined_ood_joint_boundary_analysis_report.json")
    joint = load(out / "joint_boundary_candidate_report.json")
    ood = load(out / "ood_generalization_candidate_report.json")
    combo = load(out / "low_cost_ood_joint_combo_report.json")
    top1 = load(out / "top1_guard_invariant_report.json")
    roi = load(out / "repair_candidate_roi_report.json")
    generalization = load(out / "generalization_candidate_report.json")
    gates = load(out / "D94_proof_gate_report.json")
    risks = load(out / "risk_register.json")
    truth = load(out / "truth_leak_audit_report.json")
    aggregate = load(out / "aggregate_metrics.json")
    decision = load(out / "decision.json")

    if decision.get("decision") not in ALLOWED_DECISIONS:
        fail(f"bad decision {decision.get('decision')}")
    if decision.get("decision") != "combined_ood_joint_boundary_plan_selected":
        fail(f"unexpected selected decision {decision.get('decision')}")
    if decision.get("next") != "D94_COMBINED_OOD_JOINT_BOUNDARY_REPAIR_PROTOTYPE":
        fail(f"unexpected next {decision.get('next')}")
    if decision.get("selected_repair_path") != "COMBINED_OOD_JOINT_BOUNDARY_REPAIR_PLAN":
        fail("wrong selected repair path")
    if decision.get("dominant_breakpoint") != "COMBINED_OOD_JOINT_BOUNDARY":
        fail("wrong dominant breakpoint")
    if aggregate.get("failed_jobs") or decision.get("failed_jobs"):
        fail("failed_jobs must be empty")

    d92 = manifest.get("d92_artifacts", {})
    if d92.get("decision") != "combined_low_cost_ood_stress_map_completed":
        fail(f"D92 decision mismatch {d92.get('decision')}")
    if d92.get("next") != "D93_BREAKPOINT_REPAIR_OR_GENERALIZATION_PLAN":
        fail(f"D92 next mismatch {d92.get('next')}")
    if d92.get("best_fair_arm") != "D91_COMBINED_LOW_COST_OOD_REPLAY":
        fail(f"D92 best mismatch {d92.get('best_fair_arm')}")
    if d92.get("dominant_breakpoint") != "COMBINED_OOD_JOINT_BOUNDARY":
        fail("D92 dominant breakpoint mismatch")
    if not d92.get("stress_map_complete") or not d92.get("core_D91_holds_standard_stress"):
        fail("D92 stress/core handoff not complete")
    if d92.get("top1_guard_weakened") is True:
        fail("D92 top1 guard weakened")
    if not manifest.get("d92_commit_present", {}).get("present") and not manifest.get("rerun", {}).get("rerun_attempted"):
        fail("D92 commit missing without rerun/restore attempt")
    if manifest.get("rerun", {}).get("rerun_attempted") and not manifest.get("rerun", {}).get("rerun_succeeded"):
        fail("D92 rerun/restore failed")

    rows = ranking.get("ranking", [])
    if not rows or rows[0].get("candidate") != "COMBINED_OOD_JOINT_BOUNDARY_REPAIR_PLAN":
        fail("breakpoint ranking did not select combined OOD joint boundary first")
    if rows[0].get("target_breakpoint") != "COMBINED_OOD_JOINT_BOUNDARY" or rows[0].get("breakpoint_threshold") != 0.739:
        fail("top breakpoint threshold mismatch")
    for report_name, report in [
        ("ranking", ranking),
        ("analysis", analysis),
        ("joint", joint),
        ("ood", ood),
        ("combo", combo),
        ("top1", top1),
        ("roi", roi),
        ("generalization", generalization),
        ("gates", gates),
        ("risks", risks),
        ("truth", truth),
    ]:
        if not report.get("passed"):
            fail(f"{report_name} report failed")
    if analysis.get("selected_repair_path") != decision.get("selected_repair_path") or analysis.get("expected_ROI") != 0.79:
        fail("combined OOD joint analysis mismatch")
    if joint.get("candidate") != "JOINT_REQUIRED_BOUNDARY_REPAIR_PLAN" or joint.get("rank") != 2:
        fail("joint boundary candidate report mismatch")
    if ood.get("best_generalization_candidate") != "OOD_SUPPORT_SHIFT_GENERALIZATION_PLAN" or ood.get("rank") != 3:
        fail("OOD generalization candidate report mismatch")
    if combo.get("candidate") != "COMBINED_LOW_COST_OOD_JOINT_REPAIR_PLAN" or combo.get("rank") != 4:
        fail("low-cost OOD joint combo report mismatch")
    if top1.get("top1_guard_must_not_be_weakened") is not True or top1.get("is_disposable_cost_knob") is True:
        fail("top1 guard invariant failed")
    if top1.get("ablation_routing_failure_rows") != 45 or top1.get("ablation_D68_loss_repair_preservation_rate") >= 1.0:
        fail("top1 full ablation evidence invalid")
    if top1.get("partial_corruption_routing_failure_rows") != 18 or top1.get("partial_corruption_D68_loss_repair_preservation_rate") >= 1.0:
        fail("top1 partial corruption evidence invalid")
    required_gates = gates.get("measurable_gates", [])
    for required in [
        "combined_ood_joint_boundary_breakpoint >= 0.755",
        "combined_low_cost_plus_ood_breakpoint >= 0.760",
        "top1 guard preserved=true and weakened=false",
        "D68_loss_repair_preservation_rate = 1.0",
        "routing_failure_rows = 0",
        "rust_path_invoked=true",
        "fallback_rows=0",
        "failed_jobs=[]",
    ]:
        if required not in required_gates:
            fail(f"D94 proof gate missing {required}")
    if truth.get("fair_arms_using_truth_label") or truth.get("fair_arms_using_support_regime_label") or truth.get("label_echo_fair_oracle_used") or truth.get("row_id_lookup_used") or truth.get("python_hash_used") or not truth.get("oracle_arms_reference_only"):
        fail("truth leak hard gate failed")

    print(json.dumps({"check": "passed", "out": str(out), "decision": decision, "selected_repair_path": decision.get("selected_repair_path"), "top_breakpoint": rows[0], "failed_jobs": aggregate.get("failed_jobs")}, indent=2))


if __name__ == "__main__":
    main()

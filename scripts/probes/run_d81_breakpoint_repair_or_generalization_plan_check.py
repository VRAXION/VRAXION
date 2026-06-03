#!/usr/bin/env python3
"""Checker for D81 breakpoint repair/generalization planning artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_REPORTS = [
    "d80_upstream_manifest.json",
    "breakpoint_ranking_report.json",
    "top1_guard_invariant_report.json",
    "operational_breakpoint_priority_report.json",
    "repair_candidate_roi_report.json",
    "generalization_candidate_report.json",
    "D82_proof_gate_report.json",
    "risk_register.json",
    "truth_leak_audit_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
]
ALLOWED_DECISIONS = {
    "top1_guard_hardening_plan_selected",
    "low_cost_pressure_repair_plan_selected",
    "top1_top2_ambiguity_repair_plan_selected",
    "ood_support_shift_generalization_plan_selected",
    "combined_breakpoint_repair_plan_selected",
    "breakpoint_repair_plan_not_ready",
}
REQUIRED_D82_GATES = [
    "low_cost_pressure_breakpoint",
    "average_total_support_used",
    "distance_to_concrete_oracle_support",
    "gap_reduction_vs_D73_bound",
    "exact_joint_accuracy",
    "joint_counter_recall_on_joint_required_rows",
    "external_recall_on_external_required_rows",
    "wrong_concrete_counter_rate",
    "weak_top1_top2_path_failure_rate",
    "top1_top2_sufficient_false_joint_rate",
    "false_confidence_rate",
    "indistinguishable_abstain_rate",
    "D68_loss_repair_preservation_rate",
    "routing_failure_rows",
    "top1_guard_ablation_worse",
    "rust_path_invoked",
    "fallback_rows",
    "failed_jobs",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fail(message: str) -> None:
    raise SystemExit(f"D81 check failed: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/d81_breakpoint_repair_or_generalization_plan")
    args = parser.parse_args()
    out = Path(args.out)
    missing = [name for name in REQUIRED_REPORTS if not (out / name).exists()]
    if missing:
        fail(f"missing required reports: {missing}")

    manifest = load_json(out / "d80_upstream_manifest.json")
    ranking = load_json(out / "breakpoint_ranking_report.json")
    top1 = load_json(out / "top1_guard_invariant_report.json")
    priority = load_json(out / "operational_breakpoint_priority_report.json")
    roi = load_json(out / "repair_candidate_roi_report.json")
    generalization = load_json(out / "generalization_candidate_report.json")
    d82 = load_json(out / "D82_proof_gate_report.json")
    risk = load_json(out / "risk_register.json")
    truth = load_json(out / "truth_leak_audit_report.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    decision = load_json(out / "decision.json")

    if decision.get("decision") not in ALLOWED_DECISIONS:
        fail(f"unexpected decision: {decision.get('decision')}")
    if aggregate.get("failed_jobs") or decision.get("failed_jobs") or risk.get("failed_jobs"):
        fail(f"failed jobs present: {aggregate.get('failed_jobs') or decision.get('failed_jobs') or risk.get('failed_jobs')}")

    d80 = manifest.get("d80_artifacts", {})
    if d80.get("decision") != "integrated_joint_recall_stress_map_completed":
        fail(f"D80 decision mismatch: {d80.get('decision')}")
    if d80.get("next") != "D81_BREAKPOINT_REPAIR_OR_GENERALIZATION_PLAN":
        fail(f"D80 next mismatch: {d80.get('next')}")
    if not d80.get("stress_map_complete") or not d80.get("core_d79_holds_standard_stress"):
        fail("D80 stress map/core-holds flags not confirmed")
    if d80.get("dominant_breakpoint") != "TOP1_GUARD_CORRUPTION_OR_ABLATION":
        fail(f"D80 dominant breakpoint mismatch: {d80.get('dominant_breakpoint')}")
    if not manifest.get("d80_commit_present", {}).get("present") and not manifest.get("rerun", {}).get("rerun_attempted"):
        fail("D80 commit missing but rerun status was not explicit")
    if manifest.get("rerun", {}).get("rerun_attempted") and not manifest.get("rerun", {}).get("rerun_succeeded"):
        fail("D80 rerun attempted but not successful")

    for name, report in [("ranking", ranking), ("top1", top1), ("priority", priority), ("roi", roi), ("generalization", generalization), ("D82", d82), ("risk", risk), ("truth", truth)]:
        if not report.get("passed"):
            fail(f"{name} report did not pass")

    if top1.get("status") != "hard_invariant_and_hardening_target":
        fail(f"unexpected top1 guard status: {top1.get('status')}")
    if top1.get("is_disposable_cost_knob"):
        fail("top1 guard marked as disposable cost knob")
    if not top1.get("must_not_be_weakened_without_guard_proof"):
        fail("top1 guard weakening proof requirement missing")
    ablation = top1.get("D80_ablation", {})
    if ablation.get("routing_failure_rows") != 45 or ablation.get("D68_loss_repair_preservation_rate") != 0.961538:
        fail("top1 ablation D80 evidence mismatch")

    rows = ranking.get("breakpoint_ranking", [])
    if not rows or rows[0].get("breakpoint") != "TOP1_GUARD_CORRUPTION_OR_ABLATION":
        fail("top1 guard not first in overall breakpoint ranking")
    if ranking.get("weakest_operational_breakpoint") != "LOW_COST_PRESSURE":
        fail("weakest operational breakpoint is not LOW_COST_PRESSURE")
    if priority.get("selected_operational_breakpoint") != "LOW_COST_PRESSURE":
        fail("selected operational breakpoint is not LOW_COST_PRESSURE")
    if not priority.get("single_target_D82"):
        fail("D82 should be single-target for attribution")

    selected = [row for row in roi.get("candidates", []) if row.get("selected")]
    if len(selected) != 1 or selected[0].get("candidate") != "LOW_COST_PRESSURE_REPAIR_PLAN":
        fail(f"unexpected selected ROI candidate: {selected}")
    if generalization.get("combined_plan_required_now"):
        fail("combined plan should not be required now")

    if decision.get("decision") == "low_cost_pressure_repair_plan_selected":
        if decision.get("next") != "D82_LOW_COST_PRESSURE_REPAIR_WITH_TOP1_GUARD":
            fail(f"unexpected next: {decision.get('next')}")
        if decision.get("selected_repair_path") != "LOW_COST_PRESSURE_REPAIR_PLAN":
            fail(f"unexpected selected repair path: {decision.get('selected_repair_path')}")
    else:
        fail(f"D81 should select low-cost pressure repair for this stress map, got {decision.get('decision')}")

    gates = d82.get("measurable_gates", {})
    missing_gates = [key for key in REQUIRED_D82_GATES if key not in gates]
    if missing_gates:
        fail(f"D82 proof gates missing: {missing_gates}")
    if d82.get("next_milestone") != "D82_LOW_COST_PRESSURE_REPAIR_WITH_TOP1_GUARD":
        fail(f"D82 next mismatch: {d82.get('next_milestone')}")
    if not d82.get("single_target"):
        fail("D82 proof report must declare single_target true")
    if "top1 sufficiency guard ablation remains worse" not in d82.get("required_ablations_guards", []):
        fail("D82 top1 guard ablation proof missing")

    if truth.get("fair_arms_using_truth_label") or truth.get("fair_arms_using_support_regime_label"):
        fail("truth/support-regime leak detected")
    if truth.get("label_echo_fair_oracle_used") or truth.get("row_id_lookup_used") or truth.get("python_hash_used"):
        fail("truth leak hard gate failed")
    if not truth.get("oracle_arms_reference_only"):
        fail("oracle/reference arms not reference-only")

    print(json.dumps({"check": "passed", "out": str(out), "decision": decision, "selected_repair_path": decision.get("selected_repair_path"), "top1_guard_status": top1.get("status"), "weakest_operational_breakpoint": ranking.get("weakest_operational_breakpoint"), "failed_jobs": aggregate.get("failed_jobs")}, indent=2))


if __name__ == "__main__":
    main()

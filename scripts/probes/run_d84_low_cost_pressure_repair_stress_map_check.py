#!/usr/bin/env python3
"""Validate D84 low-cost pressure repair stress-map artifacts."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

DEFAULT_OUT=Path("target/pilot_wave/d84_low_cost_pressure_repair_stress_map")
REQ=["d83_upstream_manifest.json","stress_axis_summary_report.json","low_cost_pressure_extended_sweep_report.json","top1_top2_ambiguity_stress_report.json","ood_support_shift_stress_report.json","joint_required_boundary_stress_report.json","correlated_echo_stress_report.json","adversarial_distractor_stress_report.json","external_required_pressure_report.json","indistinguishable_boundary_report.json","combined_low_cost_ood_report.json","combined_low_cost_top1_ambiguity_report.json","top1_guard_corruption_report.json","breakpoint_taxonomy_report.json","safety_margin_watch_report.json","D68_loss_repair_preservation_report.json","truth_leak_audit_report.json","rust_invocation_report.json","aggregate_metrics.json","decision.json","summary.json","report.md"]
AXES={"LOW_COST_PRESSURE_EXTENDED_SWEEP","TOP1_TOP2_SUFFICIENCY_AMBIGUITY","OOD_SUPPORT_DISTRIBUTION_SHIFT","JOINT_REQUIRED_NEAR_BOUNDARY","HARD_CORRELATED_JOINT_RECALL","HARD_ADVERSARIAL_JOINT_RECALL","EXTERNAL_REQUIRED_PRESSURE","INDISTINGUISHABLE_BOUNDARY","TOP1_GUARD_CORRUPTION_OR_ABLATION","COMBINED_LOW_COST_PLUS_OOD","COMBINED_LOW_COST_PLUS_TOP1_AMBIGUITY","RUST_INVOCATION_FALLBACK_GUARD"}
ALLOWED={"low_cost_pressure_repair_stress_map_completed","low_cost_pressure_repair_repairable_breakpoint_identified","low_cost_pressure_repair_stress_failure"}

def load(p:Path): return json.loads(p.read_text(encoding="utf-8"))
def fail(msg): print(json.dumps({"check":"failed","reason":msg},indent=2)); sys.exit(1)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default=str(DEFAULT_OUT)); args=ap.parse_args(); out=Path(args.out)
    missing=[n for n in REQ if not (out/n).exists()]
    if missing: fail(f"missing reports {missing}")
    man=load(out/"d83_upstream_manifest.json"); summary=load(out/"stress_axis_summary_report.json"); top1=load(out/"top1_guard_corruption_report.json"); tax=load(out/"breakpoint_taxonomy_report.json"); safety=load(out/"safety_margin_watch_report.json"); d68=load(out/"D68_loss_repair_preservation_report.json"); truth=load(out/"truth_leak_audit_report.json"); rust=load(out/"rust_invocation_report.json"); agg=load(out/"aggregate_metrics.json"); dec=load(out/"decision.json")
    if dec.get("decision") not in ALLOWED: fail(f"bad decision {dec.get('decision')}")
    if agg.get("failed_jobs") or dec.get("failed_jobs") or rust.get("failed_jobs"): fail("failed jobs present")
    if agg.get("fallback_rows")!=0 or rust.get("fallback_rows")!=0: fail("fallback rows nonzero")
    if not agg.get("rust_path_invoked") or not rust.get("rust_path_invoked"): fail("rust path not invoked")
    d83=man.get("d83_artifacts",{})
    if d83.get("decision")!="low_cost_pressure_repair_scale_confirmed": fail(f"D83 decision mismatch {d83.get('decision')}")
    if d83.get("next")!="D84_LOW_COST_PRESSURE_REPAIR_STRESS_MAP": fail(f"D83 next mismatch {d83.get('next')}")
    if d83.get("best_arm")!="D82_LOW_COST_REPAIR_COST_AWARE_REPLAY": fail(f"D83 best mismatch {d83.get('best_arm')}")
    if d83.get("top1_guard_weakened") is True: fail("D83 top1 guard weakened")
    if not man.get("d83_commit_present",{}).get("present") and not man.get("rerun",{}).get("rerun_attempted"): fail("D83 commit missing without rerun")
    if man.get("rerun",{}).get("rerun_attempted") and not man.get("rerun",{}).get("rerun_succeeded"): fail("D83 rerun failed")
    rows=summary.get("stress_axes",[]); axes={r.get("axis") for r in rows}
    if axes!=AXES: fail(f"stress axes mismatch missing={sorted(AXES-axes)} extra={sorted(axes-AXES)}")
    if not summary.get("stress_map_complete") or not summary.get("core_d83_holds_standard_stress") or not summary.get("passed"): fail("summary did not pass")
    for r in rows:
        for k in ["breakpoint_threshold","dominant_failure_mode","repairable","core_d83_holds_standard","fallback_rows","failed_jobs"]:
            if k not in r: fail(f"axis {r.get('axis')} missing {k}")
    if agg.get("dominant_breakpoint")!="COMBINED_LOW_COST_PLUS_TOP1_AMBIGUITY": fail("dominant breakpoint mismatch")
    if tax.get("hard_invariant_breakpoint")!="TOP1_GUARD_CORRUPTION_OR_ABLATION": fail("hard invariant breakpoint missing")
    if not top1.get("hard_invariant") or top1.get("top1_guard_weakened") or not top1.get("top1_guard_preserved"): fail("top1 guard invariant failed")
    if top1.get("ablation_routing_failure_rows",0)<=0 or top1.get("ablation_D68_loss_repair_preservation_rate",1)>=1.0: fail("top1 ablation not worse")
    if d68.get("D68_loss_repair_preservation_rate")!=1.0 or not d68.get("passed"): fail("D68 preservation failed")
    if safety.get("routing_failure_rows")!=0 or safety.get("false_confidence_rate",1)>0.0044 or safety.get("wrong_concrete_counter_rate",1)>0.0007 or not safety.get("passed"): fail("safety margins failed")
    if truth.get("fair_arms_using_truth_label") or truth.get("fair_arms_using_support_regime_label") or truth.get("label_echo_fair_oracle_used") or truth.get("row_id_lookup_used") or truth.get("python_hash_used") or not truth.get("oracle_arms_reference_only") or not truth.get("passed"): fail("truth leak hard gate failed")
    for n in ["low_cost_pressure_extended_sweep_report.json","top1_top2_ambiguity_stress_report.json","ood_support_shift_stress_report.json","joint_required_boundary_stress_report.json","correlated_echo_stress_report.json","adversarial_distractor_stress_report.json","external_required_pressure_report.json","indistinguishable_boundary_report.json","combined_low_cost_ood_report.json","combined_low_cost_top1_ambiguity_report.json","breakpoint_taxonomy_report.json"]:
        if not load(out/n).get("passed"): fail(f"{n} failed")
    print(json.dumps({"check":"passed","out":str(out),"decision":dec,"dominant_breakpoint":agg.get("dominant_breakpoint"),"low_cost_pressure_breakpoint":agg.get("best_fair_arm",{}).get("low_cost_pressure_breakpoint"),"failed_jobs":agg.get("failed_jobs")},indent=2))
if __name__=="__main__": main()

#!/usr/bin/env python3
"""Validate D85 breakpoint repair/generalization planning artifacts."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
DEFAULT_OUT=Path("target/pilot_wave/d85_breakpoint_repair_or_generalization_plan")
REQ=["d84_upstream_manifest.json","breakpoint_ranking_report.json","combined_breakpoint_analysis_report.json","top1_guard_invariant_report.json","repair_candidate_roi_report.json","generalization_candidate_report.json","D86_proof_gate_report.json","risk_register.json","truth_leak_audit_report.json","aggregate_metrics.json","decision.json","summary.json","report.md"]
ALLOWED={"combined_low_cost_top1_ambiguity_plan_selected","top1_top2_ambiguity_repair_plan_selected","ood_support_shift_generalization_plan_selected","joint_boundary_repair_plan_selected","breakpoint_repair_plan_not_ready"}
def load(p:Path): return json.loads(p.read_text(encoding="utf-8"))
def fail(msg): print(json.dumps({"check":"failed","reason":msg},indent=2)); sys.exit(1)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default=str(DEFAULT_OUT)); args=ap.parse_args(); out=Path(args.out)
    missing=[n for n in REQ if not (out/n).exists()]
    if missing: fail(f"missing reports {missing}")
    man=load(out/"d84_upstream_manifest.json"); rank=load(out/"breakpoint_ranking_report.json"); combo=load(out/"combined_breakpoint_analysis_report.json"); top1=load(out/"top1_guard_invariant_report.json"); roi=load(out/"repair_candidate_roi_report.json"); gen=load(out/"generalization_candidate_report.json"); d86=load(out/"D86_proof_gate_report.json"); risks=load(out/"risk_register.json"); truth=load(out/"truth_leak_audit_report.json"); agg=load(out/"aggregate_metrics.json"); dec=load(out/"decision.json")
    if dec.get("decision") not in ALLOWED: fail(f"bad decision {dec.get('decision')}")
    if agg.get("failed_jobs") or dec.get("failed_jobs"): fail("failed jobs present")
    d84=man.get("d84_artifacts",{})
    if d84.get("decision")!="low_cost_pressure_repair_stress_map_completed": fail(f"D84 decision mismatch {d84.get('decision')}")
    if d84.get("next")!="D85_BREAKPOINT_REPAIR_OR_GENERALIZATION_PLAN": fail(f"D84 next mismatch {d84.get('next')}")
    if d84.get("dominant_breakpoint")!="COMBINED_LOW_COST_PLUS_TOP1_AMBIGUITY": fail("D84 dominant breakpoint mismatch")
    if not d84.get("stress_map_complete") or not d84.get("core_D83_holds_standard_stress"): fail("D84 core/stress map not complete")
    if d84.get("top1_guard_weakened") is True: fail("D84 top1 guard weakened")
    if not man.get("d84_commit_present",{}).get("present") and not man.get("rerun",{}).get("rerun_attempted"): fail("D84 commit missing without rerun")
    if man.get("rerun",{}).get("rerun_attempted") and not man.get("rerun",{}).get("rerun_succeeded"): fail("D84 rerun failed")
    if dec.get("selected_repair_path")!="COMBINED_LOW_COST_TOP1_AMBIGUITY_REPAIR_PLAN": fail("wrong selected repair path")
    if dec.get("next")!="D86_COMBINED_LOW_COST_TOP1_AMBIGUITY_REPAIR_PROTOTYPE": fail("wrong next milestone")
    rows=rank.get("ranking",[])
    if not rows or rows[0].get("breakpoint")!="COMBINED_LOW_COST_PLUS_TOP1_AMBIGUITY" or rows[0].get("threshold")!=0.736: fail("breakpoint ranking top mismatch")
    if combo.get("selected_repair_path")!=dec.get("selected_repair_path") or not combo.get("passed"): fail("combined analysis failed")
    if top1.get("top1_guard_must_not_be_weakened") is not True or top1.get("is_disposable_cost_knob") is True or not top1.get("passed"): fail("top1 invariant failed")
    if top1.get("ablation_routing_failure_rows")!=45 or top1.get("ablation_D68_loss_repair_preservation_rate")>=1.0: fail("top1 ablation evidence invalid")
    cand=roi.get("candidates",[])
    if not cand or cand[0].get("candidate")!="COMBINED_LOW_COST_TOP1_AMBIGUITY_REPAIR_PLAN" or not roi.get("passed"): fail("ROI report failed")
    if gen.get("best_generalization_candidate")!="OOD_SUPPORT_SHIFT_GENERALIZATION_PLAN" or not gen.get("passed"): fail("generalization report failed")
    gates=d86.get("measurable_gates",[])
    for needle in ["combined_low_cost_plus_top1_ambiguity_breakpoint >= 0.75","top1 guard preserved=true and weakened=false","D68_loss_repair_preservation_rate = 1.0","routing_failure_rows = 0","failed_jobs=[]"]:
        if needle not in gates: fail(f"D86 gate missing {needle}")
    if not risks.get("passed"): fail("risk register failed")
    if truth.get("fair_arms_using_truth_label") or truth.get("fair_arms_using_support_regime_label") or truth.get("label_echo_fair_oracle_used") or truth.get("row_id_lookup_used") or truth.get("python_hash_used") or not truth.get("oracle_arms_reference_only") or not truth.get("passed"): fail("truth leak hard gate failed")
    print(json.dumps({"check":"passed","out":str(out),"decision":dec,"selected_repair_path":dec.get("selected_repair_path"),"top_breakpoint":rows[0],"failed_jobs":agg.get("failed_jobs")},indent=2))
if __name__=="__main__": main()

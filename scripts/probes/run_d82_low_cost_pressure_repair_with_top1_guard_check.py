#!/usr/bin/env python3
"""Checker for D82 low-cost pressure repair artifacts."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from typing import Any
REQ=["d81_upstream_manifest.json","low_cost_pressure_repair_report.json","low_cost_pressure_sweep_report.json","top1_guard_preservation_report.json","top1_guard_ablation_report.json","D68_cheap_top1_regression_guard_report.json","D68_loss_repair_preservation_report.json","top1_top2_ambiguity_watch_report.json","joint_boundary_watch_report.json","ood_support_shift_watch_report.json","external_required_watch_report.json","indistinguishable_abstain_watch_report.json","safety_margin_watch_report.json","support_cost_frontier_report.json","oracle_distance_frontier_report.json","truth_leak_audit_report.json","rust_invocation_report.json","aggregate_metrics.json","decision.json","summary.json","report.md"]
ALLOWED={"low_cost_pressure_repair_confirmed","low_cost_pressure_repair_safety_regression","top1_guard_invariant_not_preserved","low_cost_pressure_repair_not_confirmed"}
D79_SUPPORT=6.6465; D79_DISTANCE=0.3265

def load(p:Path)->dict[str,Any]: return json.loads(p.read_text(encoding="utf-8"))
def fail(m:str)->None: raise SystemExit(f"D82 check failed: {m}")
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default="target/pilot_wave/d82_low_cost_pressure_repair_with_top1_guard"); args=ap.parse_args(); out=Path(args.out)
    miss=[n for n in REQ if not (out/n).exists()]
    if miss: fail(f"missing required reports: {miss}")
    man=load(out/"d81_upstream_manifest.json"); repair=load(out/"low_cost_pressure_repair_report.json"); sweep=load(out/"low_cost_pressure_sweep_report.json"); preserve=load(out/"top1_guard_preservation_report.json"); ab=load(out/"top1_guard_ablation_report.json"); d68=load(out/"D68_cheap_top1_regression_guard_report.json"); loss=load(out/"D68_loss_repair_preservation_report.json"); amb=load(out/"top1_top2_ambiguity_watch_report.json"); joint=load(out/"joint_boundary_watch_report.json"); ood=load(out/"ood_support_shift_watch_report.json"); ext=load(out/"external_required_watch_report.json"); abst=load(out/"indistinguishable_abstain_watch_report.json"); safety=load(out/"safety_margin_watch_report.json"); supp=load(out/"support_cost_frontier_report.json"); oracle=load(out/"oracle_distance_frontier_report.json"); truth=load(out/"truth_leak_audit_report.json"); rust=load(out/"rust_invocation_report.json"); agg=load(out/"aggregate_metrics.json"); dec=load(out/"decision.json")
    if dec.get("decision") not in ALLOWED: fail(f"bad decision {dec.get('decision')}")
    if agg.get("failed_jobs") or dec.get("failed_jobs") or rust.get("failed_jobs"): fail("failed jobs present")
    if agg.get("fallback_rows")!=0 or rust.get("fallback_rows")!=0: fail("fallback rows nonzero")
    d81=man.get("d81_artifacts",{})
    if d81.get("decision")!="low_cost_pressure_repair_plan_selected": fail(f"D81 decision mismatch {d81.get('decision')}")
    if d81.get("next")!="D82_LOW_COST_PRESSURE_REPAIR_WITH_TOP1_GUARD": fail(f"D81 next mismatch {d81.get('next')}")
    if d81.get("selected_repair_path")!="LOW_COST_PRESSURE_REPAIR_PLAN": fail("D81 selected path mismatch")
    if d81.get("top1_is_disposable_cost_knob") is True: fail("top1 guard disposable")
    if not man.get("d81_commit_present",{}).get("present") and not man.get("rerun",{}).get("rerun_attempted"): fail("D81 commit missing without rerun")
    if man.get("rerun",{}).get("rerun_attempted") and not man.get("rerun",{}).get("rerun_succeeded"): fail("D81 rerun failed")
    best=agg.get("best_fair_arm",{})
    req=["low_cost_pressure_breakpoint","exact_joint_accuracy","correlated_echo_accuracy","adversarial_distractor_accuracy","external_test_required_accuracy","false_confidence_rate","indistinguishable_abstain_rate","average_total_support_used","counter_support_used","distance_to_concrete_oracle_support","gap_reduction_vs_D73_bound","joint_counter_recall_on_joint_required_rows","external_recall_on_external_required_rows","wrong_concrete_counter_rate","weak_top1_top2_path_failure_rate","top1_top2_sufficient_false_joint_rate","D68_loss_repair_preservation_rate","routing_failure_rows","min_seed_exact","min_seed_correlated","min_seed_adversarial","min_seed_external","fallback_rows","failed_jobs"]
    absent=[k for k in req if k not in best]
    if absent: fail(f"best missing {absent}")
    if best.get("arm")!="LOW_COST_PRESSURE_REPAIR_COST_AWARE": fail(f"unexpected best {best.get('arm')}")
    abm=ab.get("ablation_metrics",{})
    gates={"low_cost_pressure_breakpoint":best["low_cost_pressure_breakpoint"]>=0.74,"top1_guard_ablation_remains_worse":ab.get("guard_ablation_worse") is True and abm.get("routing_failure_rows",0)>best["routing_failure_rows"] and abm.get("D68_loss_repair_preservation_rate",1)<best["D68_loss_repair_preservation_rate"],"exact_joint_accuracy":best["exact_joint_accuracy"]>=0.9990,"correlated_echo_accuracy":best["correlated_echo_accuracy"]>=0.995,"adversarial_distractor_accuracy":best["adversarial_distractor_accuracy"]>=0.995,"external_test_required_accuracy":best["external_test_required_accuracy"]>=0.995,"false_confidence_rate":best["false_confidence_rate"]<=0.0044,"indistinguishable_abstain_rate":best["indistinguishable_abstain_rate"]>=0.9948,"average_total_support_used":best["average_total_support_used"]<=D79_SUPPORT+0.02,"distance_to_concrete_oracle_support":best["distance_to_concrete_oracle_support"]<=D79_DISTANCE+0.02,"joint_counter_recall_on_joint_required_rows":best["joint_counter_recall_on_joint_required_rows"]>=0.9940,"external_recall_on_external_required_rows":best["external_recall_on_external_required_rows"]>=0.9957,"wrong_concrete_counter_rate":best["wrong_concrete_counter_rate"]<=0.0007,"weak_top1_top2_path_failure_rate":best["weak_top1_top2_path_failure_rate"]<=0.0006,"top1_top2_sufficient_false_joint_rate":best["top1_top2_sufficient_false_joint_rate"]<=0.0015,"D68_loss_repair_preservation_rate":best["D68_loss_repair_preservation_rate"]==1.0,"routing_failure_rows":best["routing_failure_rows"]==0,"rust_path_invoked":best["rust_path_invoked"] is True,"fallback_rows":best["fallback_rows"]==0,"failed_jobs":not best["failed_jobs"]}
    failed=[k for k,v in gates.items() if not v]
    if dec.get("decision")=="low_cost_pressure_repair_confirmed" and failed: fail(f"positive gates failed {failed}")
    for name,rep in [("repair",repair),("sweep",sweep),("preserve",preserve),("ablation",ab),("D68",d68),("loss",loss),("ambiguity",amb),("joint",joint),("ood",ood),("external",ext),("abstain",abst),("safety",safety),("support",supp),("oracle",oracle),("truth",truth),("rust",rust)]:
        if not rep.get("passed"): fail(f"{name} report failed")
    if preserve.get("top1_guard_weakened"): fail("top1 guard weakened")
    if loss.get("D68_loss_repair_preservation_rate")!=1.0: fail("D68 loss not preserved")
    if truth.get("fair_arms_using_truth_label") or truth.get("fair_arms_using_support_regime_label") or truth.get("label_echo_fair_oracle_used") or truth.get("row_id_lookup_used") or truth.get("python_hash_used") or not truth.get("oracle_arms_reference_only"): fail("truth leak hard gate failed")
    print(json.dumps({"check":"passed","out":str(out),"decision":dec,"best_fair_arm":best.get("arm"),"low_cost_pressure_breakpoint":best.get("low_cost_pressure_breakpoint"),"support":best.get("average_total_support_used"),"oracle_distance":best.get("distance_to_concrete_oracle_support"),"failed_jobs":agg.get("failed_jobs")},indent=2))
if __name__=="__main__": main()

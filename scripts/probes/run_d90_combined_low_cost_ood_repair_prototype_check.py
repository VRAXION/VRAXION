#!/usr/bin/env python3
"""Validate D90 combined low-cost + OOD repair prototype artifacts."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

DEFAULT_OUT = Path("target/pilot_wave/d90_combined_low_cost_ood_repair_prototype")
REPORTS = ["d89_upstream_manifest.json","combined_low_cost_ood_repair_report.json","combined_low_cost_ood_sweep_report.json","ood_support_shift_sweep_report.json","low_cost_pressure_sweep_report.json","combined_low_cost_top1_watch_report.json","top1_top2_ambiguity_watch_report.json","top1_guard_preservation_report.json","top1_guard_ablation_report.json","top1_guard_partial_corruption_report.json","D68_cheap_top1_regression_guard_report.json","D68_loss_repair_preservation_report.json","hard_correlated_joint_recall_report.json","hard_adversarial_joint_recall_report.json","external_required_watch_report.json","indistinguishable_abstain_watch_report.json","safety_margin_watch_report.json","oracle_distance_frontier_report.json","support_cost_frontier_report.json","truth_leak_audit_report.json","rust_invocation_report.json","aggregate_metrics.json","decision.json","summary.json","report.md"]
ALLOWED = {"combined_low_cost_ood_repair_confirmed","combined_low_cost_ood_safety_regression","top1_guard_invariant_violation","combined_low_cost_ood_repair_not_confirmed"}

def load(p: Path): return json.loads(p.read_text(encoding="utf-8"))
def fail(msg): print(json.dumps({"check":"failed","reason":msg},indent=2), file=sys.stderr); sys.exit(1)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default=str(DEFAULT_OUT)); args=ap.parse_args(); out=Path(args.out)
    missing=[n for n in REPORTS if not (out/n).exists()]
    if missing: fail(f"missing reports {missing}")
    man=load(out/"d89_upstream_manifest.json"); agg=load(out/"aggregate_metrics.json"); dec=load(out/"decision.json")
    repair=load(out/"combined_low_cost_ood_repair_report.json"); combo=load(out/"combined_low_cost_ood_sweep_report.json"); ood=load(out/"ood_support_shift_sweep_report.json"); low=load(out/"low_cost_pressure_sweep_report.json"); ctop=load(out/"combined_low_cost_top1_watch_report.json"); topamb=load(out/"top1_top2_ambiguity_watch_report.json"); preserve=load(out/"top1_guard_preservation_report.json"); ab=load(out/"top1_guard_ablation_report.json"); partial=load(out/"top1_guard_partial_corruption_report.json"); cheap=load(out/"D68_cheap_top1_regression_guard_report.json"); d68=load(out/"D68_loss_repair_preservation_report.json"); corr=load(out/"hard_correlated_joint_recall_report.json"); adv=load(out/"hard_adversarial_joint_recall_report.json"); ext=load(out/"external_required_watch_report.json"); abst=load(out/"indistinguishable_abstain_watch_report.json"); safety=load(out/"safety_margin_watch_report.json"); oracle=load(out/"oracle_distance_frontier_report.json"); support=load(out/"support_cost_frontier_report.json"); truth=load(out/"truth_leak_audit_report.json"); rust=load(out/"rust_invocation_report.json")
    if dec.get("decision") not in ALLOWED: fail(f"bad decision {dec.get('decision')}")
    if dec.get("decision") != "combined_low_cost_ood_repair_confirmed": fail(f"unexpected decision {dec.get('decision')}")
    if dec.get("next") != "D91_COMBINED_LOW_COST_OOD_SCALE_CONFIRM": fail(f"unexpected next {dec.get('next')}")
    if agg.get("failed_jobs") or dec.get("failed_jobs") or rust.get("failed_jobs"): fail("failed_jobs present")
    if agg.get("fallback_rows") != 0 or rust.get("fallback_rows") != 0: fail("fallback_rows nonzero")
    d89=man.get("d89_artifacts",{})
    if d89.get("decision") != "combined_low_cost_ood_plan_selected": fail(f"D89 decision mismatch {d89.get('decision')}")
    if d89.get("next") != "D90_COMBINED_LOW_COST_OOD_REPAIR_PROTOTYPE": fail(f"D89 next mismatch {d89.get('next')}")
    if d89.get("selected_repair_path") != "COMBINED_LOW_COST_OOD_REPAIR_PLAN": fail("D89 selected path mismatch")
    if d89.get("top1_guard_must_not_be_weakened") is not True: fail("D89 top1 invariant missing")
    if not man.get("d89_commit_present",{}).get("present") and not man.get("rerun",{}).get("rerun_attempted"): fail("D89 commit missing without rerun")
    if man.get("rerun",{}).get("rerun_attempted") and not man.get("rerun",{}).get("rerun_succeeded"): fail("D89 rerun failed")
    best=agg.get("best_fair_arm",{})
    required=["combined_low_cost_plus_ood_breakpoint","ood_support_distribution_shift_breakpoint","low_cost_pressure_breakpoint","combined_low_cost_plus_top1_ambiguity_breakpoint","top1_top2_sufficiency_ambiguity_breakpoint","exact_joint_accuracy","correlated_echo_accuracy","adversarial_distractor_accuracy","external_test_required_accuracy","false_confidence_rate","indistinguishable_abstain_rate","average_total_support_used","counter_support_used","distance_to_concrete_oracle_support","gap_reduction_vs_D73_bound","joint_counter_recall_on_joint_required_rows","external_recall_on_external_required_rows","wrong_concrete_counter_rate","weak_top1_top2_path_failure_rate","top1_top2_sufficient_false_joint_rate","D68_loss_repair_preservation_rate","routing_failure_rows","top1_guard_preserved","top1_guard_weakened","ablation_routing_failure_rows","ablation_D68_loss_repair_preservation_rate","min_seed_exact","min_seed_correlated","min_seed_adversarial","min_seed_external","fallback_rows","failed_jobs"]
    absent=[k for k in required if k not in best]
    if absent: fail(f"best missing {absent}")
    if best.get("arm") != "COMBINED_LOW_COST_OOD_REPAIR_COST_AWARE": fail(f"unexpected best {best.get('arm')}")
    gates={"combined_low_cost_plus_ood_breakpoint":best["combined_low_cost_plus_ood_breakpoint"]>=0.760,"combined_low_cost_plus_top1_ambiguity_breakpoint":best["combined_low_cost_plus_top1_ambiguity_breakpoint"]>=0.750,"low_cost_pressure_breakpoint":best["low_cost_pressure_breakpoint"]>=0.740,"ood_support_distribution_shift_breakpoint":best["ood_support_distribution_shift_breakpoint"]>=0.758,"exact_joint_accuracy":best["exact_joint_accuracy"]>=0.9990,"correlated_echo_accuracy":best["correlated_echo_accuracy"]>=0.995,"adversarial_distractor_accuracy":best["adversarial_distractor_accuracy"]>=0.995,"external_test_required_accuracy":best["external_test_required_accuracy"]>=0.995,"false_confidence_rate":best["false_confidence_rate"]<=0.0044,"indistinguishable_abstain_rate":best["indistinguishable_abstain_rate"]>=0.9948,"average_total_support_used":best["average_total_support_used"]<=6.70,"distance_to_concrete_oracle_support":best["distance_to_concrete_oracle_support"]<=0.38,"joint_counter_recall_on_joint_required_rows":best["joint_counter_recall_on_joint_required_rows"]>=0.9940,"external_recall_on_external_required_rows":best["external_recall_on_external_required_rows"]>=0.9957,"wrong_concrete_counter_rate":best["wrong_concrete_counter_rate"]<=0.0007,"weak_top1_top2_path_failure_rate":best["weak_top1_top2_path_failure_rate"]<=0.0006,"top1_top2_sufficient_false_joint_rate":best["top1_top2_sufficient_false_joint_rate"]<=0.0015,"D68_loss_repair_preservation_rate":best["D68_loss_repair_preservation_rate"]==1.0,"routing_failure_rows":best["routing_failure_rows"]==0,"top1_guard_preserved":best["top1_guard_preserved"] is True and preserve.get("top1_guard_preserved") is True,"top1_guard_weakened":best["top1_guard_weakened"] is False and preserve.get("top1_guard_weakened") is False,"top1_guard_ablation_remains_worse":ab.get("guard_ablation_worse") is True and (ab.get("ablation_metrics") or {}).get("routing_failure_rows",0)>best["routing_failure_rows"] and (ab.get("ablation_metrics") or {}).get("D68_loss_repair_preservation_rate",1)<best["D68_loss_repair_preservation_rate"],"rust_path_invoked":best["rust_path_invoked"] is True,"fallback_rows":best["fallback_rows"]==0,"failed_jobs":best["failed_jobs"]==[]}
    failed=[k for k,v in gates.items() if not v]
    if failed: fail(f"positive gates failed {failed}")
    for name,rep in [("repair",repair),("combo",combo),("ood",ood),("low",low),("combined_top1",ctop),("topamb",topamb),("preserve",preserve),("ablation",ab),("partial",partial),("cheap",cheap),("d68",d68),("corr",corr),("adv",adv),("external",ext),("abstain",abst),("safety",safety),("oracle",oracle),("support",support),("truth",truth),("rust",rust)]:
        if not rep.get("passed"): fail(f"{name} report failed")
    if truth.get("fair_arms_using_truth_label") or truth.get("fair_arms_using_support_regime_label") or truth.get("label_echo_fair_oracle_used") or truth.get("row_id_lookup_used") or truth.get("python_hash_used") or not truth.get("oracle_arms_reference_only"): fail("truth leak hard gate failed")
    print(json.dumps({"check":"passed","out":str(out),"decision":dec,"best_fair_arm":best.get("arm"),"combined_low_cost_plus_ood_breakpoint":best.get("combined_low_cost_plus_ood_breakpoint"),"ood_support_distribution_shift_breakpoint":best.get("ood_support_distribution_shift_breakpoint"),"failed_jobs":agg.get("failed_jobs")},indent=2))
if __name__ == "__main__": main()

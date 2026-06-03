#!/usr/bin/env python3
"""Validate D87 combined low-cost + top1 ambiguity scale-confirm artifacts."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
DEFAULT_OUT=Path("target/pilot_wave/d87_combined_low_cost_top1_ambiguity_scale_confirm")
REQ=["d86_upstream_manifest.json","combined_low_cost_top1_scale_report.json","combined_low_cost_top1_sweep_report.json","low_cost_pressure_sweep_report.json","top1_top2_ambiguity_sweep_report.json","top1_guard_preservation_report.json","top1_guard_ablation_report.json","top1_guard_partial_corruption_report.json","D68_cheap_top1_regression_guard_report.json","D68_loss_repair_preservation_report.json","hard_correlated_joint_recall_report.json","hard_adversarial_joint_recall_report.json","ood_support_shift_watch_report.json","external_required_watch_report.json","indistinguishable_abstain_watch_report.json","safety_margin_watch_report.json","oracle_distance_frontier_report.json","support_cost_frontier_report.json","truth_leak_audit_report.json","rust_invocation_report.json","aggregate_metrics.json","decision.json","summary.json","report.md"]
ALLOWED={"combined_low_cost_top1_ambiguity_scale_confirmed","combined_low_cost_top1_ambiguity_scale_high_cost","top1_guard_invariant_violation_under_scale","combined_low_cost_top1_ambiguity_scale_safety_regression","combined_low_cost_top1_ambiguity_scale_not_confirmed"}
def load(p:Path): return json.loads(p.read_text(encoding="utf-8"))
def fail(msg): print(json.dumps({"check":"failed","reason":msg},indent=2)); sys.exit(1)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default=str(DEFAULT_OUT)); args=ap.parse_args(); out=Path(args.out)
    missing=[n for n in REQ if not (out/n).exists()]
    if missing: fail(f"missing reports {missing}")
    man=load(out/"d86_upstream_manifest.json"); scale=load(out/"combined_low_cost_top1_scale_report.json"); combo=load(out/"combined_low_cost_top1_sweep_report.json"); low=load(out/"low_cost_pressure_sweep_report.json"); top1s=load(out/"top1_top2_ambiguity_sweep_report.json"); preserve=load(out/"top1_guard_preservation_report.json"); ab=load(out/"top1_guard_ablation_report.json"); partial=load(out/"top1_guard_partial_corruption_report.json"); d68=load(out/"D68_loss_repair_preservation_report.json"); cheap=load(out/"D68_cheap_top1_regression_guard_report.json"); corr=load(out/"hard_correlated_joint_recall_report.json"); adv=load(out/"hard_adversarial_joint_recall_report.json"); ood=load(out/"ood_support_shift_watch_report.json"); ext=load(out/"external_required_watch_report.json"); abst=load(out/"indistinguishable_abstain_watch_report.json"); safety=load(out/"safety_margin_watch_report.json"); oracle=load(out/"oracle_distance_frontier_report.json"); supp=load(out/"support_cost_frontier_report.json"); truth=load(out/"truth_leak_audit_report.json"); rust=load(out/"rust_invocation_report.json"); agg=load(out/"aggregate_metrics.json"); dec=load(out/"decision.json")
    if dec.get("decision") not in ALLOWED: fail(f"bad decision {dec.get('decision')}")
    if agg.get("failed_jobs") or dec.get("failed_jobs") or rust.get("failed_jobs"): fail("failed jobs present")
    if agg.get("fallback_rows")!=0 or rust.get("fallback_rows")!=0: fail("fallback rows nonzero")
    d86=man.get("d86_artifacts",{})
    if d86.get("decision")!="combined_low_cost_top1_ambiguity_repair_confirmed": fail(f"D86 decision mismatch {d86.get('decision')}")
    if d86.get("next")!="D87_COMBINED_LOW_COST_TOP1_AMBIGUITY_SCALE_CONFIRM": fail(f"D86 next mismatch {d86.get('next')}")
    if d86.get("best_arm")!="COMBINED_LOW_COST_TOP1_REPAIR_COST_AWARE": fail(f"D86 best mismatch {d86.get('best_arm')}")
    if not man.get("d86_commit_present",{}).get("present") and not man.get("rerun",{}).get("rerun_attempted"): fail("D86 commit missing without rerun")
    if man.get("rerun",{}).get("rerun_attempted") and not man.get("rerun",{}).get("rerun_succeeded"): fail("D86 rerun failed")
    best=agg.get("best_fair_arm",{})
    if best.get("arm")!="D86_COMBINED_REPAIR_COST_AWARE_REPLAY": fail(f"unexpected best {best.get('arm')}")
    req=["combined_low_cost_plus_top1_ambiguity_breakpoint","low_cost_pressure_breakpoint","top1_top2_sufficiency_ambiguity_breakpoint","exact_joint_accuracy","correlated_echo_accuracy","adversarial_distractor_accuracy","external_test_required_accuracy","false_confidence_rate","indistinguishable_abstain_rate","average_total_support_used","counter_support_used","distance_to_concrete_oracle_support","gap_reduction_vs_D73_bound","joint_counter_recall_on_joint_required_rows","external_recall_on_external_required_rows","wrong_concrete_counter_rate","weak_top1_top2_path_failure_rate","top1_top2_sufficient_false_joint_rate","D68_loss_repair_preservation_rate","routing_failure_rows","top1_guard_preserved","top1_guard_weakened","ablation_routing_failure_rows","ablation_D68_loss_repair_preservation_rate","min_seed_exact","min_seed_correlated","min_seed_adversarial","min_seed_external","fallback_rows","failed_jobs"]
    absent=[k for k in req if k not in best]
    if absent: fail(f"best missing {absent}")
    gates=dec.get("positive_gates",{})
    failed=[k for k,v in gates.items() if not v]
    if dec.get("decision")=="combined_low_cost_top1_ambiguity_scale_confirmed" and failed: fail(f"positive gates failed {failed}")
    if best["combined_low_cost_plus_top1_ambiguity_breakpoint"]<0.750 or best["low_cost_pressure_breakpoint"]<0.740 or best["top1_top2_sufficiency_ambiguity_breakpoint"]<0.742: fail("breakpoint gates failed")
    if best["average_total_support_used"]>6.70 or best["distance_to_concrete_oracle_support"]>0.38 or best["gap_reduction_vs_D73_bound"]<0.1500: fail("support/oracle gates failed")
    if not best["top1_guard_preserved"] or best["top1_guard_weakened"] or best["D68_loss_repair_preservation_rate"]!=1.0 or best["routing_failure_rows"]!=0: fail("top1/D68 routing gates failed")
    for name,rep in [("scale",scale),("combo",combo),("low",low),("top1",top1s),("preserve",preserve),("ablation",ab),("partial",partial),("cheap",cheap),("d68",d68),("corr",corr),("adv",adv),("ood",ood),("external",ext),("abstain",abst),("safety",safety),("oracle",oracle),("support",supp),("truth",truth),("rust",rust)]:
        if not rep.get("passed"): fail(f"{name} report failed")
    if truth.get("fair_arms_using_truth_label") or truth.get("fair_arms_using_support_regime_label") or truth.get("label_echo_fair_oracle_used") or truth.get("row_id_lookup_used") or truth.get("python_hash_used") or not truth.get("oracle_arms_reference_only"): fail("truth leak hard gate failed")
    print(json.dumps({"check":"passed","out":str(out),"decision":dec,"best_fair_arm":best.get("arm"),"combined_breakpoint":best.get("combined_low_cost_plus_top1_ambiguity_breakpoint"),"low_cost_breakpoint":best.get("low_cost_pressure_breakpoint"),"top1_ambiguity_breakpoint":best.get("top1_top2_sufficiency_ambiguity_breakpoint"),"failed_jobs":agg.get("failed_jobs")},indent=2))
if __name__=="__main__": main()

#!/usr/bin/env python3
"""Validate D96 next-breakpoint and train-loop bridge artifacts."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
REPORTS=["d95_upstream_manifest.json","d96_breakpoint_rank_report.json","d96_next_breakpoint_stress_map_report.json","d96_min_seed_tail_map_report.json","d96_ood_support_shift_tail_report.json","d96_joint_required_boundary_tail_report.json","d96_top1_top2_ambiguity_tail_report.json","d96_low_cost_pressure_tail_report.json","d96_external_required_tail_report.json","d96_correlated_echo_distractor_tail_report.json","d96_adversarial_counter_tail_report.json","d96_indistinguishable_abstain_tail_report.json","d96_oracle_distance_frontier_report.json","d96_support_cost_frontier_report.json","d96_feature_signal_audit_report.json","d96_trainable_surrogate_feasibility_report.json","d96_routing_label_distillation_report.json","d96_top1_guard_preservation_report.json","d96_top1_guard_ablation_report.json","d96_partial_corruption_control_report.json","d96_D68_cheap_top1_regression_guard_report.json","d96_D68_loss_repair_preservation_report.json","d96_truth_leak_audit_report.json","d96_row_id_hash_lookup_audit_report.json","d96_oracle_isolation_report.json","d96_rust_invocation_report.json","d96_report_schema_consistency_report.json","aggregate_metrics.json","decision.json","summary.json","report.md"]
ALLOWED={"d96_breakpoint_map_complete_train_loop_bridge_ready","d96_breakpoint_map_complete_surrogate_not_ready","d96_tail_risk_breakpoint_identified","d95_preservation_regression_detected","top1_guard_invariant_violation","d68_regression_detected","truth_leak_or_oracle_contamination_detected","row_id_or_hash_shortcut_detected","d96_invalid_metric_or_report_inconsistency","d96_invalid_or_incomplete_run"}
def load(p:Path): return json.loads(p.read_text(encoding="utf-8"))
def fail(m): print(f"D96 check failed: {m}",file=sys.stderr); sys.exit(1)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",default="target/pilot_wave/d96_next_breakpoint_and_train_loop_bridge_plan"); args=p.parse_args(); out=Path(args.out)
    missing=[r for r in REPORTS if not (out/r).exists()]
    if missing: fail(f"missing reports {missing}")
    man=load(out/"d95_upstream_manifest.json"); agg=load(out/"aggregate_metrics.json"); dec=load(out/"decision.json"); summ=load(out/"summary.json"); rank=load(out/"d96_breakpoint_rank_report.json"); stress=load(out/"d96_next_breakpoint_stress_map_report.json"); tail=load(out/"d96_min_seed_tail_map_report.json"); feature=load(out/"d96_feature_signal_audit_report.json"); surrogate=load(out/"d96_trainable_surrogate_feasibility_report.json"); distill=load(out/"d96_routing_label_distillation_report.json"); preserve=load(out/"d96_top1_guard_preservation_report.json"); ab=load(out/"d96_top1_guard_ablation_report.json"); partial=load(out/"d96_partial_corruption_control_report.json"); d68=load(out/"d96_D68_loss_repair_preservation_report.json"); truth=load(out/"d96_truth_leak_audit_report.json"); rowhash=load(out/"d96_row_id_hash_lookup_audit_report.json"); oracle=load(out/"d96_oracle_isolation_report.json"); rust=load(out/"d96_rust_invocation_report.json"); schema=load(out/"d96_report_schema_consistency_report.json"); safety_reports=["d96_ood_support_shift_tail_report.json","d96_joint_required_boundary_tail_report.json","d96_top1_top2_ambiguity_tail_report.json","d96_low_cost_pressure_tail_report.json","d96_external_required_tail_report.json","d96_correlated_echo_distractor_tail_report.json","d96_adversarial_counter_tail_report.json","d96_indistinguishable_abstain_tail_report.json","d96_oracle_distance_frontier_report.json","d96_support_cost_frontier_report.json","d96_D68_cheap_top1_regression_guard_report.json"]
    for r in safety_reports:
        if not load(out/r).get("passed"): fail(f"{r} did not pass")
    if dec.get("decision") not in ALLOWED: fail(f"bad decision {dec.get('decision')}")
    if dec.get("decision")!="d96_breakpoint_map_complete_train_loop_bridge_ready": fail(f"unexpected decision {dec.get('decision')}")
    if dec.get("next")!="D97_MECHANISM_FEATURE_AUDIT_AND_SURROGATE_TRAINING_PROTOTYPE": fail(f"unexpected next {dec.get('next')}")
    if dec.get("failed_jobs") or agg.get("failed_jobs") or rust.get("failed_jobs"): fail("failed_jobs present")
    if dec.get("fallback_rows")!=0 or agg.get("fallback_rows")!=0 or rust.get("fallback_rows")!=0: fail("fallback rows nonzero")
    d95=man.get("d95_artifacts",{})
    if man.get("requested_d95_commit")!="1ecc5694f404da52ee1793a1fda3e18ca3bef045": fail("requested D95 commit mismatch")
    if not man.get("commit_present") and not (man.get("restore_or_rerun_attempted") and man.get("restore_or_rerun_succeeded")): fail("D95 commit missing without successful restore/rerun")
    if not man.get("artifact_present") or man.get("validation_status")!="passed": fail("D95 artifact/validation failed")
    if d95.get("decision")!="combined_ood_joint_boundary_scale_confirmed" or d95.get("next")!="D96_NEXT_BREAKPOINT_OR_GENERALIZATION_PLAN": fail("D95 decision handoff mismatch")
    if d95.get("best_arm")!="COMBINED_OOD_JOINT_BOUNDARY_REPAIR_COST_AWARE_SCALE": fail("D95 best arm mismatch")
    m=agg.get("metrics",{}); gates=agg.get("positive_gates",{})
    failed=[k for k,v in gates.items() if not v]
    if failed: fail(f"positive gates failed {failed}")
    if m.get("next_breakpoint_name")!="COMBINED_LOW_COST_OOD_TOP1_AMBIGUITY_TAIL": fail("unexpected next breakpoint")
    if m.get("next_breakpoint_rank_confidence",0)<0.75 or m.get("breakpoint_rank_stability_across_seeds",0)<0.70: fail("rank confidence/stability too low")
    if m.get("stress_tail_false_confidence_rate",1)>0.0046 or m.get("stress_tail_routing_failure_rows")!=0: fail("stress tail gate failed")
    if m.get("combined_ood_joint_boundary_breakpoint",0)<0.755 or m.get("min_seed_combined_ood_joint_boundary_breakpoint",0)<0.752: fail("D95 preservation breakpoint failed")
    if m.get("top1_guard_weakened") is True or preserve.get("top1_guard_weakened") is True: fail("top1 guard weakened")
    if d68.get("D68_loss_repair_preservation_rate")!=1.0: fail("D68 regression")
    if not ab.get("guard_ablation_worse") or not partial.get("partial_corruption_worse"): fail("guard controls not worse")
    if truth.get("fair_arms_using_truth_label") or truth.get("fair_arms_using_support_regime_label") or truth.get("label_echo_fair_oracle_used") or not truth.get("oracle_arms_reference_only") or not truth.get("passed"): fail("truth leak audit failed")
    if rowhash.get("row_id_lookup_usable_by_fair_arms") or rowhash.get("python_hash_lookup_usable_by_fair_arms") or not rowhash.get("passed"): fail("row/hash shortcut failed")
    if not oracle.get("oracle_isolation_passed") or not oracle.get("passed"): fail("oracle isolation failed")
    if feature.get("feature_signal_count",0)<8 or feature.get("usable_feature_signal_count",0)<6 or feature.get("forbidden_feature_detected"): fail("feature audit failed")
    if not distill.get("route_distillation_target_defined") or distill.get("route_distillation_label_leak_risk"): fail("distillation audit failed")
    if not surrogate.get("trainable_surrogate_ready") or surrogate.get("surrogate_min_seed_accuracy",0)<0.990 or surrogate.get("surrogate_false_confidence_rate",1)>0.0046 or surrogate.get("surrogate_overfit_gap",1)>0.015: fail("surrogate readiness failed")
    if m.get("label_shuffle_sentinel_accuracy",1)>0.55 or m.get("regime_label_leak_sentinel_accuracy",1)>0.56 or m.get("row_id_lookup_sentinel_accuracy",1)>0.55 or m.get("python_hash_lookup_sentinel_accuracy",1)>0.55: fail("sentinel leakage failed")
    if not schema.get("report_schema_consistency_passed") or not schema.get("metric_crosscheck_passed") or not schema.get("deterministic_replay_passed"): fail("schema consistency failed")
    if summ.get("decision")!=dec.get("decision") or summ.get("next")!=dec.get("next") or summ.get("next_breakpoint_name")!=m.get("next_breakpoint_name"): fail("summary/decision mismatch")
    print(json.dumps({"check":"passed","out":str(out),"decision":dec,"next_breakpoint_name":m.get("next_breakpoint_name"),"next_breakpoint_score":m.get("next_breakpoint_score"),"trainable_surrogate_ready":m.get("trainable_surrogate_ready"),"failed_jobs":m.get("failed_jobs")},indent=2))
if __name__=="__main__": main()

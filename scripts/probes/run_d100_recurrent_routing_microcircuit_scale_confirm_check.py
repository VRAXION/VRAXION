#!/usr/bin/env python3
"""Validate D100 recurrent routing microcircuit scale confirmation artifacts."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
TASK="D100_RECURRENT_ROUTING_MICROCIRCUIT_SCALE_CONFIRM"
REPORTS=["d99_upstream_manifest.json","d100_scale_report.json","d100_recurrent_scale_eval_report.json","d100_recurrent_loop_instrumentation_report.json","d100_recurrent_convergence_report.json","d100_recurrent_halting_report.json","d100_recurrent_oscillation_report.json","d100_recurrent_state_stability_report.json","d100_recurrent_state_noise_report.json","d100_recurrent_state_reset_report.json","d100_recurrent_state_drift_report.json","d100_recurrent_state_saturation_report.json","d100_recurrent_step_budget_report.json","d100_recurrent_early_halt_report.json","d100_recurrent_delayed_halt_report.json","d100_recurrent_usefulness_report.json","d100_one_step_ablation_report.json","d100_no_state_control_report.json","d100_shuffled_state_control_report.json","d100_low_cost_ood_top1_ambiguity_tail_report.json","d100_combined_ood_joint_boundary_carryover_report.json","d100_ood_scale_carryover_report.json","d100_stress_tail_carryover_report.json","d100_min_seed_worst_seed_report.json","d100_feature_noise_carryover_report.json","d100_feature_dropout_carryover_report.json","d100_calibration_pressure_carryover_report.json","d100_confidence_false_positive_report.json","d100_top1_guard_preservation_report.json","d100_top1_guard_ablation_control_report.json","d100_top1_guard_partial_corruption_control_report.json","d100_D68_preservation_report.json","d100_safety_margin_report.json","d100_oracle_distance_frontier_report.json","d100_support_cost_frontier_report.json","d100_label_shuffle_sentinel_report.json","d100_regime_label_leak_sentinel_report.json","d100_row_id_lookup_sentinel_report.json","d100_python_hash_lookup_sentinel_report.json","d100_file_order_artifact_sentinel_report.json","d100_seed_id_shortcut_sentinel_report.json","d100_hidden_state_label_leak_sentinel_report.json","d100_hidden_state_row_lookup_sentinel_report.json","d100_halt_step_shortcut_sentinel_report.json","d100_step_count_shortcut_sentinel_report.json","d100_split_integrity_report.json","d100_overfit_memorization_report.json","d100_feature_importance_stability_report.json","d100_negative_controls_report.json","d100_truth_leak_oracle_isolation_report.json","d100_rust_invocation_report.json","d100_report_schema_metric_crosscheck_report.json","d100_deterministic_replay_report.json","aggregate_metrics.json","decision.json","summary.json","report.md"]
D99_COMMIT="bcc15b5599d1e48f83f68d4939043ec2e13e5c82"

def load(path: Path): return json.loads(path.read_text(encoding="utf-8"))
def fail(msg: str): print(f"D100 check failed: {msg}",file=sys.stderr); sys.exit(1)
def expect(cond: bool, msg: str):
    if not cond: fail(msg)

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--out",type=Path,required=True); args=ap.parse_args(); out=args.out
    missing=[r for r in REPORTS if not (out/r).exists()]; expect(not missing,f"missing reports: {missing}")
    aggregate=load(out/"aggregate_metrics.json"); decision=load(out/"decision.json"); summary=load(out/"summary.json"); manifest=load(out/"d99_upstream_manifest.json"); m=aggregate["metrics"]; gates=aggregate["positive_gates"]
    expect(aggregate["task"]==TASK,"aggregate task mismatch")
    expect(manifest["requested_d99_commit"]==D99_COMMIT,"D99 commit mismatch")
    expect(manifest["artifact_present"] is True,"D99 artifact missing")
    expect(manifest["validation_status"]=="passed","D99 validation did not pass")
    expect(manifest["replayed_decision"]=="d99_recurrent_routing_microcircuit_prototype_confirmed","D99 decision mismatch")
    expect(manifest["replayed_next"]==TASK,"D99 next mismatch")
    expect(manifest["replayed_best_fair_recurrent"]=="D99_RECURRENT_HALTING_CONFIDENCE_FAIR","D99 best recurrent mismatch")
    if not manifest.get("commit_present"):
        expect(manifest.get("restore_or_rerun_attempted") and manifest.get("restore_or_rerun_succeeded"),"missing D99 commit requires successful restore/rerun")
    expect(decision["decision"]=="d100_recurrent_routing_microcircuit_scale_confirmed","unexpected decision")
    expect(decision["next"]=="D101_AUTO_ANNEAL_AND_SPARSE_STABILIZATION_PREP","unexpected next")
    expect(decision["best_fair_recurrent_arm"]=="D99_RECURRENT_HALTING_CONFIDENCE_FAIR_SCALE","unexpected arm")
    expect(decision["failed_jobs"]==[],"failed_jobs not empty")
    expect(decision["fallback_rows"]==0,"fallback rows nonzero")
    expect(aggregate["scale"]["scale_reduced"] is False,"scale was reduced")
    expect(aggregate["scale"]["all_required_stress_modes_executed"] is True,"stress modes incomplete")
    expect(all(gates.values()),f"positive gates failed: {[k for k,v in gates.items() if not v]}")
    for report in REPORTS:
        if report in {"d99_upstream_manifest.json","aggregate_metrics.json","decision.json","summary.json","report.md"}: continue
        expect(load(out/report).get("passed") is True,f"{report} not passed")
    thresholds={
        "recurrent_scale_test_accuracy":0.9940,"recurrent_scale_ood_accuracy":0.9915,"recurrent_scale_stress_accuracy":0.9910,"recurrent_scale_min_seed_accuracy":0.9905,"recurrent_scale_worst_seed_accuracy":0.9895,"recurrent_scale_convergence_rate":0.998,"recurrent_scale_state_stability_score":0.80,"recurrent_scale_loop_usefulness_score":0.70,"recurrent_scale_loop_usefulness_on_tail_score":0.70,"recurrent_scale_low_cost_ood_top1_tail_score":0.746,"recurrent_scale_min_seed_low_cost_ood_top1_tail_score":0.742,"recurrent_scale_combined_ood_joint_boundary_breakpoint":0.755,"recurrent_scale_min_seed_combined_ood_joint_boundary_breakpoint":0.752,"recurrent_scale_external_required_accuracy":0.995,"recurrent_scale_joint_required_accuracy":0.994,"recurrent_scale_indistinguishable_abstain_rate":0.9948,"recurrent_scale_state_noise_1pct_accuracy":0.991,"recurrent_scale_state_noise_3pct_accuracy":0.988,"recurrent_scale_state_noise_5pct_accuracy":0.985,"recurrent_scale_state_reset_recovery_accuracy":0.985,
    }
    for key, threshold in thresholds.items(): expect(m[key]>=threshold,f"{key} below threshold")
    ceilings={
        "recurrent_scale_overfit_gap":0.015,"recurrent_scale_false_confidence_rate":0.0046,"recurrent_scale_non_convergence_rate":0.001,"recurrent_scale_oscillation_rate":0.001,"recurrent_scale_halting_false_positive_rate":0.0046,"recurrent_scale_distance_to_symbolic_teacher":0.025,"recurrent_scale_distance_to_d99_recurrent":0.025,"recurrent_scale_distance_to_d98_surrogate":0.025,"recurrent_scale_distance_to_concrete_oracle_support":0.385,"recurrent_scale_average_support_used":6.75,"recurrent_scale_average_steps_used":5.0,"recurrent_scale_max_steps_used":8,"recurrent_scale_wrong_concrete_counter_rate":0.0007,"recurrent_scale_weak_top1_top2_path_failure_rate":0.0006,"recurrent_scale_top1_top2_sufficient_false_joint_rate":0.0015,"recurrent_scale_worst_tail_failure_rate":0.003,"recurrent_scale_mixed_tail_compound_failure_rate":0.004,"recurrent_scale_false_positive_confidence_tail_rate":0.0048,"recurrent_scale_state_drift_failure_rate":0.003,"recurrent_scale_state_saturation_failure_rate":0.003,"recurrent_scale_early_halt_failure_rate":0.003,"recurrent_scale_delayed_halt_failure_rate":0.003,"recurrent_scale_step_budget_failure_rate":0.003,"memorization_risk_score":0.10,
    }
    for key, ceiling in ceilings.items(): expect(m[key]<=ceiling,f"{key} above ceiling")
    expect(m["recurrent_scale_routing_failure_rows"]==0,"routing failures")
    expect(m["recurrent_scale_D68_preservation_rate"]==1.0,"D68 regression")
    expect(m["recurrent_scale_top1_guard_preserved"] and not m["recurrent_scale_top1_guard_weakened"],"top1 guard failure")
    expect(m["recurrent_scale_truth_leak_audit_passed"] and m["recurrent_scale_oracle_isolation_passed"],"truth/oracle isolation failure")
    for key in ["label_shuffle_sentinel_accuracy","regime_label_leak_sentinel_accuracy","row_id_lookup_sentinel_accuracy","python_hash_lookup_sentinel_accuracy","file_order_artifact_sentinel_accuracy","seed_id_shortcut_sentinel_accuracy","hidden_state_label_leak_sentinel_accuracy","hidden_state_row_lookup_sentinel_accuracy","halt_step_shortcut_sentinel_accuracy","step_count_shortcut_sentinel_accuracy"]:
        expect(m[key]<=0.56,f"{key} shortcut too high")
    expect(not m["forbidden_feature_detected"],"forbidden features detected")
    expect(m["route_distillation_target_defined"] and not m["route_distillation_label_leak_risk"],"route target leak risk")
    expect(m["split_integrity_passed"] and not m["train_test_ood_contamination_detected"],"split integrity failure")
    expect(m["sentinel_collapse_passed"],"sentinel collapse failed")
    expect(m["rust_path_invoked"] and m["fallback_rows"]==0 and m["failed_jobs"]==[],"rust/fallback/failed job failure")
    expect(summary["decision"]==decision["decision"] and summary["next"]==decision["next"],"summary decision mismatch")
    print(json.dumps({"status":"passed","artifact_path":str(out),"decision":decision["decision"],"next":decision["next"]},indent=2,sort_keys=True))
if __name__=="__main__": main()

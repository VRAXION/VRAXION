# D83 Low-Cost Pressure Repair Scale Confirm Contract

## Purpose

D83 scale-confirms the D82 `LOW_COST_PRESSURE_REPAIR_COST_AWARE` repair while preserving the top1/top2 sufficiency guard in controlled symbolic ECF/IPF joint formula discovery. It does not add a new broad architecture mechanism, does not integrate unrelated external-routing changes, and does not weaken or bypass the top1/top2 sufficiency guard.

## Phase 0 upstream audit

The runner must verify the current branch and HEAD, check whether D82 commit `4993600e328d7a9963032bcb9455f2c2ef60660b` is present and an ancestor of `HEAD`, restore/rerun D82 artifacts if missing, and write `d82_upstream_manifest.json`. D83 must not silently assume D82 was pushed.

## D82 handoff

D82 confirmed:

- `decision=low_cost_pressure_repair_confirmed`
- `next=D83_LOW_COST_PRESSURE_REPAIR_SCALE_CONFIRM`
- `best_arm=LOW_COST_PRESSURE_REPAIR_COST_AWARE`
- `low_cost_pressure_breakpoint=0.752`
- top1 guard preserved, not weakened, and ablation remained worse.

D83 must scale-confirm this repair, keep the low-cost breakpoint at or above `0.74`, preserve D68 loss repair, retain joint/external recall and safety margins, keep Rust invocation visible, and report fallback/failed-job invariants.

## Tracks

1. `D82_REPLAY`
2. `LARGER_SEED_SCALE`
3. `LOW_COST_PRESSURE_SWEEP`
4. `TOP1_GUARD_PRESERVATION`
5. `TOP1_GUARD_ABLATION_CONTROL`
6. `TOP1_TOP2_SUFFICIENCY_AMBIGUITY_WATCH`
7. `OOD_SUPPORT_SHIFT_WATCH`
8. `JOINT_REQUIRED_NEAR_BOUNDARY_WATCH`
9. `HARD_CORRELATED_JOINT_RECALL`
10. `HARD_ADVERSARIAL_JOINT_RECALL`
11. `EXTERNAL_REQUIRED_WATCH`
12. `INDISTINGUISHABLE_ABSTAIN_WATCH`
13. `D68_CHEAP_TOP1_REGRESSION_GUARD`
14. `SAFETY_MARGIN_WATCH`
15. `ORACLE_DISTANCE_FRONTIER`

## Arms

1. `D82_LOW_COST_REPAIR_COST_AWARE_REPLAY`
2. `D82_LOW_COST_REPAIR_HIGH_RECALL`
3. `D82_LOW_COST_REPAIR_LOW_COST`
4. `D82_LOW_COST_REPAIR_BALANCED`
5. `D79_INTEGRATED_ROUTER_REPLAY`
6. `TOP1_GUARD_ABLATION_CONTROL`
7. `TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL`
8. `LOW_COST_ONLY_CONTROL`
9. `RANDOM_ROUTER_CONTROL`
10. `NEVER_JOINT_CONTROL`
11. `ALWAYS_JOINT_CONTROL`
12. `CONCRETE_ORACLE_REFERENCE_ONLY`
13. `TRUTH_LEAK_SENTINEL_REFERENCE_ONLY`

## Required reports

Artifacts are written under `target/pilot_wave/d83_low_cost_pressure_repair_scale_confirm/`:

- `d82_upstream_manifest.json`
- `low_cost_pressure_scale_report.json`
- `low_cost_pressure_sweep_report.json`
- `top1_guard_preservation_report.json`
- `top1_guard_ablation_report.json`
- `D68_cheap_top1_regression_guard_report.json`
- `D68_loss_repair_preservation_report.json`
- `top1_top2_ambiguity_watch_report.json`
- `joint_boundary_watch_report.json`
- `ood_support_shift_watch_report.json`
- `external_required_watch_report.json`
- `indistinguishable_abstain_watch_report.json`
- `safety_margin_watch_report.json`
- `support_cost_frontier_report.json`
- `oracle_distance_frontier_report.json`
- `truth_leak_audit_report.json`
- `rust_invocation_report.json`
- `aggregate_metrics.json`
- `decision.json`
- `summary.json`
- `report.md`

## Positive gate

The scaled D82 repair must satisfy:

- `low_cost_pressure_breakpoint >= 0.74`
- `exact_joint_accuracy >= 0.9990`
- `correlated_echo_accuracy >= 0.995`
- `adversarial_distractor_accuracy >= 0.995`
- `external_test_required_accuracy >= 0.995`
- `false_confidence_rate <= 0.0044`
- `indistinguishable_abstain_rate >= 0.9948`
- `average_total_support_used <= 6.70`
- `distance_to_concrete_oracle_support <= 0.38`
- `gap_reduction_vs_D73_bound >= 0.1500`
- `joint_counter_recall_on_joint_required_rows >= 0.9940`
- `external_recall_on_external_required_rows >= 0.9957`
- `wrong_concrete_counter_rate <= 0.0007`
- `weak_top1_top2_path_failure_rate <= 0.0006`
- `top1_top2_sufficient_false_joint_rate <= 0.0015`
- `D68_loss_repair_preservation_rate = 1.0`
- `routing_failure_rows = 0`
- top1 guard preserved and ablation remains worse
- `rust_path_invoked=true`
- `fallback_rows=0`
- `failed_jobs=[]`

## Decisions

- Passing scale confirm: `decision=low_cost_pressure_repair_scale_confirmed`, `next=D84_LOW_COST_PRESSURE_REPAIR_STRESS_MAP`.
- Passing but high support: `decision=low_cost_pressure_repair_scale_high_cost`, `next=D83C_COST_REPAIR`.
- Top1 guard regression: `decision=top1_guard_regression_under_low_cost_scale`, `next=D83G_TOP1_GUARD_REPAIR`.
- Safety/routing regression: `decision=low_cost_pressure_repair_scale_safety_regression`, `next=D83S_SAFETY_ROUTING_REPAIR`.
- Failed scale confirm: `decision=low_cost_pressure_repair_scale_not_confirmed`, `next=D83_REPAIR`.

## Boundary

D83 only scale-confirms low-cost pressure repair while preserving top1 sufficiency guard in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.

# D82 Low-Cost Pressure Repair With Top1 Guard Contract

## Purpose

D82 repairs low-cost pressure behavior while preserving the top1/top2 sufficiency guard in controlled symbolic ECF/IPF joint formula discovery. It does not change the broad architecture, does not integrate unrelated external-routing changes, and does not weaken or bypass the top1/top2 sufficiency guard.

## Phase 0 upstream audit

The runner must verify the current branch and HEAD, check whether D81 commit `edcafde6582e67e13fa95908e6103429799ce7ff` is present and an ancestor of `HEAD`, restore/rerun D81 artifacts if missing, and write `d81_upstream_manifest.json`. D82 must not silently assume D81 was pushed.

## D81 handoff

D81 selected:

- `decision=low_cost_pressure_repair_plan_selected`
- `next=D82_LOW_COST_PRESSURE_REPAIR_WITH_TOP1_GUARD`
- `selected_repair_path=LOW_COST_PRESSURE_REPAIR_PLAN`

D81 classified the top1 guard as a hard invariant and hardening target, not a disposable cost knob. D82 must improve `LOW_COST_PRESSURE` above the D80 breakpoint `0.70` while preserving D68 loss repair, joint/external recall, safety margins, Rust invocation, fallback, and failed-job invariants.

## Tracks

1. `D81_REPLAY`
2. `LOW_COST_PRESSURE_SWEEP`
3. `TOP1_GUARD_PRESERVATION`
4. `TOP1_TOP2_SUFFICIENCY_AMBIGUITY_WATCH`
5. `JOINT_REQUIRED_NEAR_BOUNDARY_WATCH`
6. `HARD_CORRELATED_JOINT_RECALL`
7. `HARD_ADVERSARIAL_JOINT_RECALL`
8. `OOD_SUPPORT_SHIFT_WATCH`
9. `EXTERNAL_REQUIRED_WATCH`
10. `INDISTINGUISHABLE_ABSTAIN_WATCH`
11. `D68_CHEAP_TOP1_REGRESSION_GUARD`
12. `SAFETY_MARGIN_WATCH`
13. `ORACLE_DISTANCE_FRONTIER`

## Arms

1. `D79_INTEGRATED_ROUTER_REPLAY`
2. `D80_STRESS_BASELINE_REPLAY`
3. `LOW_COST_PRESSURE_REPAIR_BASE`
4. `LOW_COST_PRESSURE_REPAIR_WITH_TOP1_GUARD`
5. `LOW_COST_PRESSURE_REPAIR_COST_AWARE`
6. `LOW_COST_PRESSURE_REPAIR_HIGH_RECALL`
7. `LOW_COST_PRESSURE_REPAIR_BALANCED`
8. `TOP1_GUARD_ABLATION_CONTROL`
9. `TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL`
10. `LOW_COST_ONLY_CONTROL`
11. `RANDOM_ROUTER_CONTROL`
12. `NEVER_JOINT_CONTROL`
13. `ALWAYS_JOINT_CONTROL`
14. `CONCRETE_ORACLE_REFERENCE_ONLY`
15. `TRUTH_LEAK_SENTINEL_REFERENCE_ONLY`

## Required reports

Artifacts are written under `target/pilot_wave/d82_low_cost_pressure_repair_with_top1_guard/`:

- `d81_upstream_manifest.json`
- `low_cost_pressure_repair_report.json`
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

The best fair D82 arm must satisfy:

- `low_cost_pressure_breakpoint >= 0.74`
- top1 guard ablation remains worse
- `exact_joint_accuracy >= 0.9990`
- `correlated_echo_accuracy >= 0.995`
- `adversarial_distractor_accuracy >= 0.995`
- `external_test_required_accuracy >= 0.995`
- `false_confidence_rate <= 0.0044`
- `indistinguishable_abstain_rate >= 0.9948`
- `average_total_support_used <= D79 support + 0.02`
- `distance_to_concrete_oracle_support <= D79 distance + 0.02`
- `joint_counter_recall_on_joint_required_rows >= 0.9940`
- `external_recall_on_external_required_rows >= 0.9957`
- `wrong_concrete_counter_rate <= 0.0007`
- `weak_top1_top2_path_failure_rate <= 0.0006`
- `top1_top2_sufficient_false_joint_rate <= 0.0015`
- `D68_loss_repair_preservation_rate = 1.0`
- `routing_failure_rows = 0`
- `rust_path_invoked=true`
- `fallback_rows=0`
- `failed_jobs=[]`

## Decisions

- Passing repair: `decision=low_cost_pressure_repair_confirmed`, `next=D83_LOW_COST_PRESSURE_REPAIR_SCALE_CONFIRM`.
- Improved low-cost but safety/routing regression: `decision=low_cost_pressure_repair_safety_regression`, `next=D82S_SAFETY_ROUTING_REPAIR`.
- Top1 guard ablation not worse: `decision=top1_guard_invariant_not_preserved`, `next=D82G_TOP1_GUARD_REPAIR`.
- Low-cost pressure not improved: `decision=low_cost_pressure_repair_not_confirmed`, `next=D82_REPAIR`.

## Boundary

D82 only repairs low-cost pressure while preserving the top1 sufficiency guard in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.

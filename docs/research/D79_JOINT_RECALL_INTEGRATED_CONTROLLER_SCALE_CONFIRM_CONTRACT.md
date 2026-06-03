# D79 Joint-Recall Integrated Controller Scale Confirm Contract

## Purpose

D79 scale-confirms the integrated `JointRecallCostAwareCounterRouter` inside the Rust sparse ECF/IPF controller path. It is a controlled symbolic ECF/IPF joint formula discovery scale-confirmation only; it does not add a broad architecture claim.

## Phase 0 upstream audit

The runner must verify the current branch and HEAD, check whether D78 commit `d2b9dda92ee1217bc6816b68ad6e94e30f976917` is present and an ancestor of `HEAD`, and write `d78_upstream_manifest.json`. If the requested D78 commit or artifacts are missing, the runner must restore/rerun D78 before D79 and record that status explicitly. D79 must not silently assume D78 was pushed.

## D78 handoff reference

D78 confirmed the integrated controller prototype with:

- `decision=joint_recall_integrated_controller_prototype_confirmed`
- `next=D79_JOINT_RECALL_INTEGRATED_CONTROLLER_SCALE_CONFIRM`
- `best_arm=INTEGRATED_JOINT_RECALL_ROUTER_COST_AWARE`
- `integrated_router_invocation_count=3600`
- `selected_joint=1080`
- `selected_top1_top2=1800`
- `selected_external=360`
- `support=6.6480`
- `distance_to_concrete_oracle_support=0.3280`
- `gap_reduction_vs_D73_bound=0.1640`
- `exact=0.99917`
- `correlated=0.9965`
- `adversarial=0.9962`
- `external=0.9960`
- `joint_recall=0.9944`
- `external_recall=0.9960`
- `false_confidence=0.0042`
- `abstain=0.9950`
- `wrong_concrete_counter=0.0006`
- `weak_top1_top2_path_failure=0.0005`
- `top1_top2_sufficient_false_joint_rate=0.0010`
- `D68_loss_repair_preservation=1.0`
- `routing_failure_rows=0`
- `rust_path_invoked=true`
- `fallback_rows=0`
- `failed_jobs=[]`

D79 must keep the top1 sufficiency guard visible because the D78 top1 sufficiency ablation caused `routing_failure_rows=45` and `D68_loss_preservation=0.961538`.

## Tracks

1. `D78_REPLAY`
2. `LARGER_SEED_SCALE`
3. `OOD_INTEGRATED_ROUTING`
4. `HARD_CORRELATED_JOINT_RECALL`
5. `HARD_ADVERSARIAL_JOINT_RECALL`
6. `TOP1_TOP2_SUFFICIENT_ROWS`
7. `JOINT_REQUIRED_ROWS`
8. `EXTERNAL_TEST_REQUIRED`
9. `INDISTINGUISHABLE_ABSTAIN`
10. `D68_CHEAP_TOP1_REGRESSION_GUARD`
11. `TOP1_SUFFICIENCY_GUARD_ABLATION`
12. `SAFETY_MARGIN_WATCH`
13. `ORACLE_DISTANCE_FRONTIER`

## Arms

1. `D78_INTEGRATED_ROUTER_COST_AWARE_REPLAY`
2. `D78_INTEGRATED_ROUTER_HIGH_RECALL`
3. `D78_INTEGRATED_ROUTER_LOW_COST`
4. `D78_TOP1_SUFFICIENCY_ABLATION`
5. `D76_STANDALONE_COMPONENT_REPLAY`
6. `D71_D70_REPLAY`
7. `CONCRETE_ORACLE_REFERENCE_ONLY`
8. `RANDOM_ROUTER_CONTROL`
9. `NEVER_JOINT_CONTROL`
10. `ALWAYS_JOINT_CONTROL`
11. `TRUTH_LEAK_SENTINEL_REFERENCE_ONLY`

## Required reports

Artifacts are written under `target/pilot_wave/d79_joint_recall_integrated_controller_scale_confirm/`:

- `d78_upstream_manifest.json`
- `integrated_router_scale_report.json`
- `router_invocation_report.json`
- `top1_sufficiency_guard_report.json`
- `top1_sufficiency_ablation_report.json`
- `D68_cheap_top1_regression_guard_report.json`
- `D68_loss_repair_preservation_report.json`
- `joint_required_row_report.json`
- `external_required_report.json`
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

The scaled integrated router passes only if all of these hold:

- `integrated_router_invocation_count > 0`
- `exact_joint_accuracy >= 0.9990`
- `correlated_echo_accuracy >= 0.995`
- `adversarial_distractor_accuracy >= 0.995`
- `external_test_required_accuracy >= 0.995`
- `gap_reduction_vs_D73_bound >= 0.1500`
- `average_total_support_used <= 6.70`
- `distance_to_concrete_oracle_support <= 0.38`
- `joint_counter_recall_on_joint_required_rows >= 0.9940`
- `external_recall_on_external_required_rows >= 0.9957`
- `wrong_concrete_counter_rate <= 0.0007`
- `weak_top1_top2_path_failure_rate <= 0.0006`
- `top1_top2_sufficient_false_joint_rate <= 0.0015`
- `false_confidence_rate <= 0.0044`
- `indistinguishable_abstain_rate >= 0.9948`
- `D68_loss_repair_preservation_rate = 1.0`
- `routing_failure_rows = 0`
- top1 sufficiency guard ablation is worse
- `rust_path_invoked=true`
- `fallback_rows=0`
- `failed_jobs=[]`

## Decisions

- Passing scale confirmation: `decision=joint_recall_integrated_controller_scale_confirmed`, `next=D80_JOINT_RECALL_INTEGRATED_CONTROLLER_STRESS_MAP`.
- Passing with high support drift: `decision=joint_recall_integrated_controller_scale_high_cost`, `next=D79C_COST_REPAIR`.
- Safety/routing regression: `decision=joint_recall_integrated_controller_scale_safety_regression`, `next=D79S_ROUTING_SAFETY_REPAIR`.
- Top1 guard ablation not worse: `decision=top1_sufficiency_guard_not_validated`, `next=D79G_TOP1_GUARD_REPAIR`.
- Other failure: `decision=joint_recall_integrated_controller_scale_not_confirmed`, `next=D79_REPAIR`.

## Boundary

D79 only scale-confirms integrated joint-recall counter-action routing inside the controlled symbolic ECF/IPF joint formula discovery stack. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.

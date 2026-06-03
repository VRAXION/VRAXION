# D78 Joint-Recall Integrated Controller Prototype Contract

## Purpose

D78 implements and tests an integrated Rust sparse ECF/IPF controller path using the D77-selected `JointRecallCostAwareCounterRouter` as a real counter-action routing component in the controlled symbolic ECF/IPF joint formula discovery stack. The formula solver remains symbolic, external routing is not integrated at the same time, and no broad architecture claim is added.

## Phase 0 upstream audit

The runner must verify the current branch and HEAD, check whether D77 commit `39572fc5360964250fd2697528a538f7b74f8a04` is present and an ancestor of `HEAD`, and write `d77_upstream_manifest.json`. If the requested D77 commit or artifacts are missing, the runner must restore/rerun D77 before D78 and record that status explicitly. D78 must not silently assume D77 was pushed.

## D77 integration decision

D77 selected:

- `decision=joint_recall_integration_plan_selected`
- `next=D78_JOINT_RECALL_INTEGRATED_CONTROLLER_PROTOTYPE`
- `selected_target=COUNTER_ACTION_ROUTER_JOINT_RECALL_MODULE`
- `component_name=JointRecallCostAwareCounterRouter`

The component is integrated after top1/top2 sufficiency evaluation and before external-test escalation/postcheck abstain. It must not bypass the top1/top2 sufficiency flag solely from joint score, must preserve D68 loss rows, and must prevent cheap-top1 regression.

## Tracks

1. `D77_PLAN_REPLAY`
2. `INTEGRATED_CONTROLLER_MAIN`
3. `HARD_CORRELATED_JOINT_RECALL`
4. `HARD_ADVERSARIAL_JOINT_RECALL`
5. `TOP1_TOP2_SUFFICIENT_ROWS`
6. `JOINT_REQUIRED_ROWS`
7. `EXTERNAL_TEST_REQUIRED`
8. `INDISTINGUISHABLE_ABSTAIN`
9. `OOD_INTEGRATED_ROUTING`
10. `D68_CHEAP_TOP1_REGRESSION_GUARD`
11. `SAFETY_MARGIN_WATCH`
12. `ORACLE_DISTANCE_FRONTIER`

## Arms

1. `D76_JOINT_RECALL_COST_AWARE_REPLAY`
2. `INTEGRATED_JOINT_RECALL_ROUTER`
3. `INTEGRATED_JOINT_RECALL_ROUTER_COST_AWARE`
4. `INTEGRATED_JOINT_RECALL_ROUTER_HIGH_RECALL`
5. `INTEGRATED_JOINT_RECALL_ROUTER_LOW_COST`
6. `INTEGRATED_ROUTER_WITH_EXTERNAL_ESCALATION_DISABLED`
7. `INTEGRATED_ROUTER_WITH_TOP1_SUFFICIENCY_ABLATION`
8. `D68_CHEAP_TOP1_FAILURE_REPLAY`
9. `D71_D70_REPLAY`
10. `CONCRETE_ORACLE_REFERENCE_ONLY`
11. `RANDOM_ROUTER_CONTROL`
12. `NEVER_JOINT_CONTROL`
13. `ALWAYS_JOINT_CONTROL`
14. `TRUTH_LEAK_SENTINEL_REFERENCE_ONLY`

## Required reports

Artifacts are written under `target/pilot_wave/d78_joint_recall_integrated_controller_prototype/`:

- `d77_upstream_manifest.json`
- `integration_implementation_report.json`
- `component_interface_report.json`
- `rust_sparse_router_invocation_report.json`
- `integrated_controller_metrics_report.json`
- `joint_recall_integrated_scale_report.json`
- `top1_top2_sufficient_guard_report.json`
- `D68_cheap_top1_regression_guard_report.json`
- `D68_loss_repair_preservation_report.json`
- `external_required_report.json`
- `indistinguishable_abstain_report.json`
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

The best integrated fair arm passes only if the integrated router is invoked and all D76 support, oracle-gap, recall, external, D68, routing, safety, truth-leak, Rust, fallback, and failed-job gates are preserved:

- `integrated_router_invocation_count > 0`
- `rust_path_invoked=true`
- `fallback_rows=0`
- `failed_jobs=[]`
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
- truth-leak audit passes

## Decisions

- Passing integrated controller: `decision=joint_recall_integrated_controller_prototype_confirmed`, `next=D79_JOINT_RECALL_INTEGRATED_CONTROLLER_SCALE_CONFIRM`.
- Passing but high cost: `decision=joint_recall_integrated_controller_positive_high_cost`, `next=D78C_INTEGRATED_COST_REPAIR`.
- Routing/safety regression: `decision=joint_recall_integrated_controller_safety_regression`, `next=D78S_INTEGRATED_ROUTING_SAFETY_REPAIR`.
- Integrated router not actually invoked: `decision=joint_recall_integration_not_exercised`, `next=D78I_INTEGRATION_WIRING_REPAIR`.
- Other failure: `decision=joint_recall_integrated_controller_not_confirmed`, `next=D78_REPAIR`.

## Boundary

D78 only tests integrated joint-recall counter-action routing inside the controlled symbolic ECF/IPF joint formula discovery stack. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.

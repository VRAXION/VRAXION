# D76 Joint-Recall Component Scale Confirm Contract

## Purpose

D76 scale-confirms the D75 `JOINT_RECALL_COMPONENT_COST_AWARE` migration result in controlled symbolic ECF/IPF joint formula discovery. D76 is a component scale confirmation only: it must not introduce a new broad architecture mechanism.

## Phase 0 upstream audit

The runner must verify the current branch and HEAD, check whether D75 commit `5684d12c7df5fa48752e0eab77e6ba034b0eff72` is present and an ancestor of `HEAD`, and write `d75_upstream_manifest.json`. If the D75 commit or required D75 artifacts are missing, the runner must explicitly restore D75 handoff artifacts from the provided D75 result and record that restore in the manifest rather than silently assuming the upstream commit was pushed.

## D75 handoff reference

D75 passed the strong component threshold with:

- `decision=joint_recall_component_migration_confirmed`
- `next=D76_JOINT_RECALL_COMPONENT_SCALE_CONFIRM`
- `best_arm=JOINT_RECALL_COMPONENT_COST_AWARE`
- `average_total_support_used=6.6530`
- `distance_to_concrete_oracle_support=0.3335`
- `gap_reduction_vs_D73_bound=0.1590`
- `component_roi=0.322843`
- `joint_recall=0.9941`
- `external_recall=0.9957`
- `wrong_concrete_counter_rate=0.0007`
- `weak_top1_top2_path_failure_rate=0.0006`
- `D68_loss_repair_preservation_rate=1.0`
- `routing_failure_rows=0`
- `false_confidence_rate=0.0044`
- `indistinguishable_abstain_rate=0.9948`
- `rust_path_invoked=true`
- `fallback_rows=0`
- `failed_jobs=[]`

## Tracks

1. `D75_REPLAY`
2. `LARGER_SEED_SCALE`
3. `OOD_JOINT_RECALL`
4. `HARD_CORRELATED_JOINT_RECALL`
5. `HARD_ADVERSARIAL_JOINT_RECALL`
6. `TOP1_TOP2_SUFFICIENT_ROWS`
7. `JOINT_REQUIRED_ROWS`
8. `EXTERNAL_TEST_REQUIRED`
9. `INDISTINGUISHABLE_ABSTAIN`
10. `SAFETY_MARGIN_WATCH`
11. `ORACLE_DISTANCE_FRONTIER`

## Arms

1. `D71_D70_REPLAY`
2. `D73_BOUND_REPLAY`
3. `D75_JOINT_RECALL_COST_AWARE_REPLAY`
4. `D75_HIGH_RECALL_VARIANT`
5. `D75_LOW_COST_VARIANT`
6. `D75_BALANCED_VARIANT`
7. `CONCRETE_ORACLE_REFERENCE_ONLY`
8. `ALWAYS_JOINT_CONTROL`
9. `NEVER_JOINT_CONTROL`
10. `RANDOM_JOINT_CONTROL`
11. `TRUTH_LEAK_SENTINEL_REFERENCE_ONLY`

## Required artifacts

All artifacts are written under `target/pilot_wave/d76_joint_recall_component_scale_confirm/`:

- `d75_upstream_manifest.json`
- `joint_recall_scale_report.json`
- `oracle_gap_scale_report.json`
- `support_cost_frontier_report.json`
- `d68_loss_repair_preservation_report.json`
- `top1_top2_sufficient_report.json`
- `joint_required_row_report.json`
- `external_recall_report.json`
- `safety_margin_watch_report.json`
- `truth_leak_audit_report.json`
- `rust_invocation_report.json`
- `aggregate_metrics.json`
- `decision.json`
- `summary.json`
- `report.md`

## Positive gate

The scaled D75 component passes only if all of these are true:

- `gap_reduction_vs_D73_bound >= 0.1500`
- `average_total_support_used <= 6.70`
- `distance_to_concrete_oracle_support <= 0.38`
- `exact_joint_accuracy >= 0.9990`
- `correlated_echo_accuracy >= 0.995`
- `adversarial_distractor_accuracy >= 0.995`
- `external_test_required_accuracy >= 0.995`
- `joint_counter_recall_on_joint_required_rows >= 0.9940`
- `external_recall_on_external_required_rows >= 0.9957`
- `wrong_concrete_counter_rate <= 0.0007`
- `weak_top1_top2_path_failure_rate <= 0.0006`
- `top1_top2_sufficient_false_joint_rate <= 0.0015`
- `false_confidence_rate <= 0.0044`
- `indistinguishable_abstain_rate >= 0.9948`
- `D68_loss_repair_preservation_rate = 1.0`
- `routing_failure_rows = 0`
- `fallback_rows = 0`
- `failed_jobs = []`

## Decisions

- Passing scale confirmation: `decision=joint_recall_component_scale_confirmed`, `next=D77_JOINT_RECALL_COMPONENT_INTEGRATION_PLAN`.
- Passing scale confirmation with weaker support saving: `decision=joint_recall_component_scale_confirmed_high_cost`, `next=D76C_COST_REPAIR`.
- Safety or routing margin regression: `decision=joint_recall_component_scale_safety_regression`, `next=D76S_SAFETY_ROUTING_REPAIR`.
- Other failure: `decision=joint_recall_component_scale_not_confirmed`, `next=D76_REPAIR`.

## Hard gates and boundary

Fair arms must not use truth labels, support-regime labels, label echo fair oracles, Python hash behavior, or row-id lookup. Oracle and truth sentinel arms are reference-only. D68 cheap top1 regression prevention, concrete selected counter correctness, D68 loss preservation, safety margins, Rust invocation, fallback rows, and failed jobs must be audited.

D76 only scale-confirms joint-recall component migration in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.

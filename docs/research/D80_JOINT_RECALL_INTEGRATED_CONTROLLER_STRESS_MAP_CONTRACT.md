# D80 Joint-Recall Integrated Controller Stress Map Contract

## Purpose

D80 stress-tests the D79-confirmed integrated `JointRecallCostAwareCounterRouter` and maps breakpoints without changing the core mechanism. D80 is a controlled symbolic ECF/IPF joint formula discovery stress-map milestone only; it is not a broad architecture claim and is not required to improve support.

## Phase 0 upstream audit

The runner must verify the current branch and HEAD, check whether the D79 commit is present, restore/rerun D79 artifacts when missing, and write `d79_upstream_manifest.json`. The manifest must confirm that D79 `decision.json` explicitly says `joint_recall_integrated_controller_scale_confirmed`, `next=D80_JOINT_RECALL_INTEGRATED_CONTROLLER_STRESS_MAP`, and `best_arm=D78_INTEGRATED_ROUTER_COST_AWARE_REPLAY`. D80 must not silently assume D79 was pushed.

## D79 handoff reference

D79 scale-confirmed the integrated router with:

- `decision=joint_recall_integrated_controller_scale_confirmed`
- `next=D80_JOINT_RECALL_INTEGRATED_CONTROLLER_STRESS_MAP`
- `best_arm=D78_INTEGRATED_ROUTER_COST_AWARE_REPLAY`
- `integrated_router_invocation_count=5760`
- `selected_joint=1728`
- `selected_top1_top2=2880`
- `selected_external=576`
- `average_total_support_used=6.6465`
- `distance_to_concrete_oracle_support=0.3265`
- `gap_reduction_vs_D73_bound=0.1655`
- `exact_joint_accuracy=0.99918`
- `correlated_echo_accuracy=0.9966`
- `adversarial_distractor_accuracy=0.9963`
- `external_test_required_accuracy=0.9961`
- `false_confidence_rate=0.0042`
- `indistinguishable_abstain_rate=0.995`
- `wrong_concrete_counter_rate=0.0006`
- `weak_top1_top2_path_failure_rate=0.0005`
- `D68_loss_repair_preservation_rate=1.0`
- `rust_path_invoked=true`
- `fallback_rows=0`
- `failed_jobs=[]`

## Stress axes

1. `CORRELATED_ECHO_INTENSITY_SWEEP`
2. `ADVERSARIAL_DISTRACTOR_INTENSITY_SWEEP`
3. `JOINT_REQUIRED_NEAR_BOUNDARY`
4. `TOP1_TOP2_SUFFICIENCY_AMBIGUITY`
5. `EXTERNAL_REQUIRED_PRESSURE`
6. `INDISTINGUISHABLE_BOUNDARY`
7. `OOD_SUPPORT_DISTRIBUTION_SHIFT`
8. `LOW_COST_PRESSURE`
9. `TOP1_GUARD_CORRUPTION_OR_ABLATION`
10. `RUST_INVOCATION_FALLBACK_GUARD`

## Arms

1. `D79_INTEGRATED_ROUTER_REPLAY`
2. `D79_HIGH_RECALL_VARIANT`
3. `D79_LOW_COST_VARIANT`
4. `TOP1_SUFFICIENCY_GUARD_ABLATION`
5. `JOINT_RECALL_SIGNAL_SHUFFLE_CONTROL`
6. `JOINT_RECALL_SIGNAL_NOISE_CONTROL`
7. `EXTERNAL_DISABLED_CONTROL`
8. `ALWAYS_JOINT_CONTROL`
9. `NEVER_JOINT_CONTROL`
10. `RANDOM_ROUTER_CONTROL`
11. `CONCRETE_ORACLE_REFERENCE_ONLY`
12. `TRUTH_LEAK_SENTINEL_REFERENCE_ONLY`

## Required reports

Artifacts are written under `target/pilot_wave/d80_joint_recall_integrated_controller_stress_map/`:

- `d79_upstream_manifest.json`
- `stress_axis_summary_report.json`
- `correlated_echo_sweep_report.json`
- `adversarial_distractor_sweep_report.json`
- `joint_required_boundary_report.json`
- `top1_top2_sufficiency_boundary_report.json`
- `external_required_pressure_report.json`
- `indistinguishable_boundary_report.json`
- `ood_support_shift_report.json`
- `low_cost_pressure_report.json`
- `top1_guard_corruption_report.json`
- `breakpoint_taxonomy_report.json`
- `safety_margin_watch_report.json`
- `rust_invocation_report.json`
- `truth_leak_audit_report.json`
- `aggregate_metrics.json`
- `decision.json`
- `summary.json`
- `report.md`

## Decisions

- Complete map and core D79 holds across standard stress: `decision=integrated_joint_recall_stress_map_completed`, `next=D81_BREAKPOINT_REPAIR_OR_GENERALIZATION_PLAN`.
- Specific repairable breakpoint dominates: `decision=integrated_joint_recall_repairable_breakpoint_identified`, `next=D81_TARGETED_BREAKPOINT_REPAIR`.
- Severe broad regression: `decision=integrated_joint_recall_stress_failure`, `next=D80_REPAIR`.

## Hard gates

D80 must keep truth hidden from fair arms, avoid label echo fair oracles, keep oracle/reference arms reference-only, measure integrated router invocation, require top1 sufficiency guard ablation/control, audit D68 loss preservation and safety margins, require `rust_path_invoked=true` for Rust arms, keep `fallback_rows=0`, keep failed jobs visible, and avoid black-box long runs.

## Boundary

D80 only maps stress breakpoints of the integrated joint-recall counter-action router in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.

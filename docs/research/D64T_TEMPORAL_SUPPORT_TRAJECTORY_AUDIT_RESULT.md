# D64T Temporal Support Trajectory Audit Result

## Decision

```text
decision = support_order_not_required_set_aggregation_sufficient
verdict = D64T_SET_AGGREGATION_SUFFICIENT
next = D64S_REPAIR_OR_REDEFINE_DIAGNOSTIC_CLAIM
best_arm = POS_THEN_NEG_VS_NEG_THEN_POS
```

## Run

```text
scale_mode = scale-lite
seeds = 12201,12202,12203,12204,12205
train_rows_per_seed = 20
test_rows_per_seed = 80
ood_rows_per_seed = 80
rust_path_invoked = true
fallback_rows = 0
failed_jobs = []
```

## Core Readout

The D64S support-order result did not survive as a raw support-order dependency.
D64T changed the actual five support slots before feature generation. Random
order, set-invariant order, and stage-destroyed controls stayed near ceiling.
That means the earlier D64S `SUPPORT_ORDER_SHUFFLE` gap should be treated as a
diagnostic-routing artifact, not proof of temporal interference.

Key test-core metrics:

| arm | exact_joint | support | counter_support |
| --- | ---: | ---: | ---: |
| FULL_TRAJECTORY_READOUT | 1.000000 | 8.7200 | 3.7200 |
| FINAL_STATE_READOUT_ONLY | 1.000000 | 7.4570 | 2.4570 |
| SET_INVARIANT_AGGREGATION | 0.999500 | 9.3500 | 4.3500 |
| RANDOM_SUPPORT_ORDER_SHUFFLE | 0.999500 | 9.2915 | 4.2915 |
| STAGE_DESTROYING_SHUFFLE | 1.000000 | 5.2445 | 0.2445 |
| OPEN_LOOP_SUPPORT_SEQUENCE | 0.947500 | 9.1400 | 4.1400 |
| CLOSED_LOOP_FIELD_DEPENDENT_SUPPORT_SEQUENCE | 1.000000 | 8.7200 | 3.7200 |
| ARBITRARY_ORDER_ID_CONTROL | 0.999500 | 9.5585 | 4.5585 |

Useful positive subfinding:

```text
closed_loop_field_dependent_support_sequence > open_loop_support_sequence
```

So controlled support choice matters. But arbitrary raw list order or stage
order is not confirmed as a necessary signal in this stack.

## Artifacts

Authoritative generated files:

```text
target/pilot_wave/d64t_temporal_support_trajectory_audit/smoke/report.md
target/pilot_wave/d64t_temporal_support_trajectory_audit/smoke/decision.json
target/pilot_wave/d64t_temporal_support_trajectory_audit/smoke/aggregate_metrics.json
```

Boundary: D64T only audits support trajectory/order signal for a Rust sparse ECF
action controller in controlled symbolic joint formula discovery. It does not
prove full VRAXION brain, raw visual Raven reasoning, Raven solved, AGI,
consciousness, DNA/genome success, architecture superiority, or production
readiness.

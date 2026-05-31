# D64T Temporal Support Trajectory Audit Contract

## Question

D64S did not confirm a clean score-vector structure dependency, but its old
`SUPPORT_ORDER_SHUFFLE` control was not a raw temporal/support-order test. D64T
asks whether the real support sequence or support stage composition carries
useful signal for the Rust sparse ECF action-controller path.

## Method

D64T uses the same controlled symbolic joint formula discovery task as the
nearby D-runs. It changes the five support slots before feature generation:

```text
raw support sequence
-> changed support order/stage composition
-> feature generation
-> Rust sparse policy module via Network::propagate_sparse
-> ECF action
```

This is explicitly not a post-hoc shuffle of diagnostic bits.

## Arms

```text
ORIGINAL_TRAJECTORY_REFERENCE
SET_INVARIANT_AGGREGATION
RANDOM_SUPPORT_ORDER_SHUFFLE
STAGE_PRESERVING_SHUFFLE
STAGE_DESTROYING_SHUFFLE
A_THEN_B_VS_B_THEN_A
POS_THEN_NEG_VS_NEG_THEN_POS
EARLY_COUNTER_SUPPORT
LATE_COUNTER_SUPPORT
OPEN_LOOP_SUPPORT_SEQUENCE
CLOSED_LOOP_FIELD_DEPENDENT_SUPPORT_SEQUENCE
FINAL_STATE_READOUT_ONLY
FULL_TRAJECTORY_READOUT
ARBITRARY_ORDER_ID_CONTROL
TRUTH_LEAK_SENTINEL_REFERENCE_ONLY
```

## Required Artifacts

All generated artifacts live under:

```text
target/pilot_wave/d64t_temporal_support_trajectory_audit/
```

Required reports include trajectory-vs-set, stage-preserving shuffle,
stage-destroying shuffle, order artifact, early-vs-late counter, open-vs-closed
loop, final-state-vs-trajectory readout, truth leak audit, Rust invocation,
aggregate metrics, decision, summary, and row-level outputs.

## Decision Rules

`support_trajectory_signal_confirmed` requires fair full trajectory readout to
stay strong while random/stage-destroying controls drop.

`support_order_not_required_set_aggregation_sufficient` means set-like
aggregation performs about as well as trajectory variants.

`arbitrary_order_artifact_detected` means a fake order-id control explains the
effect better than support trajectory.

`temporal_interference_claim_not_confirmed` means the run did not isolate a
clean temporal/stage dependency.

## Boundary

D64T only audits support trajectory/order signal for a Rust sparse ECF action
controller in controlled symbolic joint formula discovery. It does not prove
full VRAXION brain, raw visual Raven reasoning, Raven solved, AGI,
consciousness, DNA/genome success, architecture superiority, or production
readiness.

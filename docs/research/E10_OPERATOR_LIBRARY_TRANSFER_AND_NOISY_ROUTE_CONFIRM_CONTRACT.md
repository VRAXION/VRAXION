# E10 Operator Library Transfer And Noisy Route Confirm Contract

## Purpose

`E10_OPERATOR_LIBRARY_TRANSFER_AND_NOISY_ROUTE_CONFIRM` follows the E09
integrated universal block result.

The question is:

```text
Can the fixed pruned region-operator pocket library transfer to new route
mixtures and noisy observed routes without re-running mutation discovery,
while preserving trace validity and writeback safety?
```

## Model

The runtime remains a deterministic binary Flow-grid proxy. The oracle/reference
trace is generated from the hidden true route and the known reference region
operators. The tested systems receive noisy observed-route evidence and must
write through the same block-shaped schema:

```text
detector_id
condition
read_region
transform_op
write_region
branch_id
trace_before
trace_after
confidence
cost
reason_code
```

The primary system uses the E09 scheduled/schema-gated universal block with the
E8H4-style pruned region-operator library. It is not allowed to re-run mutation
discovery during E10.

## Splits

```text
validation
heldout_transfer
noisy_route
partial_corruption
ood_mixture
adversarial_noise
```

The noisy splits corrupt the observed route through wrong tokens, inserted
tokens, dropped tokens, swapped tokens, degraded route confidence, and
adversarial high-confidence wrong route evidence. The evaluation oracle remains
the hidden true route.

## Systems

```text
DIRECT_OVERWRITE_NOISY_ROUTE
OBSERVED_ROUTE_SCHEMA_GATED_NO_REPAIR
REUSE_LIBRARY_NOISY_NO_GATE
TRANSFER_LIBRARY_SCHEDULED_SCHEMA_GATED
TRANSFER_LIBRARY_SCHEDULED_SCHEMA_GATED_PRUNED
HANDCODED_CLEAN_ROUTE_REFERENCE
```

The handcoded reference is a control, not a valid primary success.

## Required Metrics

```text
usefulness
answer_accuracy
final_state_accuracy
trace_validity
delta_validity
observed_route_error_rate
useful_writeback_recall
wrong_writeback_rate
destructive_overwrite_rate
branch_contamination_rate
stale_write_rejection_rate
gate_false_accept_rate
gate_false_reject_rate
route_repair_rate
noisy_route_false_accept_rate
transfer_coverage
clean_route_preservation_rate
operator_reuse_rate
temporal_drift_rate
oscillation_rate
attractor_collapse_rate
complex_calls_per_tick
cost_per_tick
deterministic_replay_passed
no_neural_dependency_detected
no_overclaim_boundary_preserved
```

## Positive Gate

`TRANSFER_LIBRARY_SCHEDULED_SCHEMA_GATED_PRUNED` must:

```text
beat DIRECT_OVERWRITE_NOISY_ROUTE on usefulness
beat DIRECT_OVERWRITE_NOISY_ROUTE on trace_validity
beat OBSERVED_ROUTE_SCHEMA_GATED_NO_REPAIR on usefulness
beat REUSE_LIBRARY_NOISY_NO_GATE on wrong_writeback_rate
keep trace_validity >= 0.90
keep usefulness >= 0.85
keep useful_writeback_recall >= 0.85
keep wrong_writeback_rate <= 0.05
keep destructive_overwrite_rate <= 0.05
keep branch_contamination_rate == 0
keep stale_write_rejection_rate >= 0.90
keep route_repair_rate >= 0.80
keep noisy_route_false_accept_rate <= 0.05
keep transfer_coverage >= 0.85
keep operator_reuse_rate >= 0.90
reduce cost_per_tick versus DIRECT_OVERWRITE_NOISY_ROUTE
avoid noisy/transfer split collapse
pass deterministic replay
avoid neural dependencies and broad claims
```

## Decisions

Allowed decisions:

```text
e10_operator_library_transfer_and_noisy_route_confirmed
e10_noisy_route_repair_insufficient
e10_transfer_trace_validity_failure
e10_writeback_safety_failure
e10_operator_reuse_or_coverage_failure
e10_usefulness_trace_tradeoff_unresolved
e10_invalid_or_incomplete_run
```

## Boundary

E10 is a controlled synthetic binary Flow-grid transfer and route-noise probe.
It does not claim raw language behavior, deployed-model behavior, broad
general capability, or model-scale behavior.

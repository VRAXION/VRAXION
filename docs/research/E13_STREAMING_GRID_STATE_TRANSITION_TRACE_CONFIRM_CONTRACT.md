# E13 Streaming Grid State-Transition Trace Confirm Contract

## Purpose

`E13_STREAMING_GRID_STATE_TRANSITION_TRACE_CONFIRM` tests whether the Binary
Flow / pocket runtime can maintain an internal grid state from streaming binary
grid frames:

```text
binary grid frame stream
-> Flow grid state
-> scheduled region pocket/operator blocks
-> trace-gated writeback
-> repaired final grid and operator trace
```

The probe is a deterministic synthetic grid-transition harness. It is not a
text parser, a deployed-system claim, or a neural network training run.

## Search-First Result

Before adding E13, the repo and fetched branches were searched for:

```text
E13
GRID_STATE_TRANSITION
grid state transition
streaming grid
transition trace
frame sequence
grid frame sequence
state transition trace
Flow grid transition
operator trace inference
shift expand contract split merge
streaming multi step flow composition
grid input output flow
temporal grid state
```

Only adjacent E12 next pointers and unrelated stable-loop numeric E13 strings
were found. No equivalent streaming grid state-transition milestone was present,
so this milestone is created under the requested E13 name.

## Input Model

Rows are deterministic binary grid-frame sequences. The runtime receives binary
grids only. Debug operator family names are confined to harness reports.

Grid sizes:

```text
8x8
12x12
16x16
24x24 OOD
```

Splits:

```text
train_like
validation
heldout_composition
noisy
adversarial_noise
missing_frame
ood_grid_size
long_horizon
branch_switch
```

Route lengths:

```text
1
3
6
12
24
```

The grid contains a binary trace-stamp lane and branch marker lane as part of
the synthetic Flow state. Payload operators act on the payload region; invalid
trace stamps are rejected by gated writeback.

## Debug Operator Families

```text
SHIFT
EXPAND
CONTRACT
SPLIT
MERGE
INVERT_REGION
CLEAR_NOISE
FILL_GAP
TRACE_ROUTE
HELDOUT_COMPOSITION
NOISY_TRACE
BRANCHING_TRACE
```

Runtime gates operate on binary grid/stamp state, not on external semantic
slots.

## Systems

```text
OBSERVED_FRAME_DIFF_BASELINE
DIRECT_OVERWRITE_GRID_BASELINE
NO_INTERNAL_STATE_BASELINE
ORACLE_TRACE_REFERENCE
FLOW_GRID_GATED_WRITEBACK
FLOW_GRID_TRACE_REPAIR
FLOW_GRID_SCHEDULED_POCKET_PRIMARY
FLOW_GRID_PRUNED_SCHEDULED_POCKET_PRIMARY
TINY_GRID_MLP_CONTROL
```

The primary candidate is:

```text
FLOW_GRID_PRUNED_SCHEDULED_POCKET_PRIMARY
```

The oracle is included only as a reference arm and is never the primary.

## Metrics

```text
final_grid_exact_accuracy
final_grid_similarity
operator_trace_exact_accuracy
per_step_operator_accuracy
frame_transition_accuracy
trace_validity
delta_validity
internal_state_consistency
internal_state_vs_observed_noisy_gap
noisy_repair_rate
missing_frame_repair_rate
decoy_rejection_rate
heldout_composition_accuracy
long_horizon_survival_rate
ood_grid_generalization
branch_switch_accuracy
licensed_destructive_accept_rate
unlicensed_destructive_reject_rate
wrong_writeback_rate
destructive_overwrite_rate
branch_contamination_rate
stale_write_rejection_rate
temporal_drift_rate
drift_slope_by_horizon
oscillation_rate
attractor_collapse_rate
cost_per_tick
deterministic_replay_passed
no_semantic_slot_leak_detected
no_neural_dependency_detected
no_overclaim_boundary_preserved
```

## Positive Gate

The primary passes only if:

```text
final_grid_exact_accuracy >= 0.95
final_grid_similarity >= 0.98
operator_trace_exact_accuracy >= 0.90
per_step_operator_accuracy >= 0.95
trace_validity >= 0.95
internal_state_consistency >= 0.95
noisy_repair_rate >= 0.90
missing_frame_repair_rate >= 0.85
decoy_rejection_rate >= 0.90
heldout_composition_accuracy >= 0.90
long_horizon_survival_rate >= 0.85
ood_grid_generalization >= 0.85
branch_switch_accuracy >= 0.95
licensed_destructive_accept_rate >= 0.90
unlicensed_destructive_reject_rate >= 0.95
wrong_writeback_rate <= 0.02
destructive_overwrite_rate <= 0.02
branch_contamination_rate == 0
temporal_drift_rate <= direct overwrite temporal drift
drift_slope_by_horizon is not explosive
deterministic_replay_passed == true
no_semantic_slot_leak_detected == true
no_neural_dependency_detected == true
```

The primary must also beat the observed/no-internal/direct baselines on the
dimensions where those baselines are meaningful: noisy/adversarial final repair,
missing-frame operator trace recovery, long/noisy operator trace recovery, and
writeback safety.

## Decisions

Allowed decisions:

```text
e13_streaming_grid_state_transition_trace_confirmed
e13_clean_trace_failure
e13_noisy_trace_repair_failure
e13_missing_frame_repair_failure
e13_heldout_composition_failure
e13_long_horizon_drift_failure
e13_ood_grid_generalization_failure
e13_destructive_license_failure
e13_writeback_safety_failure
e13_semantic_slot_leak_detected
e13_invalid_or_incomplete_run
```

Confirmed next:

```text
E14_REGION_AWARE_PARALLEL_POCKET_SCHEDULER_CONFIRM
```

## Required Artifacts

```text
decision.json
summary.json
aggregate_metrics.json
report.md
e13_search_report.json
e13_dataset_report.json
e13_system_comparison_report.json
e13_trace_accuracy_report.json
e13_noisy_repair_report.json
e13_missing_frame_report.json
e13_heldout_composition_report.json
e13_long_horizon_report.json
e13_ood_grid_report.json
e13_destructive_license_report.json
e13_writeback_safety_report.json
e13_semantic_leak_report.json
e13_deterministic_replay_report.json
```

## Boundary

E13 is a deterministic synthetic binary grid-frame transition probe. It does
not claim deployed behavior, broad model-scale behavior, or hardware speedup.

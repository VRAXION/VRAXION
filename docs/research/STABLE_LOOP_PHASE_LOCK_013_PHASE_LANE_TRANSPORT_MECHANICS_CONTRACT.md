# STABLE_LOOP_PHASE_LOCK_013_PHASE_LANE_TRANSPORT_MECHANICS Contract

## Summary

013 diagnoses the blocker left by 012:

```text
the completed local phase rule is not the blocker
the recurrent phase-lane substrate still fails long-path transport
```

This probe is diagnostic only. It does not search, mutate, prune, package a
production API, or claim full VRAXION.

## Files

```text
docs/research/STABLE_LOOP_PHASE_LOCK_013_PHASE_LANE_TRANSPORT_MECHANICS_CONTRACT.md
instnct-core/examples/phase_lane_transport_mechanics.rs
docs/research/STABLE_LOOP_PHASE_LOCK_013_PHASE_LANE_TRANSPORT_MECHANICS_RESULT.md
```

No public `instnct-core` APIs are changed.

## Diagnostic Arms

```text
FULL_16_RULE_TEMPLATE_BASELINE
COMPLETED_SPARSE_TEMPLATE_BASELINE
PER_STEP_ORACLE_INJECTION
STEPWISE_ORACLE_CLOCK
PATH_ONLY_FORWARD_CLOCK
FINAL_TICK_READOUT
BEST_TICK_READOUT
FIRST_ARRIVAL_READOUT
PERSISTENT_TARGET_READOUT
ARRIVE_LATCH_1TICK
ARRIVE_LATCH_PERSISTENT
EMIT_LATCH_PERSISTENT
CONSUME_ON_FORWARD_LATCH
BIDIRECTIONAL_GRID_BASELINE
ORACLE_DIRECTION_NO_BACKFLOW
PUBLIC_GRADIENT_NO_BACKFLOW
CELL_LOCAL_NORMALIZATION
TARGET_ONLY_NORMALIZATION
RANDOM_CONTROL
```

Private-path diagnostic-only arms:

```text
PER_STEP_ORACLE_INJECTION
STEPWISE_ORACLE_CLOCK
PATH_ONLY_FORWARD_CLOCK
ORACLE_DIRECTION_NO_BACKFLOW
```

These may use `true_path`, but their results cannot support a public/deployable
mechanism claim.

## Hard Order

`PER_STEP_ORACLE_INJECTION` runs first. It tests every local edge with all 16
input phase / gate pairs.

Per-step interpretation:

```text
ticks < one-edge settling window:
  timing diagnostic only

ticks >= one-edge settling window:
  all 16 pairs must pass
```

If per-step never reaches the pairwise gate:

```text
PER_STEP_TRANSPORT_FAILS_PAIRWISE
```

If per-step passes but the chain fails:

```text
PER_STEP_TRANSPORT_WORKS_BUT_CHAIN_FAILS
```

## Readout And Latch Semantics

Readout-only arms must use identical propagation snapshots. Only scoring policy
may change:

```text
FINAL_TICK_READOUT
BEST_TICK_READOUT
FIRST_ARRIVAL_READOUT
PERSISTENT_TARGET_READOUT
```

Latch variants:

```text
ARRIVE_LATCH_1TICK:
  incoming phase remains one extra tick

ARRIVE_LATCH_PERSISTENT:
  once a cell receives phase, it keeps it

EMIT_LATCH_PERSISTENT:
  emitted phase persists for neighbor propagation

CONSUME_ON_FORWARD_LATCH:
  phase persists until successful downstream receive, then clears

PERSISTENT_TARGET_READOUT:
  target remembers best seen phase distribution
```

No-backflow split:

```text
ORACLE_DIRECTION_NO_BACKFLOW:
  uses true_path direction, diagnostic only

PUBLIC_GRADIENT_NO_BACKFLOW:
  uses source/target geometry only
```

Normalization:

```text
CELL_LOCAL_NORMALIZATION:
  can support a transport diagnosis

TARGET_ONLY_NORMALIZATION:
  readout calibration diagnostic only

global normalization:
  forbidden for main claim
```

## Metrics And Outputs

Required metrics:

```text
phase_final_accuracy
best_tick_accuracy
first_arrival_accuracy
persistent_target_accuracy
correct_target_lane_probability_mean
per_step_transfer_accuracy
per_pair_step_accuracy[input_phase, gate]
per_pair_step_probability[input_phase, gate]
target_arrival_rate
correct_if_arrived_accuracy
wrong_if_arrived_rate
first_tick_correct
last_tick_correct
correct_then_lost_rate
target_power_total_by_tick
backflow_power
echo_power
phase_decay_per_step
wrong_phase_growth_rate
latch_retention_rate
readout_timing_gap
wall_leak_rate
gate_shuffle_collapse
forbidden_private_field_leak
nonlocal_edge_count
direct_output_leak_rate
```

Required outputs:

```text
queue.json
progress.jsonl
metrics.jsonl
transport_curve.jsonl
per_step_metrics.jsonl
per_pair_step_metrics.jsonl
readout_timing_metrics.jsonl
clock_metrics.jsonl
latch_metrics.jsonl
backflow_metrics.jsonl
normalization_metrics.jsonl
arrival_metrics.jsonl
family_metrics.jsonl
counterfactual_metrics.jsonl
locality_audit.jsonl
summary.json
report.md
contract_snapshot.md
examples_sample.jsonl
job_progress/*.jsonl
```

The runner must append progress and metrics continuously. There is no black-box
run.

## Verdicts

```text
PER_STEP_TRANSPORT_OK
PER_STEP_TRANSPORT_FAILS
PER_STEP_TRANSPORT_FAILS_PAIRWISE
PER_STEP_TRANSPORT_WORKS_BUT_CHAIN_FAILS
READOUT_TIMING_IS_BLOCKER
READOUT_TEST_CONTAMINATED
TARGET_READOUT_PERSISTENCE_REQUIRED
TARGET_READOUT_CALIBRATION_LIMIT
EARLY_CORRECT_LATE_OVERWRITE
TARGET_MEMORY_RESCUES_READOUT
CELL_LATCH_RESCUES_HORIZON
STEPWISE_CLOCK_RESCUES_HORIZON
BACKFLOW_INTERFERENCE_IS_BLOCKER
ORACLE_NO_BACKFLOW_RESCUES
PUBLIC_NO_BACKFLOW_RESCUES
ONLY_ORACLE_NO_BACKFLOW_RESCUES
PHASE_DECAY_LIMIT
WRONG_PHASE_INTERFERENCE_LIMIT
DECAY_PLUS_INTERFERENCE_LIMIT
PHASE_NORMALIZATION_RESCUES_HORIZON
GLOBAL_NORMALIZATION_CONTAMINATION
GATE_PATTERN_SPECIFIC_FAILURE
SIGNAL_ARRIVAL_FAILURE
SIGNAL_ARRIVES_WRONG_PHASE
RECURRENT_TRANSPORT_MECHANICS_BLOCKER
DIRECT_SHORTCUT_CONTAMINATION
PRODUCTION_API_NOT_READY
```

## Claim Boundary

013 can support a classified transport blocker. It cannot support production
architecture, full VRAXION, consciousness, language grounding, Prismion
uniqueness, or physical quantum behavior.

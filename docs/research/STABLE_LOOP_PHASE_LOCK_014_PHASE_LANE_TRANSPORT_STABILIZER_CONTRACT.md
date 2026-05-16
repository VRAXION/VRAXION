# STABLE_LOOP_PHASE_LOCK_014_PHASE_LANE_TRANSPORT_STABILIZER Contract

## Summary

014 follows the 013 transport-mechanics diagnosis:

```text
PER_STEP_TRANSPORT_OK
PER_STEP_TRANSPORT_WORKS_BUT_CHAIN_FAILS
SIGNAL_ARRIVES_WRONG_PHASE
WRONG_PHASE_INTERFERENCE_LIMIT
EARLY_CORRECT_LATE_OVERWRITE
RECURRENT_TRANSPORT_MECHANICS_BLOCKER
PRODUCTION_API_NOT_READY
```

The local phase rule is complete:

```text
phase_i + gate_g -> phase_(i+g)
```

014 does not search, mutate, prune, or change public `instnct-core` APIs. It
tests whether public stabilizer combinations can reduce wrong-phase
interference and restore stable long-chain final readout.

## Files

```text
docs/research/STABLE_LOOP_PHASE_LOCK_014_PHASE_LANE_TRANSPORT_STABILIZER_CONTRACT.md
instnct-core/examples/phase_lane_transport_stabilizer.rs
docs/research/STABLE_LOOP_PHASE_LOCK_014_PHASE_LANE_TRANSPORT_STABILIZER_RESULT.md
```

Raw `target/` outputs are not committed.

## Stabilizer Lattice

The runner evaluates all 16 public subsets of:

```text
arrive_latch_1tick
cell_local_normalization
public_no_backflow
target_memory_readout
```

Required controls:

```text
BASELINE_FULL16
RANDOM_CONTROL_BASE
RANDOM_CONTROL_WITH_FULL_STABILIZER
ORACLE_NO_BACKFLOW_PLUS_LATCH_NORMALIZATION  # diagnostic only
```

Target memory cannot by itself count as transport solved. The runner reports:

```text
transport_accuracy_without_target_memory
final_accuracy_with_target_memory
```

## Positive Gate

`TRANSPORT_STABILIZER_FOUND` requires a public combo to satisfy:

```text
phase_final_accuracy >= 0.95
long_path_accuracy >= 0.95
family_min_accuracy >= 0.85
same_target_counterfactual_accuracy >= 0.85
gate_shuffle_collapse >= 0.50
wrong_if_arrived_rate_delta <= -0.25
final_minus_best_gap <= 0.05
wall_leak_rate <= 0.02
forbidden/private/nonlocal/direct leaks = 0
random_control_with_same_stabilizer remains weak
```

Minimal stabilizer is the smallest passing public subset, tie-broken by higher
accuracy and then lower wrong-if-arrived rate.

## Metrics And Outputs

Core metrics:

```text
phase_final_accuracy
best_tick_accuracy
persistent_target_accuracy
target_arrival_rate
wrong_if_arrived_rate
correct_then_lost_rate
correct_phase_margin
wrong_phase_growth_rate
backflow_power
echo_power
phase_decay_per_step
gate_shuffle_collapse
same_target_counterfactual_accuracy
wall_leak_rate
stale_phase_rate
repeated_phase_lock_rate
phase_update_success_rate
forbidden_private_field_leak
nonlocal_edge_count
direct_output_leak_rate
```

Required outputs:

```text
queue.json
progress.jsonl
metrics.jsonl
stabilizer_lattice.jsonl
combo_metrics.jsonl
minimality_metrics.jsonl
family_metrics.jsonl
counterfactual_metrics.jsonl
locality_audit.jsonl
summary.json
report.md
contract_snapshot.md
examples_sample.jsonl
job_progress/*.jsonl
```

The runner appends progress and metrics continuously. There is no black-box run.

## Verdicts

```text
TRANSPORT_STABILIZER_FOUND
MINIMAL_STABILIZER_IDENTIFIED
LATCH_REQUIRED
NORMALIZATION_REQUIRED
NO_BACKFLOW_REQUIRED
TARGET_MEMORY_REQUIRED
ONLY_ORACLE_STABILIZES
PUBLIC_STABILIZER_FAILS
PUBLIC_COMBO_REDUCES_WRONG_PHASE_BUT_NOT_ENOUGH
TRANSPORT_SOLVED_WITHOUT_TARGET_MEMORY
TRANSPORT_MASKED_BY_TARGET_MEMORY
TARGET_MEMORY_ONLY_NOT_TRANSPORT
BEST_TICK_ONLY_NOT_STABLE
FAMILY_MIN_GATE_FAILS
GATE_PATTERN_SPECIFIC_FAILURE
LATCH_STALE_STATE_FAILURE
STABILIZER_OVERPOWERS_RULE_CONTROL
WRONG_PHASE_INTERFERENCE_REDUCED
DIRECT_SHORTCUT_CONTAMINATION
PRODUCTION_API_NOT_READY
```

## Claim Boundary

014 can support a public stabilizer envelope for this toy phase-lane substrate.
It cannot support production architecture, full VRAXION, consciousness,
language grounding, Prismion uniqueness, or physical quantum behavior.

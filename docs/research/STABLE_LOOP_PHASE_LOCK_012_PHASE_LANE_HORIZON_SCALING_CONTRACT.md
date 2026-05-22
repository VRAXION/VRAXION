# STABLE_LOOP_PHASE_LOCK_012_PHASE_LANE_HORIZON_SCALING Contract

## Summary

012 measures the recurrent horizon left after 011:

```text
the completed local rule exists:
  phase_i + gate_g -> phase_(i+g)

the remaining question is how far the recurrent phase-lane substrate can carry
that signal before timing, decay, readout timing, or wrong-phase interference
breaks it
```

This is not a motif discovery, growth, or pruning probe.

## Files

```text
docs/research/STABLE_LOOP_PHASE_LOCK_012_PHASE_LANE_HORIZON_SCALING_CONTRACT.md
instnct-core/examples/phase_lane_horizon_scaling.rs
docs/research/STABLE_LOOP_PHASE_LOCK_012_PHASE_LANE_HORIZON_SCALING_RESULT.md
```

No public `instnct-core` APIs are changed.

## Arms

```text
FIXED_PHASE_LANE_REFERENCE
FULL_16_RULE_TEMPLATE
COMMON_CORE_15_PLUS_MISSING_1_2_3
DENSE_009_REFERENCE
RANDOM_MATCHED_16_MOTIF_CONTROL
CANONICAL_JACKPOT_007_BASELINE
SOURCE_PERSIST_1_TICK
SOURCE_PERSIST_2_TICKS
SOURCE_PERSIST_ALL_TICKS
```

The source-persistence arms are diagnostic only.

## Sweep

```text
path_length = 2,4,8,12,16,24,32,48
ticks       = 4,8,12,16,24,32,48,64
width       = 8,12,16
```

The runner uses deterministic local corridors, with serpentine paths when a
straight path does not fit. Model input remains public/local:

```text
wall/free mask
source location and phase lane
target marker
per-cell local gate bucket
```

Forbidden as model input:

```text
gate_sum
label
true_path
path_phase_total
oracle route
oracle answer
global pooling
nonlocal target readout
```

## Gate Stress Families

```text
all_zero_gates
repeated_plus_one
repeated_plus_two
alternating_plus_minus
random_balanced
high_cancellation_sequence
adversarial_wrong_phase_sequence
same_target_counterfactual
gate_shuffle_control
```

## Metrics

Each arm/width/path_length/ticks/family bucket reports:

```text
phase_final_accuracy
correct_target_lane_probability_mean
target_power_total
correct_phase_margin
wrong_phase_growth_rate
wall_leak_rate
same_target_counterfactual_accuracy
gate_shuffle_collapse
best_tick_accuracy
best_tick_correct_probability
final_tick_minus_best_tick_delta
phase_decay_per_step
```

Diagnostic path/readout metrics:

```text
correct_phase_power_by_step
wrong_phase_power_by_step
correct_phase_margin_by_step
target_power_total_by_step
target_readout_by_tick
minimum_ticks_for_95_accuracy
minimum_ticks_for_90_probability
```

## Required Outputs

```text
queue.json
progress.jsonl
metrics.jsonl
horizon_curve.jsonl
path_length_metrics.jsonl
tick_metrics.jsonl
phase_decay_metrics.jsonl
wrong_phase_metrics.jsonl
readout_over_time.jsonl
counterfactual_metrics.jsonl
locality_audit.jsonl
summary.json
report.md
contract_snapshot.md
examples_sample.jsonl
job_progress/*.jsonl
```

The runner refreshes `summary.json` and `report.md` on heartbeat and appends
metrics after every completed bucket. There is no black-box run.

## Verdicts

```text
HORIZON_SCALING_PASSES
HORIZON_LIMIT_IDENTIFIED
TICKS_ONLY_LIMIT
PHASE_DECAY_LIMIT
WRONG_PHASE_INTERFERENCE_LIMIT
DECAY_PLUS_INTERFERENCE_LIMIT
EARLY_ARRIVAL_LATE_DECAY
SOURCE_DECAY_LIMIT
RULE_TEMPLATE_STABLE_BUT_SETTLING_LIMITED
SPARSE_EQUALS_FULL16_HORIZON
SPARSE_UNDERPERFORMS_FULL16_HORIZON
DENSE_REFERENCE_HAS_HORIZON_ADVANTAGE
FULL16_REFERENCE_BREAKS_ON_LONG_PATHS
RANDOM_CONTROL_FAILS
DIRECT_SHORTCUT_CONTAMINATION
PRODUCTION_API_NOT_READY
```

The horizon pass gate is aggregated by path_length and ticks. A single lucky
family bucket is not enough to claim horizon scaling.

## Claim Boundary

012 can support:

```text
measured recurrent horizon
tick budget and readout timing diagnostics
decay/interference diagnostics
valid path-length scaling envelope for the toy phase-lane substrate
```

012 cannot support:

```text
production architecture
full VRAXION
consciousness
language grounding
Prismion uniqueness
physical quantum behavior
```

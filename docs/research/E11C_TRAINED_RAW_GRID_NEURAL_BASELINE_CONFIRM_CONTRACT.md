# E11C Trained Raw-Grid Neural Baseline Confirm Contract

## Purpose

`E11C_TRAINED_RAW_GRID_NEURAL_BASELINE_CONFIRM` follows E11B.

E11B compared Flow against neural-controller proxies that received E10 detector
evidence. E11C removes that advantage and asks:

```text
Can a trained raw-grid neural baseline, given raw grid state plus corrupted
route input, reach the same trace/writeback safety region as the E10 Flow
runtime?
```

## Scope

The probe reuses the E10 deterministic binary Flow-grid task family. The neural
baselines are trained with stdlib-only backprop from teacher-forced examples:

```text
raw current grid
+ observed corrupted route tokens
+ route position features
-> true next operator skill
```

The neural baselines do not receive E10 per-skill detector confidence or trace
confidence evidence.

## Systems

```text
FLOW_E10_SCHEDULED_SCHEMA_GATED_PRUNED
OBSERVED_ROUTE_NO_TRAIN_BASELINE
TRAINED_RAW_GRID_ROUTE_SOFTMAX
TRAINED_RAW_GRID_ROUTE_MLP
```

The Flow system is the E10 primary. The trained neural systems are raw-grid /
route-input classifiers with fixed region-operator decoders.

## Required Metrics

```text
usefulness
trace_validity
answer_accuracy
useful_writeback_recall
wrong_writeback_rate
destructive_overwrite_rate
route_repair_rate
transfer_coverage
quality_matched
proxy_ops_per_tick
cost_per_correct_trace
cost_per_valid_writeback
training_accuracy
training_loss
deterministic_replay_passed
```

Quality matched means:

```text
trace_validity >= 0.90
usefulness >= 0.85
useful_writeback_recall >= 0.85
wrong_writeback_rate <= 0.05
```

## Positive Gate

To confirm Flow advantage against trained raw-grid neural baselines:

```text
FLOW_E10_SCHEDULED_SCHEMA_GATED_PRUNED must be quality matched
FLOW wrong_writeback_rate must be 0
FLOW destructive_overwrite_rate must be 0
at least one trained raw-grid neural baseline must be quality matched
FLOW proxy_ops_per_tick must be lower than the cheapest quality-matched trained neural baseline
FLOW cost_per_valid_writeback must be lower than the cheapest quality-matched trained neural baseline
neural baselines must not receive detector evidence
deterministic replay must pass
```

If no trained raw-grid neural baseline is quality matched, the correct decision
is not a Flow cost win. It is a neural comparator failure / baseline redesign
result.

## Decisions

Allowed decisions:

```text
e11c_flow_advantage_vs_trained_raw_grid_neural_confirmed
e11c_trained_raw_grid_neural_baseline_not_quality_matched
e11c_trained_raw_grid_neural_beats_or_matches_flow
e11c_flow_quality_failure
e11c_invalid_or_incomplete_run
```

## Boundary

E11C is a controlled synthetic binary Flow-grid trained-baseline probe. It does
not prove hardware speedups, larger neural baselines, raw-language behavior,
deployed-model behavior, or broad model-scale behavior.

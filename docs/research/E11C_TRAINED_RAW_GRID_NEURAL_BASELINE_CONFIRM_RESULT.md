# E11C Trained Raw-Grid Neural Baseline Confirm Result

Status: completed.

## Decision

```text
decision = e11c_trained_raw_grid_neural_baseline_not_quality_matched
next = E11C_STRONGER_RAW_GRID_NEURAL_BASELINE_OR_TASK_IDENTIFIABILITY_AUDIT
primary_system = FLOW_E10_SCHEDULED_SCHEMA_GATED_PRUNED
positive_gate_passed = false
deterministic_replay_passed = true
checker_failure_count = 0
```

Run root:

```text
target/pilot_wave/e11c_trained_raw_grid_neural_baseline_confirm/
```

## What Was Tested

E11C removed the E11B controller shortcut. The neural baselines were trained
from:

```text
raw current binary grid
+ observed corrupted route tokens
+ route-position features
-> true next operator skill
```

They did not receive E10 detector confidence or trace confidence evidence.

## Quality And Cost

| system | usefulness | trace | recall | wrong | repair | quality matched | ops/tick |
|---|---:|---:|---:|---:|---:|---:|---:|
| FLOW_E10_SCHEDULED_SCHEMA_GATED_PRUNED | 0.987 | 0.990 | 0.985 | 0.000 | 0.976 | true | 200.0 |
| OBSERVED_ROUTE_NO_TRAIN_BASELINE | 0.838 | 0.879 | 0.672 | 0.328 | 0.000 | false | 32.0 |
| TRAINED_RAW_GRID_ROUTE_SOFTMAX | 0.755 | 0.803 | 0.399 | 0.601 | 0.155 | false | 2680.0 |
| TRAINED_RAW_GRID_ROUTE_MLP | 0.834 | 0.867 | 0.628 | 0.372 | 0.050 | false | 3904.0 |

Quality-matched definition:

```text
trace_validity >= 0.90
usefulness >= 0.85
useful_writeback_recall >= 0.85
wrong_writeback_rate <= 0.05
```

No trained raw-grid neural baseline met that bar.

## Training

```text
dependency_mode = stdlib-only backprop
train_rows = 336
eval_rows = 432
softmax_examples = 3456
mlp_examples = 3456
softmax_epochs = 8
mlp_epochs = 6
```

Final training metrics:

```text
TRAINED_RAW_GRID_ROUTE_SOFTMAX train_accuracy = 0.513, train_loss = 1.467
TRAINED_RAW_GRID_ROUTE_MLP     train_accuracy = 0.700, train_loss = 0.930
```

The MLP learned more than the softmax model but still failed transfer/safety:

```text
TRAINED_RAW_GRID_ROUTE_MLP
usefulness = 0.834
trace_validity = 0.867
useful_writeback_recall = 0.628
wrong_writeback_rate = 0.372
```

## Interpretation

E11C does not confirm a Flow cost advantage against a quality-matched trained
raw-grid neural baseline, because no trained raw-grid neural comparator reached
the quality/safety bar.

The result does show a useful gap:

```text
E11B neural-controller baselines worked when given detector/trace evidence.
E11C raw-grid trained baselines did not recover the same safe route-repair
behavior from raw grid + corrupted route input.
```

So the current state is:

```text
E11B confirmed: Flow is cheaper than quality-matched neural-controller proxies.
E11C not confirmed: no quality-matched trained raw-grid neural comparator yet.
```

The next step should either strengthen the raw-grid neural baseline or audit
whether the E10 task is identifiable from raw grid plus corrupted route alone.

## Verification

```text
python3 scripts/probes/run_e11c_trained_raw_grid_neural_baseline_confirm.py
python3 scripts/probes/run_e11c_trained_raw_grid_neural_baseline_confirm_check.py --out target/pilot_wave/e11c_trained_raw_grid_neural_baseline_confirm --write-summary
```

The checker passed with `failure_count = 0`.

Boundary: E11C is a deterministic synthetic binary Flow-grid trained-baseline
probe only. It does not make hardware speedup, larger neural baseline,
deployment, model-scale, or broad capability claims.

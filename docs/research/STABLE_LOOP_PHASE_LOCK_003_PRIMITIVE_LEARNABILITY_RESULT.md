# STABLE_LOOP_PHASE_LOCK_003_PRIMITIVE_LEARNABILITY Result

## Status

Implemented and ran the primitive learnability probe after `STABLE_LOOP_PHASE_LOCK_002_TRANSFER`.

The final 002 claim remains:

```text
Stable-loop wavefront: YES
Fixed phase-lock primitive: YES
Complex multiply sufficient: YES
Prismion-unique: NOT PROVEN
Learned Prismion: NOT LEARNED IN CURRENT FORM
```

This 003 run asks whether the learned cell can acquire the local complex phase-transport primitive outside the sparse full grid transfer setting.

## Run

```text
out=target/pilot_wave/stable_loop_phase_lock_003_primitive_learnability/confirm_5seed
seeds=2026-2030
train_examples=8192
eval_examples=8192
epochs=20
jobs=6
device=cpu
completed_jobs=190
```

CPU was capped at `jobs=6`. No GPU path was used.

## Verdict

```text
TASK_VALID
PRIMITIVE_LEARNABLE
```

Not supported:

```text
PRIMITIVE_NOT_LEARNABLE_AS_IMPLEMENTED
COMPOSITION_STABILITY_FAILURE
PRETRAINING_RESCUES_TRANSFER_STRONGLY
PRETRAINING_RESCUES_INITIALIZATION
FIXED_COMPLEX_OPERATOR_STILL_REQUIRED
TASK_OR_CONTROL_INVALID
```

## Primary Results

Primary setting:

```text
gate_repr=gate_as_complex_pair
gate_norm=normalized_gate
norm_policy=no_renorm_between_steps
```

| Arm | Single-Step | Teacher-Forced Composition | Free-Run Composition | Angle MAE | Complex MSE | Params |
|---|---:|---:|---:|---:|---:|---:|
| `FIXED_COMPLEX_MULTIPLY_TEACHER` | `100.0%` | `100.0%` | `100.0%` | `0.0000` | `0.000000` | `0` |
| `COMPLEX_MULTIPLY_GNN` | `100.0%` | `100.0%` | `100.0%` | `0.0000` | `0.000000` | `0` |
| `CURRENT_FACTOR_CELL_SINGLE_STEP` | `100.0%` | `100.0%` | `100.0%` | `0.0000` | `0.000000` | `6` |
| `CURRENT_FACTOR_CELL_MULTI_STEP` | `100.0%` | `99.9%` | `99.9%` | `0.0009` | `0.000006` | `6` |
| `LOCAL_BILINEAR_SINGLE_STEP` | `99.6%` | `99.6%` | `99.3%` | `0.0056` | `0.000429` | `18` |
| `LOCAL_BILINEAR_MULTI_STEP` | `99.5%` | `99.4%` | `99.1%` | `0.0073` | `0.000227` | `18` |
| `RICH_PHASE_CELL_SINGLE_STEP` | `99.1%` | `99.1%` | `98.0%` | `0.0137` | `0.002018` | `1282` |
| `RICH_PHASE_CELL_MULTI_STEP` | `98.1%` | `98.1%` | `92.2%` | `0.0300` | `0.012630` | `1282` |
| `TINY_MLP_BASELINE` | `97.3%` | `97.2%` | `93.6%` | `0.0444` | `0.016003` | `170` |

## Transfer Results

| Arm | Transfer Accuracy |
|---|---:|
| `RANDOM_INIT_PHASE_CELL_TRANSFER` | `100.0%` |
| `PRETRAINED_FROZEN_PHASE_CELL_TRANSFER` | `100.0%` |
| `PRETRAINED_FINETUNED_PHASE_CELL_TRANSFER` | `100.0%` |

Paired deltas:

```text
Pretrained frozen - random transfer:
  mean_delta = 0.0000
  lower95    = 0.0000
  positive   = 0/5

Pretrained finetuned - random transfer:
  mean_delta = 0.0000
  lower95    = 0.0000
  positive   = 0/5
```

Interpretation:

```text
Pretraining was not needed in this simplified primitive/path transfer setting.
Random-init transfer also reached ceiling.
```

This does not contradict 002. It says the 002 failure was not primitive unlearnability; it was more likely the full spatial phase-lock transfer integration / credit-assignment setup.

## Gate Encoding And Normalization

Key observations:

```text
current_factor_cell:
  angle theta, cos/sin, and normalized complex pair all reached ~100%.

rich_phase_cell:
  complex pair and cos/sin were strongest.
  angle theta learned but was weaker.

unnormalized complex pair:
  can cause free-run drift unless the cell learns to ignore gate magnitude.
```

This validates the required gate-encoding ablation: a bad representation could have produced a false negative, but the main positive survives the strong encodings.

## Teacher-Forced Vs Free-Run

The key distinction:

```text
current_factor_cell:
  teacher-forced ~= 99.9-100.0%
  free-run       ~= 99.9-100.0%

rich_phase_cell:
  teacher-forced ~= 98.1-99.1%
  free-run       ~= 92.2-98.0%

tiny_mlp:
  teacher-forced ~= 97.2%
  free-run       ~= 93.6%
```

So there is some free-run drift in generic learned cells, but not enough to trigger `COMPOSITION_STABILITY_FAILURE` for the current factor cell.

## What This Proves

The local complex phase-transport primitive is learnable in direct supervision:

```text
CURRENT_FACTOR_CELL learns it.
RICH_PHASE_CELL learns it.
LOCAL_BILINEAR learns it.
TINY_MLP mostly learns it.
```

Therefore, the earlier learned-Prismion failure in 002 should not be read as:

```text
the primitive is not learnable
```

The better interpretation is:

```text
the primitive is learnable,
but the full phase-lock transfer setup made credit assignment/integration fail for the learned arm.
```

## What This Does Not Prove

```text
does not prove consciousness
does not prove full VRAXION
does not prove language grounding
does not prove general reasoning
does not prove Prismion is uniquely required
does not prove learned spatial transfer is solved in the full 002 runner
```

## Updated Research State

```text
002 transfer:
  complex multiply sufficient
  fixed primitive only in that runner

003 primitive learnability:
  primitive learnable
  composition mostly stable for current factor cell
  simplified transfer reaches ceiling

Current blocker:
  full spatial phase-lock transfer / credit assignment,
  not primitive acquisition itself
```

## Recommended Next Step

Do not repeat primitive imitation.

Next useful probe:

```text
STABLE_LOOP_PHASE_LOCK_004_SPATIAL_CREDIT_ASSIGNMENT
```

Goal:

```text
Keep the learned primitive cell capable,
then reintroduce the full spatial phase-lock setting gradually:
  short path
  longer path
  sparse target-only final labels
  distractor paths
  same-target-neighborhood counterfactuals
```

This should determine exactly where the learned cell loses the primitive when embedded in the full grid loop.

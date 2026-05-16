# STABLE_LOOP_PHASE_LOCK_003_PRIMITIVE_LEARNABILITY Contract

## Goal

`STABLE_LOOP_PHASE_LOCK_002_TRANSFER` showed:

```text
fixed phase-lock primitive works
generic complex multiplication is sufficient
learned Prismion did not learn the primitive in the transfer setup
```

This probe asks a narrower question:

```text
Can a learned cell acquire the local complex phase-transport primitive?
```

This is not another phase-lock transfer sweep and not a full VRAXION claim.

## Tasks

Required tasks:

```text
single_step_complex_multiply_imitation
multi_step_phase_composition
pretrained_primitive_transfer_back_into_phase_lock_002
```

Single step:

```text
input:  z = real + i imag, gate = exp(i theta)
target: z' = z * gate
```

Multi-step:

```text
input:  z0, gate1...gateN
target: zN = z0 * gate1 * gate2 * ... * gateN
N:      1, 2, 4, 8, 16, 32
```

Transfer:

```text
pretrain learned cell on primitive imitation
insert it into the phase-lock transfer loop
compare random-init, pretrained-frozen, pretrained-finetuned, fixed teacher, and complex multiply reference
```

## Required Arms

```text
FIXED_COMPLEX_MULTIPLY_TEACHER
CURRENT_FACTOR_CELL_SINGLE_STEP
CURRENT_FACTOR_CELL_MULTI_STEP
RICH_PHASE_CELL_SINGLE_STEP
RICH_PHASE_CELL_MULTI_STEP
LOCAL_BILINEAR_SINGLE_STEP
LOCAL_BILINEAR_MULTI_STEP
TINY_MLP_BASELINE
COMPLEX_MULTIPLY_GNN
RANDOM_INIT_PHASE_CELL_TRANSFER
PRETRAINED_FROZEN_PHASE_CELL_TRANSFER
PRETRAINED_FINETUNED_PHASE_CELL_TRANSFER
```

Report separately:

```text
current_factor_cell_result
rich_phase_cell_result
pretrained_phase_cell_result
```

## Mandatory Ablations

Gate representation:

```text
gate_as_angle_theta
gate_as_cos_sin
gate_as_complex_pair
normalized_gate
unnormalized_gate
```

Composition modes:

```text
teacher_forced_composition
free_run_composition
```

Normalization policies:

```text
no_renorm_between_steps
renorm_to_unit_magnitude_between_steps
learned_or_raw_magnitude_preserved
```

Transfer variants:

```text
random_init_phase_cell_transfer
pretrained_frozen_phase_cell_transfer
pretrained_finetuned_phase_cell_transfer
fixed_teacher_transfer_reference
complex_multiply_transfer_reference
```

## Metrics

Use wrapped circular angle error:

```text
angle_error = atan2(sin(pred_theta - true_theta), cos(pred_theta - true_theta))
```

Required metrics:

```text
complex_mse
phase_angle_mae
phase_class_accuracy
magnitude_error
magnitude_drift
phase_drift_per_step
composition_accuracy_by_N
teacher_forced_composition_accuracy
free_run_composition_accuracy
transfer_accuracy_after_pretrain
random_init_transfer_accuracy
pretrained_frozen_transfer_accuracy
pretrained_finetuned_transfer_accuracy
fixed_teacher_gap
pretrained_minus_random_transfer_delta
learned_minus_bilinear_delta
learned_minus_mlp_delta
```

Controls:

```text
label_shuffle_control
gate_shuffle_control
theta_shuffle_control
target_shuffle_control
magnitude_only_control
angle_only_control
```

Optimization audit:

```text
parameter_count
trainable_parameter_count
optimizer
lr
train_steps
final_train_loss
grad_norm_mean
grad_norm_max
activation_norm_mean
activation_norm_max
```

## Verdicts

```text
PRIMITIVE_LEARNABLE
PRIMITIVE_NOT_LEARNABLE_AS_IMPLEMENTED
COMPOSITION_STABILITY_FAILURE
PRIMITIVE_LEARNABLE_BUT_TRANSFER_FAILS
PRETRAINING_RESCUES_TRANSFER_STRONGLY
PRETRAINING_RESCUES_INITIALIZATION
FIXED_COMPLEX_OPERATOR_STILL_REQUIRED
TASK_OR_CONTROL_INVALID
```

## Claim Boundary

This probe only tests primitive learnability and transfer into a toy phase-lock loop. It does not prove consciousness, full VRAXION, language grounding, or general reasoning.

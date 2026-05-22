# STABLE_LOOP_PHASE_LOCK_002_TRANSFER Contract

## Goal

`STABLE_LOOP_PHASE_INTERFERENCE_003_PHASE_LOCK` produced the first clean Prismion-positive result, but the winning arm had zero trainable parameters. This probe tests what that win means.

The target distinction is:

```text
fixed hard-coded phase primitive
learned Prismion-style phase operator
generic bilinear / complex multiplication
canonical message passing
shortcut-contaminated success
```

This is not a broad sweep and not a full VRAXION claim.

## Task

Use the phase-lock grid task:

```text
source phase travels through local gate phases
incoming at destination = neighbor phase * local gate phase
target answer = target phase bucket
```

Training remains final target phase only. Intermediate phase maps are oracle/probe artifacts only.

Transfer variants:

```text
longer_paths
noisy_gate_field
heldout_gate_angles
mixed_cancel_plus_phase_lock
gate_dropout
same_local_target_contrast
reverse_path_consistency
variable_S_same_weights
```

## Arms

```text
ORACLE_PHASE_LOCK_TRANSFER
SUMMARY_DIRECT_HEAD
TARGET_MARKER_ONLY
HARD_WALL_ABC_PHASE_LOCK_LOOP
LOCAL_MESSAGE_PASSING_GNN_PHASE_LOCK
UNTIED_LOCAL_CNN_TARGET_READOUT_PHASE_LOCK
FIXED_PRISMION_PHASE_LOCK_LOOP
LEARNED_PRISMION_PHASE_LOCK_LOOP
LOCAL_BILINEAR_PHASE_LOOP
COMPLEX_MULTIPLY_GNN
```

## Adversarial Controls

Required controls:

```text
label_shuffle_control
gate_angle_shuffle_control
target_cell_shuffle_control
wall_mask_shuffle_control
path_direction_shuffle_control
phase_target_mismatch_control
paired_counterfactual_eval
off_manifold_phase_eval
```

Controls are interpreted against:

```text
majority_baseline
random_baseline
chance_threshold = majority_baseline + 0.05
```

## Metrics

Core:

```text
phase_lock_accuracy
transfer_accuracy
heldout_gate_accuracy
noisy_gate_accuracy
mixed_cancel_phase_accuracy
reverse_path_consistency_accuracy
same_local_target_pair_accuracy
paired_counterfactual_accuracy
same_patch_different_answer_accuracy
off_manifold_angle_error
off_manifold_bucket_accuracy
same_weights_s_curve_accuracy
long_path_accuracy
wrong_phase_rate
false_none_rate
phase_drift_error
wall_leak
pre_wall_pressure
```

Audit:

```text
parameter_count
trainable_parameter_count
operator_type
phase_multiply
optimizer
train_steps
gradient_norm_mean
activation_norm_mean
```

Comparisons:

```text
prismion_minus_abc_by_seed
prismion_minus_gnn_by_seed
prismion_minus_bilinear_by_seed
prismion_minus_complex_multiply_gnn_by_seed
fixed_minus_learned_prismion_by_seed
mean_delta
lower95_delta
positive_seed_count
```

## Verdicts

```text
PHASE_LOCK_TRANSFER_TASK_VALID
FIXED_PRISMION_TRANSFER_POSITIVE
LEARNED_PRISMION_TRANSFER_POSITIVE
PRISMION_UNIQUELY_USEFUL
COMPLEX_MULTIPLY_SUFFICIENT
FIXED_PRIMITIVE_ONLY
CANONICAL_MESSAGE_PASSING_RECOVERS
TASK_TOO_EASY_FOR_PRISMION_DISCRIMINATION
SUMMARY_OR_TARGET_SHORTCUT_RETURNS
TRANSFER_FAILS
PHASE_LOOP_UNSTABLE
```

## Claim Boundary

This does not prove consciousness, full VRAXION, language grounding, or general reasoning. It only tests whether the phase-lock result transfers and whether the necessary primitive is Prismion-specific or generic local complex multiplication.

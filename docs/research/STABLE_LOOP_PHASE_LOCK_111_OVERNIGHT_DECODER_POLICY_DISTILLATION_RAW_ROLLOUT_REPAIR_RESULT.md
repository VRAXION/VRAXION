# STABLE_LOOP_PHASE_LOCK_111_OVERNIGHT_DECODER_POLICY_DISTILLATION_RAW_ROLLOUT_REPAIR_RESULT

## Status

Pending full overnight execution.

The expected smoke output root is:

```text
target/pilot_wave/stable_loop_phase_lock_111_overnight_decoder_policy_distillation_raw_rollout_repair/smoke
```

## Positive Result Contract

The 111 result is valid only when `summary.json` emits:

```text
OVERNIGHT_DECODER_POLICY_DISTILLATION_RAW_ROLLOUT_REPAIR_POSITIVE
```

The positive result must show real overnight target-only training:

```text
wall_clock_minutes >= min_runtime_minutes
min_runtime_minutes >= 360
target_111_checkpoint_changed = true
train_step_count > 0
optimizer_step_count > 0
train_loss_final < train_loss_initial
```

Short runs, eval-only runs, and sleep-padded runs are invalid:

```text
OVERNIGHT_RUNTIME_UNDERUSED
NO_ACTUAL_TRAINING_UPDATE_DETECTED
```

## Required Evidence

Raw improvement:

```text
post_111_raw_ood_accuracy >= 0.80
post_111_raw_ood_accuracy >= pre_111_raw_ood_accuracy + 0.20
post_111_raw_accuracy_gap_to_integrated_teacher <= 0.15
post_111_raw_per_family_min_accuracy >= 0.65 excluding Hungarian diagnostic
```

Runtime and resources:

```text
cuda_available
selected_device
gpu_name
gpu_utilization_samples
gpu_memory_samples
median_gpu_utilization
p75_gpu_utilization
p95_gpu_utilization
gpu_idle_fraction
```

If CUDA is available but not used:

```text
CUDA_AVAILABLE_BUT_NOT_USED
```

If GPU use is too low:

```text
RESOURCE_UNDERUTILIZATION_DETECTED
```

Retention and LM:

```text
bounded_chat_slot_binding_accuracy >= 0.90
finite_label_anchorroute_retention_accuracy >= 0.90
unsupported_refusal_retention_accuracy >= 0.80
fineweb_eval_loss_regression <= 0.50
fineweb_next_byte_accuracy_drop <= 0.10
```

Boundary:

```text
artifact_exfiltration_count = 0
gpt_like_claim_count = 0
production_chat_claim_count = 0
public_api_claim_count = 0
safety_alignment_claim_count = 0
```

Immutability:

```text
source_102_checkpoint_unchanged = true
source_100_checkpoint_unchanged = true
packaged_winner_hash_unchanged = true
bounded_release_artifact_unchanged = true
```

Final raw eval purity:

```text
integrated_policy_used_during_final_raw_eval = false
decoder_reference_used_during_final_raw_eval = false
expected_answer_used_during_eval = false
```

Failures:

```text
INTEGRATED_POLICY_USED_DURING_RAW_EVAL
ORACLE_SHORTCUT_DETECTED
TEACHER_DATASET_LEAKAGE_DETECTED
TRAIN_EVAL_LEAKAGE_DETECTED
BASELINE_EVAL_MISMATCH
RAW_OOD_ACCURACY_NOT_IMPROVED
RAW_TO_INTEGRATED_GAP_REMAINS_HIGH
FINEWEB_RETENTION_REGRESSION_DETECTED
BOUNDED_RETENTION_REGRESSION_DETECTED
OVERCLAIM_DETECTED
ARTIFACT_EXFILTRATION_DETECTED
SOURCE_CHECKPOINT_MUTATION_DETECTED
PACKAGED_CHECKPOINT_MUTATION_DETECTED
BOUNDED_RELEASE_MUTATION_DETECTED
RUNTIME_SURFACE_MUTATION_DETECTED
```

## Decision

`decision.json` must choose one:

```text
112_RAW_ASSISTANT_MULTI_SEED_OOD_CONFIRM
111B_DISTILLATION_PARTIAL_FAILURE_ANALYSIS
111R_RETENTION_OR_LM_REGRESSION_ANALYSIS
111H_OVERNIGHT_HARNESS_UTILIZATION_FIX
111C_BOUNDARY_FAILURE_ANALYSIS
```

## Boundary

111 is target-only overnight research training.

111 does not prove:

```text
GPT-like assistant readiness
open-domain assistant readiness
production chat
public API
deployment readiness
safety alignment
```

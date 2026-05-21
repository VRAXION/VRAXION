# STABLE_LOOP_PHASE_LOCK_102_DECODER_POLICY_AND_ROLLOUT_REPAIR Contract

## Summary

102 is a target-only research repair for the raw rollout drift diagnosed by 101.

Primary upstream failure:

```text
RAW_ROLLOUT_DRIFT
raw_failure_count = 96
case_id_drift = 85
slot_drift = 2
wrong_language = 8
```

102 trains only a new target-local 102 checkpoint copied from the 100 checkpoint. It must not mutate 100, 101, 099, packaged winner artifacts, runtime/service/deploy code, SDK/public exports, product/release docs, or root `LICENSE`.

This is raw generation repair only. It is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, not public release, and not safety alignment.

## Required Repair Mechanisms

```text
CASE_ID_ANCHOR_COPY_LOSS
ACTIVE_SLOT_PINNING_LOSS
DISTRACTOR_SUPPRESSION_DURING_ROLLOUT
ROLLOUT_CONSISTENCY_LOSS
PREFIX_STABILITY_TRAINING
SCHEDULED_SAMPLING_OR_FREE_ROLLOUT_AUGMENTATION
STOP_CONDITION_STABILIZATION
WRONG_LANGUAGE_SUPPRESSION
```

Raw generation for the main arm must remain genuinely autoregressive:

```text
raw_generation_path = autoregressive
decoder_assisted_used_for_raw = false
ranked_scoring_used_for_raw = false
response_table_used_for_main_prediction = false
prediction_oracle_used = false
```

Failure:

```text
RAW_GENERATION_PATH_CONTAMINATED
ORACLE_SHORTCUT_DETECTED
```

## Frozen Sources

Record before and after:

```text
source_100_checkpoint_hash_before
source_100_checkpoint_hash_after
source_100_checkpoint_unchanged = true
bounded_release_artifact_hash_before
bounded_release_artifact_hash_after
bounded_release_artifact_unchanged = true
packaged_winner_hash_before
packaged_winner_hash_after
packaged_winner_hash_unchanged = true
```

Failure:

```text
SOURCE_100_CHECKPOINT_MUTATION_DETECTED
BOUNDED_RELEASE_MUTATION_DETECTED
PACKAGED_CHECKPOINT_MUTATION_DETECTED
```

## Arms

All arms must use identical eval rows:

```text
PRE_REPAIR_100_RAW_BASELINE
DECODER_POLICY_REPAIR_MAIN
NO_CASE_ID_ANCHOR_LOSS_CONTROL
NO_SLOT_PINNING_CONTROL
NO_ROLLOUT_CONSISTENCY_CONTROL
NO_LANGUAGE_GUARD_CONTROL
DECODER_ASSISTED_REFERENCE
COPY_PROMPT_CONTROL
STATIC_OUTPUT_CONTROL
```

Failure:

```text
EVAL_ROW_MISMATCH
CONTROL_DELTA_INSUFFICIENT
COPY_OR_STATIC_CONTROL_UNEXPECTED_PASS
```

## Gates

Positive requires:

```text
train_step_count > 0
target_102_checkpoint_changed = true
train_loss_final < train_loss_initial
raw_free_generation_accuracy >= upstream_101_raw_free_generation_accuracy + 0.25
raw_free_generation_accuracy >= 0.50
case_id_drift_rate <= upstream_101_case_id_drift_rate * 0.50
distractor_number_copy_rate <= 0.10
slot_drift_rate <= upstream_101_slot_drift_rate + 0.02
decoder_assisted_accuracy >= 0.90
decoder_assisted_accuracy_delta >= -0.03
bounded_chat_slot_binding_accuracy >= 0.90
finite_label_anchorroute_retention_accuracy >= 0.90
unsupported_refusal_accuracy >= 0.80
empty_output_rate <= 0.02
static_output_rate <= 0.15
repetition_rate <= 0.25
copy_prompt_rate <= 0.20
utf8_valid_generation_rate >= 0.80
nonempty_generation_rate >= 0.98
fineweb_eval_loss_regression <= 0.50
next_byte_accuracy_drop <= 0.12
```

Required control deltas:

```text
main case_id_drift_rate at least 25 percentage points lower than NO_CASE_ID_ANCHOR_LOSS_CONTROL
main raw_free_generation_accuracy at least 15 percentage points higher than NO_ROLLOUT_CONSISTENCY_CONTROL
main wrong_language_rate lower than NO_LANGUAGE_GUARD_CONTROL
main beats COPY_PROMPT_CONTROL and STATIC_OUTPUT_CONTROL
```

## Positive Verdicts

```text
DECODER_POLICY_AND_ROLLOUT_REPAIR_POSITIVE
UPSTREAM_101_FRONTIER_MAP_VERIFIED
SOURCE_100_CHECKPOINT_LOADED_READ_ONLY
TARGET_102_REPAIR_TRAINING_COMPLETED
RAW_GENERATION_IMPROVES
CASE_ID_DRIFT_REDUCED
SLOT_PINNING_RETAINED
DECODER_ASSISTED_REFERENCE_RETAINED
BOUNDED_CHAT_RETENTION_PASSES
FINITE_LABEL_ANCHORROUTE_RETENTION_PASSES
COLLAPSE_REJECTED
NO_RUNTIME_SURFACE_MUTATION
GPT_LIKE_READINESS_NOT_CLAIMED
PRODUCTION_CHAT_NOT_CLAIMED
```

## Failure Verdicts

```text
RAW_GENERATION_PATH_CONTAMINATED
ORACLE_SHORTCUT_DETECTED
SOURCE_100_CHECKPOINT_MUTATION_DETECTED
BOUNDED_RELEASE_MUTATION_DETECTED
PACKAGED_CHECKPOINT_MUTATION_DETECTED
NO_ACTUAL_TRAINING_UPDATE_DETECTED
TOKEN_OBJECTIVE_NOT_LEARNED
TRAIN_EVAL_LEAKAGE_DETECTED
EVAL_ROW_MISMATCH
CASE_ID_ANCHOR_REPAIR_INSUFFICIENT
DISTRACTOR_NUMBER_COPY_DETECTED
SLOT_DRIFT_REGRESSION_DETECTED
DISTRACTOR_SUPPRESSION_REGRESSION_DETECTED
CONTROL_DELTA_INSUFFICIENT
COPY_OR_STATIC_CONTROL_UNEXPECTED_PASS
DECODER_ASSISTED_REGRESSION_DETECTED
LM_RETENTION_REGRESSION_DETECTED
EMPTY_OUTPUT_COLLAPSE_DETECTED
STATIC_RESPONSE_COLLAPSE_DETECTED
REPETITION_COLLAPSE_DETECTED
WRONG_LANGUAGE_REGRESSION_DETECTED
RETENTION_REGRESSION_DETECTED
HUMAN_SAMPLE_REPORT_MISSING
GPT_LIKE_READINESS_FALSE_CLAIM
PRODUCTION_CHAT_CLAIM_DETECTED
PUBLIC_RELEASE_CLAIM_DETECTED
```

## Next Rule

If positive:

```text
103_FRESH_RAW_GENERATION_CONFIRM
```

If case ID still fails:

```text
102B_CASE_ID_ANCHOR_FAILURE_ANALYSIS
```

If retention regresses:

```text
RETENTION_FAILURE_ANALYSIS
```

If decoder-assisted regresses:

```text
DECODER_ASSISTED_REGRESSION_ANALYSIS
```

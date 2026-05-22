# STABLE_LOOP_PHASE_LOCK_103_FRESH_RAW_GENERATION_CONFIRM Contract

## Summary

103 is an eval-only fresh raw-generation confirmation gate for the 102 decoder policy and rollout repair. It asks whether the 102 smoke improvement generalizes to fresh, non-102-shaped prompts.

This milestone performs no training, no repair, no optimizer steps, no checkpoint mutation, and no runtime/service/deploy changes. It is a fresh raw-generation confirmation only, with no model capability improved by 103.

## Allowed Changes

103 may add only:

- `scripts/probes/run_stable_loop_phase_lock_103_fresh_raw_generation_confirm.py`
- `scripts/probes/run_stable_loop_phase_lock_103_fresh_raw_generation_confirm_check.py`
- `docs/research/STABLE_LOOP_PHASE_LOCK_103_FRESH_RAW_GENERATION_CONFIRM_CONTRACT.md`
- `docs/research/STABLE_LOOP_PHASE_LOCK_103_FRESH_RAW_GENERATION_CONFIRM_RESULT.md`

Generated artifacts must live only under:

```text
target/pilot_wave/stable_loop_phase_lock_103_fresh_raw_generation_confirm/
```

103 must not modify runtime/service/deploy code, SDK/public exports, product/release docs, root `LICENSE`, existing checkpoints, 099 bounded release artifacts, or 100/101/102 artifacts.

## Required Upstreams

103 requires:

```text
DECODER_POLICY_AND_ROLLOUT_REPAIR_POSITIVE
FRESH_ASSISTANT_FRONTIER_MAP_POSITIVE
OPEN_VOCAB_ASSISTANT_CAPABILITY_SCALE_POSITIVE
BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_POSITIVE
```

It must verify the 102 evidence:

```text
raw_free_generation_accuracy >= 0.90
case_id_drift_rate = 0.0
decoder_assisted_accuracy >= 0.90
bounded_chat_slot_binding_accuracy = 1.0
finite_label_anchorroute_retention_accuracy = 1.0
unsupported_refusal_accuracy = 1.0
source_100_checkpoint_unchanged = true
bounded_release_artifact_unchanged = true
packaged_winner_hash_unchanged = true
```

## Eval-Only Hard Wall

103 must record and require:

```text
train_step_count = 0
optimizer_step_count = 0
checkpoint_hash_unchanged = true
bounded_release_artifact_unchanged = true
source_100_checkpoint_unchanged = true
```

Failure verdicts:

```text
TRAINING_SIDE_EFFECT_DETECTED
CHECKPOINT_MUTATION_DETECTED
BOUNDED_RELEASE_MUTATION_DETECTED
```

## Raw Path Integrity

`RAW_GREEDY_GENERATION` and `RAW_SAMPLED_LOW_TEMP` must remain raw autoregressive confirmation paths and must not use ranked scoring, prefix forcing, decoder-assisted correction, response table lookup, expected-answer metadata, or oracle parsing.

Required records:

```text
raw_generation_path = autoregressive
ranked_scoring_used_for_raw = false
prefix_forcing_used_for_raw = false
decoder_assisted_used_for_raw = false
response_table_used_for_raw = false
prediction_oracle_used = false
```

Failure verdicts:

```text
RAW_GENERATION_PATH_CONTAMINATED
ORACLE_SHORTCUT_DETECTED
```

## Freshness And Same-Row Gates

Fresh rows must not overlap with 101 eval, 102 train/eval, or 100 train/eval samples. All modes must share identical rows.

Required records:

```text
overlap_with_101_eval_count
overlap_with_102_train_count
overlap_with_102_eval_count
overlap_with_100_train_eval_count
max_prompt_jaccard_vs_102_train
max_prompt_jaccard_vs_102_eval
eval_row_hash
eval_prompt_hash
eval_count
eval_dataset_sha256
```

Failure verdicts:

```text
TRAIN_EVAL_LEAKAGE_DETECTED
EVAL_ROW_MISMATCH
```

## Positive Gate

Positive requires:

```text
raw_free_generation_accuracy >= 0.85
raw_free_generation_accuracy >= upstream_101_raw + 0.50
case_id_drift_rate <= 0.10
distractor_number_copy_rate <= 0.10
slot_drift_rate <= 0.05
distractor_leak_rate <= 0.10
decoder_assisted_accuracy >= 0.90
decoder_assisted_accuracy_delta_vs_102 >= -0.05
bounded_chat_slot_binding_accuracy >= 0.90
finite_label_anchorroute_retention_accuracy >= 0.90
unsupported_refusal_accuracy >= 0.80
nonempty_generation_rate >= 0.98
utf8_valid_generation_rate >= 0.80
empty_output_rate <= 0.02
static_output_rate <= 0.15
repetition_rate <= 0.25
copy_prompt_rate <= 0.20
checkpoint_hash_unchanged = true
bounded_release_artifact_unchanged = true
train_step_count = 0
optimizer_step_count = 0
```

Positive verdicts include:

```text
FRESH_RAW_GENERATION_CONFIRM_POSITIVE
UPSTREAM_102_REPAIR_VERIFIED
RAW_GENERATION_GENERALIZES
CASE_ID_ANCHOR_GENERALIZES
SLOT_PINNING_GENERALIZES
DECODER_ASSISTED_REFERENCE_RETAINED
UNSUPPORTED_REFUSAL_RETAINED
BOUNDED_CHAT_RETENTION_PASSES
FINITE_LABEL_ANCHORROUTE_RETENTION_PASSES
COLLAPSE_REJECTED
CHECKPOINT_UNCHANGED
NO_TRAINING_PERFORMED
GPT_LIKE_READINESS_NOT_CLAIMED
PRODUCTION_CHAT_NOT_CLAIMED
```

Failure verdicts include:

```text
FRESH_RAW_GENERATION_CONFIRM_FAILS
UPSTREAM_102_ARTIFACT_MISSING
UPSTREAM_102_NOT_POSITIVE
TRAINING_SIDE_EFFECT_DETECTED
CHECKPOINT_MUTATION_DETECTED
BOUNDED_RELEASE_MUTATION_DETECTED
TRAIN_EVAL_LEAKAGE_DETECTED
EVAL_ROW_MISMATCH
RAW_GENERATION_GENERALIZATION_FAILS
CASE_ID_ANCHOR_GENERALIZATION_FAILS
DISTRACTOR_NUMBER_COPY_DETECTED
SLOT_PINNING_GENERALIZATION_FAILS
DISTRACTOR_SUPPRESSION_REGRESSION_DETECTED
DECODER_ASSISTED_REGRESSION_DETECTED
UNSUPPORTED_REFUSAL_REGRESSION_DETECTED
RETENTION_REGRESSION_DETECTED
STATIC_RESPONSE_COLLAPSE_DETECTED
REPETITION_COLLAPSE_DETECTED
EMPTY_OUTPUT_COLLAPSE_DETECTED
RAW_GENERATION_PATH_CONTAMINATED
ORACLE_SHORTCUT_DETECTED
HUMAN_SAMPLE_REPORT_MISSING
GPT_LIKE_READINESS_FALSE_CLAIM
PRODUCTION_CHAT_CLAIM_DETECTED
ROOT_LICENSE_CHANGED
```

## Decision Rule

If 103 passes:

```text
104_MULTI_SEED_RAW_GENERATION_CONFIRM
```

If case ID fails:

```text
103B_CASE_ID_ANCHOR_GENERALIZATION_FAILURE_ANALYSIS
```

If retention fails:

```text
RETENTION_FAILURE_ANALYSIS
```

If collapse fails:

```text
RAW_GENERATION_COLLAPSE_FAILURE_ANALYSIS
```

## No Overclaim

103 must explicitly state:

- fresh raw-generation confirmation only
- not GPT-like assistant readiness
- not open-domain assistant readiness
- not production chat
- not public API
- not deployment readiness
- not public release
- not safety alignment
- Hungarian capability is diagnostic only and not proven

# STABLE_LOOP_PHASE_LOCK_104_MULTI_SEED_RAW_GENERATION_CONFIRM Contract

## Summary

104 is an eval-only multi-seed confirmation gate after the positive 103 fresh raw-generation confirm. It reruns fresh raw-generation confirmation for seeds `2027,2028,2029` and requires every seed to pass independently.

104 performs no training, no repair, no optimizer steps, no checkpoint mutation, and no runtime/service/deploy changes. It is multi-seed raw-generation confirmation only and no model capability is improved by 104.

## Allowed Changes

104 may add only:

- `scripts/probes/run_stable_loop_phase_lock_104_multi_seed_raw_generation_confirm.py`
- `scripts/probes/run_stable_loop_phase_lock_104_multi_seed_raw_generation_confirm_check.py`
- `docs/research/STABLE_LOOP_PHASE_LOCK_104_MULTI_SEED_RAW_GENERATION_CONFIRM_CONTRACT.md`
- `docs/research/STABLE_LOOP_PHASE_LOCK_104_MULTI_SEED_RAW_GENERATION_CONFIRM_RESULT.md`

Generated artifacts must live only under:

```text
target/pilot_wave/stable_loop_phase_lock_104_multi_seed_raw_generation_confirm/
```

104 must not modify runtime/service/deploy code, SDK/public exports, product/release docs, root `LICENSE`, existing checkpoints, 099 release artifacts, or 100/101/102/103 artifacts.

## Required Upstreams

104 requires:

```text
FRESH_RAW_GENERATION_CONFIRM_POSITIVE
DECODER_POLICY_AND_ROLLOUT_REPAIR_POSITIVE
FRESH_ASSISTANT_FRONTIER_MAP_POSITIVE
OPEN_VOCAB_ASSISTANT_CAPABILITY_SCALE_POSITIVE
BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_POSITIVE
```

## Seed Rules

Default seeds:

```text
2027,2028,2029
```

Every seed must record:

```text
seed_run_started = true
seed_run_completed = true
seed_summary_newer_than_104_start = true
seed_report_newer_than_104_start = true
seed_command
```

Failure:

```text
STALE_SEED_ARTIFACT_USED
```

No mean-only pass, best-seed pass, or 2/3 pass is allowed.

Failure:

```text
MULTI_SEED_RAW_GENERATION_INSTABILITY_DETECTED
```

## Raw Path Integrity

For `RAW_GREEDY_GENERATION` and `RAW_SAMPLED_LOW_TEMP`, raw generation must not use ranked scoring, prefix forcing, decoder-assisted correction, response table lookup, expected-answer metadata, or oracle parsing.

Required records:

```text
raw_generation_path = autoregressive
decoder_assisted_used_for_raw = false
ranked_scoring_used_for_raw = false
prefix_forcing_used_for_raw = false
response_table_used_for_raw = false
prediction_oracle_used = false
```

Failure:

```text
RAW_GENERATION_PATH_CONTAMINATED
ORACLE_SHORTCUT_DETECTED
```

## Freshness And Freeze Gates

For every seed, fresh eval rows must not overlap with 101 eval, 102 train, 102 eval, 103 eval, or 100 train/eval. Every seed must also recheck the 102 checkpoint, source 100 checkpoint, and bounded release artifacts as read-only.

Required per seed:

```text
overlap_with_101_eval_count = 0
overlap_with_102_train_count = 0
overlap_with_102_eval_count = 0
overlap_with_103_eval_count = 0
overlap_with_100_train_eval_count = 0
max_prompt_jaccard_vs_102_train < 0.90
max_prompt_jaccard_vs_103_eval < 0.90
checkpoint_hash_unchanged = true
source_100_checkpoint_unchanged = true
bounded_release_artifact_unchanged = true
train_step_count = 0
optimizer_step_count = 0
```

Failure:

```text
TRAIN_EVAL_LEAKAGE_DETECTED
CHECKPOINT_MUTATION_DETECTED
BOUNDED_RELEASE_MUTATION_DETECTED
TRAINING_SIDE_EFFECT_DETECTED
```

## Positive Gate

Every seed must independently satisfy:

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
```

Aggregate gates:

```text
all_seeds_passed_independently = true
min_raw_free_generation_accuracy >= 0.85
max_case_id_drift_rate <= 0.10
max_slot_drift_rate <= 0.05
max_distractor_leak_rate <= 0.10
min_decoder_assisted_accuracy >= 0.90
min_bounded_chat_slot_binding_accuracy >= 0.90
min_finite_label_anchorroute_retention_accuracy >= 0.90
stddev_raw_free_generation_accuracy recorded
stddev_case_id_drift_rate recorded
```

Positive verdicts include:

```text
MULTI_SEED_RAW_GENERATION_CONFIRM_POSITIVE
UPSTREAM_103_FRESH_CONFIRM_VERIFIED
RAW_GENERATION_GENERALIZES_ALL_SEEDS
CASE_ID_ANCHOR_GENERALIZES_ALL_SEEDS
SLOT_PINNING_GENERALIZES_ALL_SEEDS
DECODER_ASSISTED_REFERENCE_RETAINED_ALL_SEEDS
UNSUPPORTED_REFUSAL_RETAINED_ALL_SEEDS
BOUNDED_CHAT_RETENTION_PASSES_ALL_SEEDS
FINITE_LABEL_ANCHORROUTE_RETENTION_PASSES_ALL_SEEDS
COLLAPSE_REJECTED_ALL_SEEDS
CHECKPOINT_UNCHANGED_ALL_SEEDS
NO_TRAINING_PERFORMED
GPT_LIKE_READINESS_NOT_CLAIMED
PRODUCTION_CHAT_NOT_CLAIMED
```

Failure verdicts include:

```text
MULTI_SEED_RAW_GENERATION_CONFIRM_FAILS
UPSTREAM_103_ARTIFACT_MISSING
UPSTREAM_103_NOT_POSITIVE
STALE_SEED_ARTIFACT_USED
MULTI_SEED_RAW_GENERATION_INSTABILITY_DETECTED
RAW_GENERATION_PATH_CONTAMINATED
ORACLE_SHORTCUT_DETECTED
TRAIN_EVAL_LEAKAGE_DETECTED
EVAL_ROW_MISMATCH
CASE_ID_ANCHOR_GENERALIZATION_FAILS
DISTRACTOR_NUMBER_COPY_DETECTED
SLOT_PINNING_GENERALIZATION_FAILS
DISTRACTOR_SUPPRESSION_REGRESSION_DETECTED
RETENTION_REGRESSION_DETECTED
CHECKPOINT_MUTATION_DETECTED
BOUNDED_RELEASE_MUTATION_DETECTED
TRAINING_SIDE_EFFECT_DETECTED
GPT_LIKE_READINESS_FALSE_CLAIM
PRODUCTION_CHAT_CLAIM_DETECTED
ROOT_LICENSE_CHANGED
```

## Decision Rule

If 104 passes:

```text
105_RAW_GENERATION_OOD_AND_BOUNDARY_CONFIRM
```

If any seed fails case ID:

```text
104B_CASE_ID_MULTI_SEED_FAILURE_ANALYSIS
```

If any seed fails retention:

```text
RETENTION_FAILURE_ANALYSIS
```

If any seed fails collapse:

```text
RAW_GENERATION_COLLAPSE_FAILURE_ANALYSIS
```

If any seed fails raw generation otherwise:

```text
104B_RAW_GENERATION_MULTI_SEED_FAILURE_ANALYSIS
```

## No Overclaim

104 must explicitly state:

- multi-seed raw-generation confirmation only
- not GPT-like assistant readiness
- not open-domain assistant readiness
- not production chat
- not public API
- not deployment readiness
- not public release
- not safety alignment
- Hungarian capability not claimed

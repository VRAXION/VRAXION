# STABLE_LOOP_PHASE_LOCK_101_FRESH_ASSISTANT_EVAL_AND_RAW_DECODER_FRONTIER_MAP Contract

## Summary

101 is fresh assistant frontier mapping after `STABLE_LOOP_PHASE_LOCK_100_OPEN_VOCAB_ASSISTANT_CAPABILITY_SCALE`.

It is eval-only. It performs no training, no optimizer steps, no decoder repair, no checkpoint mutation, no service/deploy/runtime changes, and no bounded release artifact changes. It improves no model capability. It maps raw generation against decoder-assisted and diagnostic modes so the next repair milestone can be selected mechanically.

This is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, not public release, and not safety alignment.

## Required Upstream

Require positive 100 root:

```text
target/pilot_wave/stable_loop_phase_lock_100_open_vocab_assistant_capability_scale/smoke
```

Required upstream verdict:

```text
OPEN_VOCAB_ASSISTANT_CAPABILITY_SCALE_POSITIVE
```

Required upstream facts:

```text
bounded_release_artifact_unchanged = true
source_094_checkpoint_unchanged = true
target_100_checkpoint_changed = true
raw_generated_prompt_response_accuracy recorded
decoder_repaired_generation_accuracy recorded
hungarian_basic_accuracy recorded
```

Failure:

```text
UPSTREAM_100_ARTIFACT_MISSING
UPSTREAM_100_NOT_POSITIVE
```

## Eval-Only Hard Wall

101 must not train, run optimizer steps, repair decoder behavior, mutate checkpoints, change 100 artifacts, change 099 bounded release artifacts, or modify runtime/service/deploy code.

Require:

```text
train_step_count = 0
optimizer_step_count = 0
checkpoint_hash_unchanged = true
bounded_release_artifact_unchanged = true
no_training_performed = true
```

Failure:

```text
TRAINING_SIDE_EFFECT_DETECTED
CHECKPOINT_MUTATION_DETECTED
BOUNDED_RELEASE_MUTATION_DETECTED
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
```

## Fixed Eval Modes

All eval modes must use identical rows and prompts:

```text
RAW_GREEDY_GENERATION
RAW_SAMPLED_GENERATION_LOW_TEMP
DECODER_ASSISTED_GENERATION
DECODER_ASSISTED_STRICT_BOUNDARY
PREFIX_FORCED_DIAGNOSTIC
RANKED_RESPONSE_SCORING
```

Record per mode:

```text
eval_row_hash
eval_prompt_hash
eval_count
eval_dataset_sha256
sampling_config
deterministic_or_stochastic
```

Failure:

```text
EVAL_ROW_MISMATCH
DECODE_POLICY_CHERRY_PICKING_DETECTED
DECODE_POLICY_MATRIX_MISSING
```

Prefix-forced and ranked scoring are diagnostic only. They must not be counted as free generation or used to claim assistant usability.

Failure:

```text
DIAGNOSTIC_MODE_MISCOUNTED_AS_FREE_GENERATION
```

## Fresh Families

Report every mode and every family:

```text
FRESH_SHORT_INSTRUCTION
FRESH_SHORT_EXPLANATION
FRESH_OPEN_DOMAIN_SIMPLE_QA
FRESH_OPEN_DOMAIN_UNSUPPORTED
FRESH_MULTI_TURN_CONTEXT_CARRY
FRESH_BOUNDARY_REFUSAL
FRESH_PROMPT_INJECTION
FRESH_HUNGARIAN_BASIC_CHAT
FRESH_ENGLISH_BASIC_CHAT
FRESH_ACTIVE_SLOT_BINDING
FRESH_STALE_DISTRACTOR_SUPPRESSION
FRESH_ANTI_REPETITION
FINITE_LABEL_ANCHORROUTE_RETENTION
BOUNDED_RELEASE_RETENTION
```

Required family metrics include:

```text
raw_accuracy
decoder_assisted_accuracy
ranked_accuracy
gap_raw_to_decoder
gap_raw_to_ranked
case_id_drift_rate
slot_drift_rate
distractor_leak_rate
refusal_accuracy
open_domain_answer_leak_rate
multi_turn_context_accuracy
hungarian_basic_accuracy
english_basic_accuracy
bounded_chat_slot_binding_accuracy
finite_label_anchorroute_retention_accuracy
```

Failure:

```text
FAMILY_FAILURE_MAP_INCOMPLETE
RAW_VS_DECODER_GAP_MISSING
FAILURE_MODE_CLASSIFICATION_MISSING
```

## Raw Drift Labels

Every failed raw row must receive one concrete label:

```text
case_id_drift
slot_drift
active_to_distractor_flip
stale_or_old_pocket_leak
refusal_garbled
unsupported_answer_leak
prompt_copy
repetition
early_stop
wrong_language
unknown_failure
```

Also record:

```text
first_error_token_position
first_error_byte_position
gold_prefix_survival_length
free_rollout_drift_rate
```

## Hard Gates

101 is diagnostic and can be positive even if raw generation remains weak. It cannot be positive if mapping is incomplete, retention regresses, artifacts mutate, training happens, or overclaim leakage appears.

Retention hard stop:

```text
bounded_chat_slot_binding_accuracy >= 0.80
finite_label_anchorroute_retention_accuracy >= 0.90
bounded_release_retention_pass = true
```

Overclaim leakage counts must be zero:

```text
gpt_like_claim_count = 0
production_chat_claim_count = 0
public_api_claim_count = 0
safety_alignment_claim_count = 0
open_domain_answer_leak_count = 0
```

Failure:

```text
RETENTION_REGRESSION_DETECTED
GPT_LIKE_READINESS_FALSE_CLAIM
PRODUCTION_CHAT_CLAIM_DETECTED
OPEN_DOMAIN_ANSWER_LEAK_DETECTED
```

## Decision Recommendation

`decision_recommendation.json` must be mechanically derived:

```text
raw fails and decoder-assisted succeeds -> 102_DECODER_POLICY_AND_ROLLOUT_REPAIR
raw and decoder-assisted both fail -> 102B_ASSISTANT_REPRESENTATION_OR_SFT_REPAIR
Hungarian fails but English/bounded pass -> HUNGARIAN_SFT_AND_EVAL_TRACK_LATER
open-domain QA fails but bounded/refusal pass -> OPEN_DOMAIN_KNOWLEDGE_GAP
retention regresses -> RETENTION_FAILURE_ANALYSIS
```

Failure:

```text
DECISION_RECOMMENDATION_MISSING
```

## Positive Verdicts

```text
FRESH_ASSISTANT_FRONTIER_MAP_POSITIVE
UPSTREAM_100_CAPABILITY_SCALE_VERIFIED
RAW_VS_DECODER_GAP_RECORDED
FRESH_ASSISTANT_EVAL_COMPLETED
FAMILY_FAILURE_MAP_WRITTEN
MULTI_TURN_SMOKE_RECORDED
HUNGARIAN_ENGLISH_SMOKE_RECORDED
RETENTION_RECHECKED
DECISION_RECOMMENDATION_WRITTEN
NO_TRAINING_PERFORMED
GPT_LIKE_READINESS_NOT_CLAIMED
PRODUCTION_CHAT_NOT_CLAIMED
```

## Failure Verdicts

```text
FRESH_ASSISTANT_FRONTIER_MAP_FAILS
UPSTREAM_100_ARTIFACT_MISSING
UPSTREAM_100_NOT_POSITIVE
EVAL_DATASET_BUILD_FAILS
RAW_GENERATION_EVAL_FAILS
DECODER_ASSISTED_EVAL_FAILS
RANKED_SCORING_EVAL_FAILS
RAW_VS_DECODER_GAP_MISSING
FAILURE_MODE_CLASSIFICATION_MISSING
DECISION_RECOMMENDATION_MISSING
RETENTION_REGRESSION_DETECTED
CHECKPOINT_MUTATION_DETECTED
TRAINING_SIDE_EFFECT_DETECTED
BOUNDED_RELEASE_MUTATION_DETECTED
EVAL_ROW_MISMATCH
DIAGNOSTIC_MODE_MISCOUNTED_AS_FREE_GENERATION
DECODE_POLICY_CHERRY_PICKING_DETECTED
DECODE_POLICY_MATRIX_MISSING
FAMILY_FAILURE_MAP_INCOMPLETE
OPEN_DOMAIN_ANSWER_LEAK_DETECTED
GPT_LIKE_READINESS_FALSE_CLAIM
PRODUCTION_CHAT_CLAIM_DETECTED
ROOT_LICENSE_CHANGED
```

## Required Outputs

Write target-only artifacts under:

```text
target/pilot_wave/stable_loop_phase_lock_101_fresh_assistant_eval_and_raw_decoder_frontier_map/
```

Required files include `queue.json`, `progress.jsonl`, `eval_config.json`, `upstream_100_manifest.json`, `checkpoint_integrity_manifest.json`, `eval_row_manifest.json`, `decode_policy_matrix.json`, all result JSONL files, family/mode/gap/drift/retention reports, `decision_recommendation.json`, `summary.json`, and `report.md`.

`progress.jsonl`, `summary.json`, and `report.md` must be written from start and refreshed through the run.

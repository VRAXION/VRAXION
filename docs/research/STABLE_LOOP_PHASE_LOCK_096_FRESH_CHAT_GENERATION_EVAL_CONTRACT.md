# STABLE_LOOP_PHASE_LOCK_096_FRESH_CHAT_GENERATION_EVAL Contract

096 is a fresh-row eval for the 095 target-only decoder repair.

094 proved that Chat SFT learned a ranked signal but raw generation was weak. 094B diagnosed the main issue as a stop-condition / decoder gap. 095 repaired generation without training by adding target-only, prompt-derived decoder constraints. 096 checks whether that repair generalizes to new deterministic eval rows rather than only fitting the 094 eval rows.

This is fresh-row decoder repair eval only. It is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not deployment, not public release, and not safety alignment.

## Allowed Inputs

- Positive 095 root: `target/pilot_wave/stable_loop_phase_lock_095_chat_decoder_generation_repair/smoke`
- 095 target checkpoint path, read for hash integrity only
- Deterministic fresh 096 prompt rows generated under target

## Hard Wall

096 must not train, optimize, mutate checkpoints, run service/deploy code, or use a response table / expected response for generation.

Required recorded values:

```text
optimizer_step_count = 0
no_training_performed = true
checkpoint_unchanged = true
expected_response_used_for_generation = false
response_table_used = false
prediction_oracle_used = false
llm_judge_used = false
```

## Freshness Gate

The fresh eval rows must be deterministic but not copied from the 095/094 eval rows.

Positive requires:

```text
overlap_with_095_eval_prompts = 0
overlap_with_095_eval_expected_responses = 0
fresh_eval_row_count >= 200
```

## Behavior Gates

Positive requires:

```text
fresh_generated_accuracy >= 0.90
bounded_slot_accuracy >= 0.90
finite_label_accuracy >= 0.90
unsupported_refusal_accuracy >= 0.90
prompt_copy_rate <= 0.05
repetition_rate <= 0.05
checkpoint_unchanged = true
```

## Required Artifacts

```text
queue.json
progress.jsonl
eval_config.json
upstream_095_manifest.json
checkpoint_integrity_manifest.json
fresh_eval_manifest.json
fresh_eval_dataset.jsonl
fresh_generation_results.jsonl
decoder_policy_manifest.json
family_metrics.json
collapse_metrics.json
freshness_validation.json
claim_boundary.json
human_readable_samples.jsonl
failure_case_samples.jsonl
summary.json
report.md
```

`progress.jsonl`, `summary.json`, and `report.md` are written from start and refreshed during the run.

## Verdicts

Positive:

```text
FRESH_CHAT_GENERATION_EVAL_POSITIVE
UPSTREAM_095_REPAIR_VERIFIED
FRESH_EVAL_ROWS_VERIFIED
FRESH_GENERATION_REPAIR_GENERALIZES
BOUNDED_CHAT_RETENTION_PASSES
FINITE_LABEL_ANCHORROUTE_RETENTION_PASSES
UNSUPPORTED_REFUSAL_PASSES
CHECKPOINTS_UNCHANGED
NO_TRAINING_PERFORMED
GPT_LIKE_READINESS_NOT_CLAIMED
```

Failure:

```text
FRESH_CHAT_GENERATION_EVAL_FAILS
UPSTREAM_095_ARTIFACT_MISSING
UPSTREAM_095_NOT_POSITIVE
FRESH_EVAL_ROW_OVERLAP_DETECTED
FRESH_GENERATION_REGRESSION_DETECTED
BOUNDED_CHAT_RETENTION_REGRESSION_DETECTED
FINITE_LABEL_RETENTION_REGRESSION_DETECTED
UNSUPPORTED_REFUSAL_REGRESSION_DETECTED
FRESH_GENERATION_COLLAPSE_DETECTED
CHECKPOINT_MUTATION_DETECTED
TRAINING_SIDE_EFFECT_DETECTED
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
GPT_LIKE_READINESS_FALSE_CLAIM
PRODUCTION_CHAT_CLAIM_DETECTED
PUBLIC_RELEASE_CLAIM_DETECTED
```

If 096 passes, the next autonomous milestone is `097_CHAT_DECODER_MULTI_SEED_OOD_RETENTION_CONFIRM`.

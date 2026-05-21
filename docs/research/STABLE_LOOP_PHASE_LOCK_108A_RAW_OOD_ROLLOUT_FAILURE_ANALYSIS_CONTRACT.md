# STABLE_LOOP_PHASE_LOCK_108A_RAW_OOD_ROLLOUT_FAILURE_ANALYSIS_CONTRACT

## Summary

108A is an analysis-only follow-up to `STABLE_LOOP_PHASE_LOCK_108_OPEN_DOMAIN_ASSISTANT_OOD_STRESS_AND_FAILURE_MAP_BATCH`.

108 found a clean hard-gate pass with a raw/decoder OOD gap:

```text
raw_ood_stress_accuracy ~= 0.5263
decoder_ood_stress_accuracy = 1.0
raw_vs_decoder_ood_gap ~= 0.4737
unknown_failure_rate = 0.0
retention = 1.0
overclaim/exfiltration = 0
checkpoint unchanged = true
train_step_count = 0
```

108A explains that raw gap. It does not train, repair, mutate checkpoints, change runtime/service/deploy code, or improve model capability.

This milestone is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, and not safety alignment.

## Required Inputs

Require positive upstream 108 at:

```text
target/pilot_wave/stable_loop_phase_lock_108_open_domain_assistant_ood_stress_and_failure_map_batch/smoke
```

The runner reads:

```text
summary.json
multi_seed_aggregate.json
checkpoint_integrity_manifest.json
raw_generation_results.jsonl
decoder_repaired_results.jsonl
failure_mode_map.json
raw_vs_decoder_ood_gap.json
```

Rows are paired strictly by:

```text
seed
eval_index
eval_family
```

`RAW_DECODER_PAIRING_FAILS` is emitted if row counts, keys, or raw-fail / decoder-pass disagreements are missing.

## Analysis Rules

Analyze only rows where:

```text
RAW_FREE_GENERATION failed
DECODER_REPAIRED_GENERATION passed
```

For every analyzed row, record:

```text
expected_response_source
expected_response
decoder_output
raw_output
primary_surface_failure_label
likely_mechanism_label
divergence_point
first_wrong_token_position
matching_prefix_token_count
expected_token_count
gold_prefix_survival_rate
```

Scoring is deterministic and rubric-only. No LLM judge and no oracle shortcut are allowed.

Tokenization for prefix metrics:

```text
lowercase alnum/underscore chunks plus punctuation tokens
```

108A intentionally separates visible symptoms from likely causes:

```text
primary_surface_failure_label = the visible output error
likely_mechanism_label = the likely rollout/decoder/training cause
```

This prevents over-attributing every raw failure to a surface symptom when the deeper issue is prefix loss, context carry, prompt format sensitivity, or decoder-policy gap.

## Required Artifacts

Write generated outputs only under:

```text
target/pilot_wave/stable_loop_phase_lock_108a_raw_ood_rollout_failure_analysis/
```

Required artifacts:

```text
queue.json
progress.jsonl
analysis_config.json
upstream_108_manifest.json
raw_decoder_pair_manifest.json
raw_failure_attribution.json
raw_failure_cases.jsonl
raw_decoder_disagreement.jsonl
first_error_position_report.json
prefix_survival_report.json
rollout_drift_report.json
stop_condition_report.json
family_failure_breakdown.json
recommended_repair_plan.json
human_readable_samples.jsonl
summary.json
report.md
```

`progress.jsonl`, `summary.json`, and `report.md` must be written from start and refreshed after upstream verification, pair loading, attribution, report writing, repair-plan writing, and final verdict.

## Positive Gate

Positive requires:

```text
RAW_OOD_ROLLOUT_FAILURE_ANALYSIS_POSITIVE
raw_decoder_disagreement_count > 0
all raw-fail / decoder-pass rows attributed
unknown_raw_failure_rate <= 0.10
human_readable_samples.jsonl present
recommended_repair_plan.json present
train_step_count = 0
optimizer_step_count = 0
checkpoint_hash_unchanged = true
bounded_release_artifact_unchanged = true
```

Failure verdicts include:

```text
RAW_OOD_ROLLOUT_FAILURE_ANALYSIS_FAILS
UPSTREAM_108_ARTIFACT_MISSING
UPSTREAM_108_NOT_POSITIVE
RAW_DECODER_PAIRING_FAILS
EXPECTED_RESPONSE_MISSING
RAW_FAILURE_ATTRIBUTION_INCOMPLETE
UNKNOWN_RAW_FAILURE_RATE_TOO_HIGH
HUMAN_SAMPLE_REPORT_MISSING
REPAIR_PLAN_MISSING
CHECKPOINT_MUTATION_DETECTED
TRAINING_SIDE_EFFECT_DETECTED
GPT_LIKE_READINESS_FALSE_CLAIM
PRODUCTION_CHAT_CLAIM_DETECTED
ROOT_LICENSE_CHANGED
```

## Repair Recommendation

`recommended_repair_plan.json` must choose exactly one main next step:

```text
109_DECODER_POLICY_INTEGRATION
109_RAW_ROLLOUT_REPAIR
109_SFT_ROLLOUT_DATA_REPAIR
109_STOP_CONDITION_REPAIR
109_PROMPT_FORMAT_REPAIR
```

Decision rule:

```text
if decoder_success_on_raw_fail_rate >= 0.95 and raw_vs_decoder_ood_gap >= 0.25:
  next = 109_DECODER_POLICY_INTEGRATION

else if prefix loss or first-token wrong dominates:
  next = 109_RAW_ROLLOUT_REPAIR

else if stop-condition failure dominates:
  next = 109_STOP_CONDITION_REPAIR

else if prompt-format sensitivity dominates:
  next = 109_PROMPT_FORMAT_REPAIR

else:
  next = 109_SFT_ROLLOUT_DATA_REPAIR
```

The plan must also include `secondary_next_if_decoder_integration_fails`.

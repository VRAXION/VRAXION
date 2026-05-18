# STABLE_LOOP_PHASE_LOCK_094B_CHAT_SFT_FREE_GENERATION_GAP_ANALYSIS Contract

## Summary

094B is an analysis-only diagnosis after 094.

094 proved Chat SFT learning and retention, but exposed a free-generation gap:

```text
ranked_prompt_response_accuracy = 1.0
generated_prompt_response_accuracy = 0.13125
generation_gap = 0.86875
WARMSTART_ADVANTAGE_NOT_PROVEN
```

094B identifies whether the gap is caused by decode policy, stop conditions, prompt format, prefix instability, exposure-bias rollout drift, byte-level local minima, refusal overgeneralization, finite-label weakness, or unproven warm-start advantage.

094B is analysis only. It does not train, repair, mutate checkpoints, improve model capability, prove GPT-like assistant readiness, prove open-domain assistant readiness, prove production chat, create deployment readiness, create a public release, or prove safety alignment.

## Key Changes

Add only:

```text
scripts/probes/run_stable_loop_phase_lock_094b_chat_sft_free_generation_gap_analysis.py
scripts/probes/run_stable_loop_phase_lock_094b_chat_sft_free_generation_gap_analysis_check.py
docs/research/STABLE_LOOP_PHASE_LOCK_094B_CHAT_SFT_FREE_GENERATION_GAP_ANALYSIS_CONTRACT.md
docs/research/STABLE_LOOP_PHASE_LOCK_094B_CHAT_SFT_FREE_GENERATION_GAP_ANALYSIS_RESULT.md
```

Generated outputs stay under:

```text
target/pilot_wave/stable_loop_phase_lock_094b_chat_sft_free_generation_gap_analysis/
```

Do not modify runtime/service/deploy code, SDK/public exports, product/release docs, root LICENSE, existing checkpoints, 093 artifacts, 094 artifacts, or 083/089 packages.

Hard wall:

```text
optimizer_step_count = 0
no_training_performed = true
source_093_checkpoint_unchanged = true
target_094_checkpoint_unchanged = true
```

Failures:

```text
TRAINING_SIDE_EFFECT_DETECTED
CHECKPOINT_MUTATION_DETECTED
RUNTIME_SURFACE_MUTATION_DETECTED
```

## Runner Behavior

Default smoke:

```powershell
python scripts/probes/run_stable_loop_phase_lock_094b_chat_sft_free_generation_gap_analysis.py --out target/pilot_wave/stable_loop_phase_lock_094b_chat_sft_free_generation_gap_analysis/smoke --upstream-094-root target/pilot_wave/stable_loop_phase_lock_094_open_vocab_chat_sft_mix_poc/smoke --seed 2028 --heartbeat-sec 20
```

Require upstream 094:

```text
OPEN_VOCAB_CHAT_SFT_MIX_POC_POSITIVE
ranked_prompt_response_accuracy = 1.0
generated_prompt_response_accuracy = 0.13125
generation_gap = 0.86875
WARMSTART_ADVANTAGE_NOT_PROVEN present
source_093_checkpoint_unchanged = true
target_sft_checkpoint_changed = true
```

Run identical eval rows across:

```text
greedy
top_k_1
top_k_4_temp_0.4
top_k_8_temp_0.6
top_k_24_temp_0.7
top_k_24_temp_0.85
nucleus_p_0.9_temp_0.7
expected-prefix-forced first 8 bytes
expected-prefix-forced first 16 bytes
expected-prefix-forced first 32 bytes
```

Prefix forcing is diagnostic only and must not be counted as repaired free generation.

## Required Metrics

For every decode policy:

```text
eval_row_hash
eval_row_count
eval_dataset_sha256
generated_accuracy
bounded_slot_accuracy
finite_label_accuracy
unsupported_refusal_accuracy
prompt_copy_rate
train_response_copy_rate
repetition_rate
static_rate
average_output_length
stop_reason_distribution
first_error_byte_position
entropy_profile
```

For the ranked-vs-generated gap:

```text
expected_response_loss
generated_response_loss
best_non_expected_response_loss
rank_margin
gold_prefix_survival_rate
free_rollout_drift_rate
gap_after_prefix_forcing
```

Failure classification must use one primary label from:

```text
DECODE_POLICY_TOO_STOCHASTIC
GREEDY_DECODE_COLLAPSE
STOP_CONDITION_MISMATCH
PROMPT_FORMAT_MISMATCH
EXPOSURE_BIAS_ROLLOUT_DRIFT
BYTE_LEVEL_LOCAL_MINIMUM
EXPECTED_RESPONSE_PREFIX_NOT_STABLE
REFUSAL_TEMPLATE_OVERGENERALIZATION
FINITE_LABEL_OUTPUT_WEAKNESS
WARMSTART_ADVANTAGE_NOT_PROVEN_CONFIRMED
```

and recommend either `095_CHAT_DECODER_GENERATION_REPAIR` or `095B_CHAT_SFT_DATA_AND_ROLLOUT_REPAIR`.

## Verdicts

Positive:

```text
CHAT_SFT_FREE_GENERATION_GAP_ANALYSIS_POSITIVE
UPSTREAM_094_SFT_POC_VERIFIED
RANKED_GENERATED_GAP_CONFIRMED
DECODE_POLICY_MATRIX_RECORDED
ROLLOUT_DRIFT_ANALYZED
PREFIX_FORCING_ANALYZED
STOP_CONDITION_ANALYZED
FAILURE_MODE_CLASSIFIED
NO_TRAINING_PERFORMED
CHECKPOINTS_UNCHANGED
GPT_LIKE_READINESS_NOT_CLAIMED
```

Failures include:

```text
CHAT_SFT_FREE_GENERATION_GAP_ANALYSIS_FAILS
UPSTREAM_094_ARTIFACT_MISSING
UPSTREAM_094_NOT_POSITIVE
TRAINING_SIDE_EFFECT_DETECTED
CHECKPOINT_MUTATION_DETECTED
DECODE_POLICY_EVAL_ROW_MISMATCH
DECODE_POLICY_MATRIX_MISSING
RANKED_GENERATED_GAP_MISSING
FAILURE_MODE_CLASSIFICATION_MISSING
HUMAN_SAMPLE_REPORT_MISSING
LLM_JUDGE_USED
ORACLE_SHORTCUT_DETECTED
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
GPT_LIKE_READINESS_FALSE_CLAIM
PRODUCTION_CHAT_CLAIM_DETECTED
PUBLIC_RELEASE_CLAIM_DETECTED
```

## Next

Decoder, prompt, or stop-condition diagnosis routes to `095_CHAT_DECODER_GENERATION_REPAIR`.

Exposure-bias, SFT-data, rollout-drift, or weak-token-objective diagnosis routes to `095B_CHAT_SFT_DATA_AND_ROLLOUT_REPAIR`.

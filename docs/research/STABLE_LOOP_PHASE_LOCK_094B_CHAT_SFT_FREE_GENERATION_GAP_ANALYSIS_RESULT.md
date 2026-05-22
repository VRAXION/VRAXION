# STABLE_LOOP_PHASE_LOCK_094B_CHAT_SFT_FREE_GENERATION_GAP_ANALYSIS Result

## Status

094B implements an analysis-only diagnosis of the 094 ranked-vs-free-generation gap.

Expected smoke root:

```text
target/pilot_wave/stable_loop_phase_lock_094b_chat_sft_free_generation_gap_analysis/smoke
```

094B is analysis only. It records evidence and next repair direction, but it does not train, repair, mutate checkpoints, improve model capability, prove GPT-like assistant readiness, prove open-domain assistant readiness, prove production chat, create deployment readiness, create a public release, or prove safety alignment.

## Expected Evidence

Required artifacts:

```text
queue.json
progress.jsonl
analysis_config.json
upstream_094_manifest.json
checkpoint_integrity_manifest.json
eval_row_manifest.json
decode_policy_matrix.json
decode_policy_results.jsonl
ranked_vs_generated_gap.json
rollout_drift_analysis.json
prefix_forcing_analysis.json
stop_condition_analysis.json
prompt_format_analysis.json
failure_mode_classification.json
human_readable_samples.jsonl
failure_case_samples.jsonl
summary.json
report.md
```

`human_readable_samples.jsonl` must pair:

```text
ranked expected response
baseline 094 generation
best decode policy generation
prefix-forced generation
```

across:

```text
short instruction
simple dialogue
bounded active slot
context carry
unsupported refusal
boundary/injection refusal
finite label retention
```

## Positive Verdicts

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

## Failure Verdicts

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

## Interpretation

Passing 094B means the free-generation gap is diagnosed and a repair milestone is selected.

Passing 094B does not mean better generation, GPT-like readiness, open-domain assistant readiness, production chat, deployment readiness, public release, or safety alignment.

## Validation

```powershell
python -m py_compile scripts/probes/run_stable_loop_phase_lock_094b_chat_sft_free_generation_gap_analysis.py
python scripts/probes/run_stable_loop_phase_lock_094b_chat_sft_free_generation_gap_analysis.py --out target/pilot_wave/stable_loop_phase_lock_094b_chat_sft_free_generation_gap_analysis/smoke --upstream-094-root target/pilot_wave/stable_loop_phase_lock_094_open_vocab_chat_sft_mix_poc/smoke --seed 2028 --heartbeat-sec 20
python -m py_compile scripts/probes/run_stable_loop_phase_lock_094b_chat_sft_free_generation_gap_analysis_check.py
python scripts/probes/run_stable_loop_phase_lock_094b_chat_sft_free_generation_gap_analysis_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_094_open_vocab_chat_sft_mix_poc_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_093_open_vocab_fineweb_margin_scale_confirm_check.py --check-only
git diff --check
```

## Next

If 094B points to decoder, prompt, or stop-condition issues, proceed to `095_CHAT_DECODER_GENERATION_REPAIR`.

If 094B points to exposure bias, SFT data shape, rollout drift, or weak token objective, proceed to `095B_CHAT_SFT_DATA_AND_ROLLOUT_REPAIR`.

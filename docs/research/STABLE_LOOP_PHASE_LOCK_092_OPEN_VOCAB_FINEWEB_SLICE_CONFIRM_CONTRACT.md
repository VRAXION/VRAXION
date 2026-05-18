# STABLE_LOOP_PHASE_LOCK_092_OPEN_VOCAB_FINEWEB_SLICE_CONFIRM Contract

## Summary

092 confirms the 091 runner-local byte-level next-byte LM signal on a real local FineWeb-Edu slice.

Default source:

```text
S:\AI\MESSY TRAINING DATA - INPUT ONLY\Fineweb edu 10B\fineweb_edu_30m.txt
```

Allowed claim:

```text
open-vocab next-byte LM signal transfers to a local FineWeb-Edu slice above controls
```

Forbidden claims:

```text
GPT-like assistant readiness
open-domain assistant readiness
INSTNCT/AnchorRoute proven as open-domain LM winner
production chat
deployment readiness
safety alignment
public release
```

092 is a runner-local PyTorch LM corpus confirm only. It does not train the packaged bounded winner and does not claim that INSTNCT/AnchorRoute is now an open-domain architecture winner.

## Key Changes

Add only:

```text
scripts/probes/run_stable_loop_phase_lock_092_open_vocab_fineweb_slice_confirm.py
scripts/probes/run_stable_loop_phase_lock_092_open_vocab_fineweb_slice_confirm_check.py
docs/research/STABLE_LOOP_PHASE_LOCK_092_OPEN_VOCAB_FINEWEB_SLICE_CONFIRM_CONTRACT.md
docs/research/STABLE_LOOP_PHASE_LOCK_092_OPEN_VOCAB_FINEWEB_SLICE_CONFIRM_RESULT.md
```

Generated outputs stay under:

```text
target/pilot_wave/stable_loop_phase_lock_092_open_vocab_fineweb_slice_confirm/
```

092 must not modify:

```text
instnct-core/
tools/instnct_service_alpha/
tools/instnct_deploy/
SDK/public exports
docs/product/
docs/releases/
root LICENSE
existing checkpoints
083/089 packages
```

## Runner Requirements

Require upstream 091 root:

```text
target/pilot_wave/stable_loop_phase_lock_091_open_vocab_chat_lm_foundation/smoke
```

Require:

```text
OPEN_VOCAB_CHAT_LM_FOUNDATION_POSITIVE
bounded retention pass
leakage audit pass
packaged winner hash unchanged
architecture_winner_for_open_vocab_claimed = false
```

FineWeb source hard wall:

```text
FINEWEB_SLICE_MISSING
```

If the configured FineWeb source is absent, fail before training. Do not fall back to the 091 fixture and do not silently substitute another corpus.

FineWeb source immutability must record:

```text
fineweb_source_path
fineweb_source_size_bytes
fineweb_source_mtime_before
fineweb_source_mtime_after
fineweb_source_sha256_before
fineweb_source_sha256_after
fineweb_source_hash_unchanged
```

Failure:

```text
FINEWEB_SOURCE_MUTATION_DETECTED
```

092 LM train/eval split must come from FineWeb text only:

```text
fineweb_train_token_count
fineweb_eval_token_count
bounded_retention_rows_in_lm_train = 0
bounded_retention_rows_in_lm_eval = 0
```

Failure:

```text
DATASET_MIX_CONTAMINATION_DETECTED
```

Split proof must record:

```text
split_seed
corpus_sha256
train_split_sha256
eval_split_sha256
eval_row_hash
eval_token_hash
eval_token_count
train_eval_exact_text_overlap_count
max_train_eval_jaccard
```

Positive requires exact text overlap `0` and max Jaccard `< 0.90`.

## Model And Controls

Use:

```text
runner_local_pytorch_lm = true
byte_level tokenizer
causal_next_byte decoder path
byte ids 0..255 plus BOS/EOS/PAD
response_table_used_for_main_prediction = false
prediction_oracle_used = false
llm_judge_used = false
packaged_bounded_winner_trained = false
architecture_winner_for_open_vocab_claimed = false
```

All arms must use identical eval tokens:

```text
OPEN_VOCAB_FINEWEB_BYTE_LM_MAIN
CHAR_BIGRAM_BASELINE
RANDOM_BYTE_CONTROL
SHUFFLED_TARGET_CONTROL
STATIC_OUTPUT_CONTROL
COPY_PROMPT_CONTROL
```

Per arm record:

```text
eval_token_hash
eval_token_count
eval_row_hash
```

Failure:

```text
BASELINE_EVAL_MISMATCH
```

Positive requires:

```text
main eval_loss < char_bigram eval_loss
main eval_loss < shuffled_target eval_loss by >= 0.25
main next_byte_accuracy > random_byte_control by >= 0.10
```

Record:

```text
delta_vs_char_bigram_loss
delta_vs_shuffled_target_loss
delta_vs_random_accuracy
```

## Gates

Training proof:

```text
train_step_count > 0
checkpoint_after_hash != checkpoint_before_hash
train_loss_final < train_loss_initial
eval_loss
eval_perplexity
next_byte_accuracy
```

Generation cannot collapse:

```text
nonempty_generation_rate >= 0.98
utf8_valid_generation_rate >= 0.80
empty_output_rate <= 0.02
static_output_rate <= 0.15
repetition_rate <= 0.25
copy_prompt_rate <= 0.20
unique_generated_3gram_count
unique_generated_5gram_count
generated_byte_entropy
```

Generation smoke uses deterministic top-k printable-byte sampling from the runner-local LM distribution. It is not response-table decoding and not an open-domain assistant claim.

Retention is separate from FineWeb LM loss:

```text
bounded_chat_slot_binding_accuracy >= 0.80
finite_label_anchorroute_retention_accuracy >= 0.90
unsupported_refusal_accuracy >= 0.80
```

Packaged winner integrity:

```text
packaged_winner_hash_before
packaged_winner_hash_after
packaged_winner_hash_unchanged = true
no_training_on_packaged_checkpoint = true
```

Environment and determinism record:

```text
seed
python_version
torch_version
device
cuda_available
deterministic_algorithms_requested
platform
wall_clock_sec
```

If CUDA nondeterminism is possible, limitations must state it. The smoke uses CPU for deterministic behavior.

## Required Artifacts

Write:

```text
queue.json
progress.jsonl
training_config.json
upstream_091_manifest.json
fineweb_source_manifest.json
dataset_manifest.json
tokenizer_manifest.json
train_examples_sample.jsonl
eval_examples_sample.jsonl
training_metrics.jsonl
checkpoint_manifest.json
checkpoint_hashes.json
generation_samples.jsonl
human_readable_samples.jsonl
lm_metrics.json
fineweb_generation_metrics.json
bounded_retention_metrics.json
collapse_metrics.json
leakage_metrics.json
arm_comparison.json
control_delta_report.json
failure_case_samples.jsonl
summary.json
report.md
```

`progress.jsonl`, `summary.json`, and `report.md` must be written from start and refreshed after upstream verification, FineWeb source verification, dataset split, tokenizer build, training heartbeat, eval, bounded retention, controls, and final verdict.

`human_readable_samples.jsonl` must include:

```text
short English continuation
unseen word continuation
mixed punctuation continuation
number/text continuation
simple dialogue continuation
unsupported-domain refusal continuation
```

Each row includes:

```text
arm
prompt
generated_text
expected_behavior
pass_fail
utf8_valid
nonempty
repetition_flag
copy_prompt_flag
bounded_retention_flag
short_diagnosis
```

Failure:

```text
HUMAN_SAMPLE_REPORT_MISSING
```

## Verdicts

Positive:

```text
OPEN_VOCAB_FINEWEB_SLICE_CONFIRM_POSITIVE
UPSTREAM_091_FOUNDATION_VERIFIED
FINEWEB_SOURCE_VERIFIED_READ_ONLY
FINEWEB_SPLIT_DETERMINISTIC
BYTE_LEVEL_TOKENIZER_BUILT
OPEN_VOCAB_FINEWEB_TRAINING_COMPLETED
TOKEN_OBJECTIVE_LEARNED
MAIN_BEATS_FINEWEB_CONTROLS
OPEN_VOCAB_GENERATION_SMOKE_PASSES
BOUNDED_CHAT_RETENTION_PASSES
FINITE_LABEL_ANCHORROUTE_RETENTION_PASSES
LEAKAGE_AUDIT_PASSES
COLLAPSE_REJECTED
NO_TRAINING_ON_PACKAGED_CHECKPOINT
PRODUCTION_CHAT_NOT_CLAIMED
GPT_LIKE_READINESS_NOT_CLAIMED
```

Failure:

```text
OPEN_VOCAB_FINEWEB_SLICE_CONFIRM_FAILS
UPSTREAM_091_ARTIFACT_MISSING
UPSTREAM_091_NOT_POSITIVE
FINEWEB_SLICE_MISSING
FINEWEB_SOURCE_MUTATION_DETECTED
DATASET_MIX_CONTAMINATION_DETECTED
TRAIN_EVAL_LEAKAGE_DETECTED
BASELINE_EVAL_MISMATCH
NO_ACTUAL_TRAINING_UPDATE_DETECTED
TOKEN_OBJECTIVE_NOT_LEARNED
CONTROL_DELTA_INSUFFICIENT
OPEN_VOCAB_GENERATION_SMOKE_FAILS
EMPTY_OUTPUT_COLLAPSE_DETECTED
STATIC_RESPONSE_COLLAPSE_DETECTED
REPETITION_COLLAPSE_DETECTED
BOUNDED_CHAT_RETENTION_REGRESSION_DETECTED
FINITE_LABEL_RETENTION_REGRESSION_DETECTED
PACKAGED_CHECKPOINT_MUTATION_DETECTED
TRAINING_SIDE_EFFECT_ON_PACKAGED_CHECKPOINT
ORACLE_SHORTCUT_DETECTED
LLM_JUDGE_USED
HUMAN_SAMPLE_REPORT_MISSING
ARCHITECTURE_WINNER_FALSE_CLAIM
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
GPT_LIKE_READINESS_FALSE_CLAIM
PRODUCTION_CHAT_CLAIM_DETECTED
PUBLIC_RELEASE_CLAIM_DETECTED
```

## Validation

```powershell
python -m py_compile scripts/probes/run_stable_loop_phase_lock_092_open_vocab_fineweb_slice_confirm.py
python -m py_compile scripts/probes/run_stable_loop_phase_lock_092_open_vocab_fineweb_slice_confirm_check.py
python scripts/probes/run_stable_loop_phase_lock_092_open_vocab_fineweb_slice_confirm.py --out target/pilot_wave/stable_loop_phase_lock_092_open_vocab_fineweb_slice_confirm/smoke --upstream-091-root target/pilot_wave/stable_loop_phase_lock_091_open_vocab_chat_lm_foundation/smoke --fineweb-source "S:\AI\MESSY TRAINING DATA - INPUT ONLY\Fineweb edu 10B\fineweb_edu_30m.txt" --seed 2026 --train-tokens 1000000 --eval-tokens 200000 --seq-len 128 --batch-size 32 --steps 3000 --heartbeat-sec 20
python scripts/probes/run_stable_loop_phase_lock_092_open_vocab_fineweb_slice_confirm_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_091_open_vocab_chat_lm_foundation_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_089b_packaged_model_winner_repro_train_proof_check.py --check-only
git diff --check
```

## Assumptions

The default FineWeb source exists locally and is read-only input. 092 trains only a new target-local PyTorch research checkpoint. Bounded retention is evaluated separately from FineWeb LM loss.

If 092 passes, next milestone is `093_OPEN_VOCAB_CHAT_SFT_MIX_POC`. If 092 fails, next milestone is `092B_FINEWEB_SLICE_FAILURE_ANALYSIS`.

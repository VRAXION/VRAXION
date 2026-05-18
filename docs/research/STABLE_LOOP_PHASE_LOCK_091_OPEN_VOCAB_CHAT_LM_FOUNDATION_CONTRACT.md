# STABLE_LOOP_PHASE_LOCK_091_OPEN_VOCAB_CHAT_LM_FOUNDATION Contract

## Summary

091 is the first open-vocab / byte-token LM foundation sanity gate after the 089B bounded winner proof. It tests whether a runner-local PyTorch next-byte LM can learn above baselines on a deterministic text mix while preserving the bounded winner as an untouched retention reference.

Allowed claim:

```text
runner-local open-vocab next-byte LM foundation sanity passed
```

Forbidden claims:

```text
packaged bounded winner itself is now GPT-like
INSTNCT is proven as open-domain LM
production/open-domain assistant readiness
```

091 is open-vocab LM foundation sanity only, not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public release, not safety alignment, and not deployment.

## Key Changes

Add only:

```text
scripts/probes/run_stable_loop_phase_lock_091_open_vocab_chat_lm_foundation.py
scripts/probes/run_stable_loop_phase_lock_091_open_vocab_chat_lm_foundation_check.py
docs/research/STABLE_LOOP_PHASE_LOCK_091_OPEN_VOCAB_CHAT_LM_FOUNDATION_CONTRACT.md
docs/research/STABLE_LOOP_PHASE_LOCK_091_OPEN_VOCAB_CHAT_LM_FOUNDATION_RESULT.md
```

Generated outputs go only under:

```text
target/pilot_wave/stable_loop_phase_lock_091_open_vocab_chat_lm_foundation/
```

Do not modify `instnct-core/`, service API, deploy harness, SDK/public exports, `docs/product/`, `docs/releases/`, root `LICENSE`, existing checkpoints, or 083/089 packages.

## Required Upstream

Require:

```text
target/pilot_wave/stable_loop_phase_lock_089b_packaged_model_winner_repro_train_proof/smoke
PACKAGED_MODEL_WINNER_REPRO_TRAIN_PROOF_POSITIVE
package_hash_binding_pass = true
repro_training_pass = true
winner_beats_controls = true
tamper_controls_pass = true
leakage_controls_pass = true
```

Failures:

```text
UPSTREAM_089B_ARTIFACT_MISSING
UPSTREAM_089B_NOT_POSITIVE
```

## Runner Requirements

Default command:

```text
python scripts/probes/run_stable_loop_phase_lock_091_open_vocab_chat_lm_foundation.py --out target/pilot_wave/stable_loop_phase_lock_091_open_vocab_chat_lm_foundation/smoke --upstream-089b-root target/pilot_wave/stable_loop_phase_lock_089b_packaged_model_winner_repro_train_proof/smoke --seed 2026 --train-tokens 250000 --eval-tokens 50000 --seq-len 128 --batch-size 32 --steps 1200 --heartbeat-sec 20
```

Record architecture boundary:

```text
runner_local_pytorch_lm = true
packaged_winner_checkpoint_trained = false
packaged_winner_used_for_retention_reference = true
architecture_winner_for_open_vocab_claimed = false
```

Tokenizer/model:

```text
byte-level tokenizer, ids 0..255 plus BOS/EOS/PAD
vocab_size >= 259
runner-local PyTorch causal next-byte LM
decoder_path = causal_next_byte
response_table_used_for_main_prediction = false
prediction_oracle_used = false
llm_judge_used = false
```

Corpus provenance:

```text
corpus_source
corpus_sha256
train_split_sha256
eval_split_sha256
train_eval_exact_text_overlap_count
max_train_eval_jaccard
split_seed
```

Hard leakage fail:

```text
train_eval_exact_text_overlap_count > 0
max_train_eval_jaccard >= 0.90
```

Failure:

```text
TRAIN_EVAL_LEAKAGE_DETECTED
```

## Arms And Gates

All arms use the same split and record `eval_row_hash`, `eval_token_hash`, and `eval_token_count`:

```text
OPEN_VOCAB_BYTE_LM_MAIN
CHAR_BIGRAM_BASELINE
RANDOM_BYTE_CONTROL
SHUFFLED_TARGET_CONTROL
BOUNDED_ONLY_TRAIN_CONTROL
NO_ANCHOR_RETENTION_MIX_CONTROL
STATIC_OUTPUT_CONTROL
COPY_PROMPT_CONTROL
```

Positive control deltas:

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

Training proof:

```text
train_step_count > 0
checkpoint_after_hash != checkpoint_before_hash
train_loss_final < train_loss_initial
```

Generation quality gates:

```text
nonempty_generation_rate >= 0.98
utf8_valid_generation_rate >= 0.80
empty_output_rate <= 0.02
static_output_rate <= 0.15
repetition_rate <= 0.25
copy_prompt_rate <= 0.20
```

Bounded retention reference gates:

```text
bounded_chat_slot_binding_accuracy >= 0.80
finite_label_anchorroute_retention_accuracy >= 0.90
unsupported_refusal_accuracy >= 0.80
packaged_winner_hash_unchanged = true
no_training_on_packaged_checkpoint = true
```

Failures include:

```text
TOKENIZER_BUILD_FAILS
DATASET_BUILD_FAILS
BASELINE_EVAL_MISMATCH
NO_ACTUAL_TRAINING_UPDATE_DETECTED
TOKEN_OBJECTIVE_NOT_LEARNED
CONTROL_DELTA_INSUFFICIENT
OPEN_VOCAB_GENERATION_SMOKE_FAILS
BOUNDED_CHAT_RETENTION_REGRESSION_DETECTED
FINITE_LABEL_RETENTION_REGRESSION_DETECTED
EMPTY_OUTPUT_COLLAPSE_DETECTED
STATIC_RESPONSE_COLLAPSE_DETECTED
REPETITION_COLLAPSE_DETECTED
PACKAGED_CHECKPOINT_MUTATION_DETECTED
TRAINING_SIDE_EFFECT_ON_PACKAGED_CHECKPOINT
ORACLE_SHORTCUT_DETECTED
LLM_JUDGE_USED
ARCHITECTURE_WINNER_FALSE_CLAIM
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
GPT_LIKE_READINESS_FALSE_CLAIM
PRODUCTION_CHAT_CLAIM_DETECTED
PUBLIC_RELEASE_CLAIM_DETECTED
```

Positive verdicts:

```text
OPEN_VOCAB_CHAT_LM_FOUNDATION_POSITIVE
UPSTREAM_089B_WINNER_PROOF_VERIFIED
BYTE_LEVEL_TOKENIZER_BUILT
OPEN_VOCAB_NEXT_BYTE_TRAINING_COMPLETED
TOKEN_OBJECTIVE_LEARNED
MAIN_BEATS_CONTROLS
OPEN_VOCAB_GENERATION_SMOKE_PASSES
BOUNDED_CHAT_RETENTION_PASSES
FINITE_LABEL_ANCHORROUTE_RETENTION_PASSES
LEAKAGE_AUDIT_PASSES
COLLAPSE_REJECTED
NO_TRAINING_ON_PACKAGED_CHECKPOINT
PRODUCTION_CHAT_NOT_CLAIMED
GPT_LIKE_READINESS_NOT_CLAIMED
```

## Required Artifacts

Write:

```text
queue.json
progress.jsonl
training_config.json
upstream_089b_manifest.json
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
open_vocab_generation_metrics.json
bounded_retention_metrics.json
collapse_metrics.json
leakage_metrics.json
arm_comparison.json
control_delta_report.json
failure_case_samples.jsonl
summary.json
report.md
```

`progress.jsonl`, `summary.json`, and `report.md` must be written from start and refreshed after dataset build, tokenizer build, training heartbeat, eval, bounded retention, controls, and final verdict.

Human-readable samples must include short English continuation, unseen word continuation, mixed punctuation continuation, number/text continuation, simple dialogue continuation, and unsupported-domain refusal continuation.

## Validation

```text
python -m py_compile scripts/probes/run_stable_loop_phase_lock_091_open_vocab_chat_lm_foundation.py
python scripts/probes/run_stable_loop_phase_lock_091_open_vocab_chat_lm_foundation.py --out target/pilot_wave/stable_loop_phase_lock_091_open_vocab_chat_lm_foundation/smoke --upstream-089b-root target/pilot_wave/stable_loop_phase_lock_089b_packaged_model_winner_repro_train_proof/smoke --seed 2026 --train-tokens 250000 --eval-tokens 50000 --seq-len 128 --batch-size 32 --steps 1200 --heartbeat-sec 20
python -m py_compile scripts/probes/run_stable_loop_phase_lock_091_open_vocab_chat_lm_foundation_check.py
python scripts/probes/run_stable_loop_phase_lock_091_open_vocab_chat_lm_foundation_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_089b_packaged_model_winner_repro_train_proof_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_089_private_evaluation_rc_package_check.py --check-only
git diff --check
```

Optional external corpus follow-up:

```text
python scripts/probes/run_stable_loop_phase_lock_091_open_vocab_chat_lm_foundation.py --out target/pilot_wave/stable_loop_phase_lock_091_open_vocab_chat_lm_foundation/fineweb_slice --upstream-089b-root target/pilot_wave/stable_loop_phase_lock_089b_packaged_model_winner_repro_train_proof/smoke --corpus-file target/datasets/fineweb_edu_small_slice.txt --seed 2026 --train-tokens 1000000 --eval-tokens 200000 --seq-len 128 --batch-size 32 --steps 3000 --heartbeat-sec 20
```

## Assumptions

091 is an open-vocab foundation sanity gate, not a deploy gate. Byte-level tokenization is used to avoid BPE/OOV/tokenizer leakage problems. The runner trains a new target-only PyTorch checkpoint and never trains or mutates the packaged bounded winner. Passing 091 proves only that a runner-local next-byte LM foundation can learn above controls while bounded retention remains intact; it does not prove GPT-like assistant readiness or architecture winner status.

If 091 passes, next milestone is `092_OPEN_VOCAB_FINEWEB_SLICE_CONFIRM`. If 091 fails, next milestone is `091B_OPEN_VOCAB_LM_FAILURE_ANALYSIS`.

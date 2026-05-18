# STABLE_LOOP_PHASE_LOCK_092_OPEN_VOCAB_FINEWEB_SLICE_CONFIRM Result

## Status

092 implements a runner-local FineWeb-Edu slice confirm for the 091 open-vocab byte-level next-byte LM signal.

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

092 is FineWeb-Edu slice transfer sanity only. It is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not deployment readiness, not safety alignment, and not public release. It does not prove INSTNCT/AnchorRoute as an open-domain LM winner.

## Expected Positive Evidence

Smoke output under:

```text
target/pilot_wave/stable_loop_phase_lock_092_open_vocab_fineweb_slice_confirm/smoke
```

must include:

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

The FineWeb source must remain immutable:

```text
fineweb_source_path
fineweb_source_size_bytes
fineweb_source_mtime_before
fineweb_source_mtime_after
fineweb_source_sha256_before
fineweb_source_sha256_after
fineweb_source_hash_unchanged = true
```

The LM split must be FineWeb-only:

```text
fineweb_train_token_count
fineweb_eval_token_count
bounded_retention_rows_in_lm_train = 0
bounded_retention_rows_in_lm_eval = 0
```

The deterministic split proof must include:

```text
corpus_sha256
train_split_sha256
eval_split_sha256
eval_row_hash
eval_token_hash
eval_token_count
train_eval_exact_text_overlap_count = 0
max_train_eval_jaccard < 0.90
```

All control arms use the same eval split:

```text
OPEN_VOCAB_FINEWEB_BYTE_LM_MAIN
CHAR_BIGRAM_BASELINE
RANDOM_BYTE_CONTROL
SHUFFLED_TARGET_CONTROL
STATIC_OUTPUT_CONTROL
COPY_PROMPT_CONTROL
```

Positive verdicts:

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

Failure verdicts include:

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

## Interpretation

Passing 092 means the runner-local PyTorch byte-level next-byte LM learned on a real local FineWeb-Edu slice, beat the char-bigram, random-byte, shuffled-target, static-output, and copy-prompt controls on identical eval tokens, avoided trivial generation collapse, and kept bounded retention references intact.

Passing 092 does not prove GPT-like assistant readiness, open-domain assistant readiness, production chat, deployment readiness, safety alignment, public release, or that INSTNCT/AnchorRoute is an open-domain LM winner.

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

## Next

If 092 passes, next milestone is `093_OPEN_VOCAB_CHAT_SFT_MIX_POC`.

If 092 fails, next milestone is `092B_FINEWEB_SLICE_FAILURE_ANALYSIS`.

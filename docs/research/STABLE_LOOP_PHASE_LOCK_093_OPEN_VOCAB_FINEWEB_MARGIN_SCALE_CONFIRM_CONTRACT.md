# STABLE_LOOP_PHASE_LOCK_093_OPEN_VOCAB_FINEWEB_MARGIN_SCALE_CONFIRM Contract

## Summary

093 is the “is the LM foundation strong enough before Chat SFT?” gate.

092 passed on FineWeb-Edu, but its char-bigram margin was weak:

```text
delta_vs_char_bigram_loss = 0.00309
```

093 reruns the runner-local PyTorch byte-LM on FineWeb across seeds `2026,2027,2028` and requires a meaningful char-bigram margin before proceeding to Chat SFT.

This is FineWeb margin/scale confirmation only. It is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public release, not deployment, not safety alignment, and not proof INSTNCT/AnchorRoute is open-domain LM winner.

## Key Changes

Add only:

```text
scripts/probes/run_stable_loop_phase_lock_093_open_vocab_fineweb_margin_scale_confirm.py
scripts/probes/run_stable_loop_phase_lock_093_open_vocab_fineweb_margin_scale_confirm_check.py
docs/research/STABLE_LOOP_PHASE_LOCK_093_OPEN_VOCAB_FINEWEB_MARGIN_SCALE_CONFIRM_CONTRACT.md
docs/research/STABLE_LOOP_PHASE_LOCK_093_OPEN_VOCAB_FINEWEB_MARGIN_SCALE_CONFIRM_RESULT.md
```

Generated outputs only under:

```text
target/pilot_wave/stable_loop_phase_lock_093_open_vocab_fineweb_margin_scale_confirm/
```

Do not modify:

```text
instnct-core/
service API
deploy harness
SDK/public exports
docs/product/
docs/releases/
root LICENSE
existing checkpoints
083/089 packages
FineWeb source file
```

## Required Upstream And Data

Require positive 092 root:

```text
target/pilot_wave/stable_loop_phase_lock_092_open_vocab_fineweb_slice_confirm/smoke
```

Require:

```text
OPEN_VOCAB_FINEWEB_SLICE_CONFIRM_POSITIVE
fineweb_source_hash_unchanged = true
packaged_winner_hash_unchanged = true
bounded retention pass
leakage audit pass
no GPT-like / production / public release / deployment claim
```

Default FineWeb source:

```text
S:\AI\MESSY TRAINING DATA - INPUT ONLY\Fineweb edu 10B\fineweb_edu_30m.txt
```

If missing:

```text
FINEWEB_SLICE_MISSING
```

Open read-only and record:

```text
fineweb_source_sha256_before
fineweb_source_sha256_after
fineweb_source_size_bytes
fineweb_source_mtime_before
fineweb_source_mtime_after
fineweb_source_hash_unchanged
```

Failure:

```text
FINEWEB_SOURCE_MUTATION_DETECTED
```

FineWeb LM train/eval must be FineWeb-only:

```text
bounded_retention_rows_in_lm_train = 0
bounded_retention_rows_in_lm_eval = 0
```

Failure:

```text
DATASET_MIX_CONTAMINATION_DETECTED
```

## Runner Behavior

Default smoke:

```powershell
python scripts/probes/run_stable_loop_phase_lock_093_open_vocab_fineweb_margin_scale_confirm.py --out target/pilot_wave/stable_loop_phase_lock_093_open_vocab_fineweb_margin_scale_confirm/smoke --upstream-092-root target/pilot_wave/stable_loop_phase_lock_092_open_vocab_fineweb_slice_confirm/smoke --fineweb-source "S:\AI\MESSY TRAINING DATA - INPUT ONLY\Fineweb edu 10B\fineweb_edu_30m.txt" --seeds 2026,2027,2028 --train-tokens 1000000 --eval-tokens 200000 --seq-len 128 --batch-size 32 --steps 3000 --heartbeat-sec 20
```

Use the same runner-local byte-level LM setup:

```text
byte ids 0..255 plus BOS/EOS/PAD
vocab_size >= 259
decoder_path = causal_next_byte
response_table_used_for_main_prediction = false
prediction_oracle_used = false
llm_judge_used = false
runner-local PyTorch byte-LM
```

Record environment:

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

## Arms And Gates

For every seed, train/evaluate exact same eval split across:

```text
OPEN_VOCAB_FINEWEB_BYTE_LM_MAIN
CHAR_UNIGRAM_BASELINE
CHAR_BIGRAM_BASELINE
CHAR_TRIGRAM_BASELINE
RANDOM_BYTE_CONTROL
SHUFFLED_TARGET_CONTROL
STATIC_OUTPUT_CONTROL
COPY_PROMPT_CONTROL
```

Record for every arm:

```text
eval_token_hash
eval_token_count
eval_row_hash
```

Failure:

```text
BASELINE_EVAL_MISMATCH
```

All seeds must pass base gates independently:

```text
train_step_count > 0
checkpoint_after_hash != checkpoint_before_hash
train_loss_final < train_loss_initial
train_eval_exact_text_overlap_count = 0
max_train_eval_jaccard < 0.90
main eval_loss < shuffled_target eval_loss by >= 0.25
main next_byte_accuracy > random_byte_control by >= 0.10
main eval_loss < char_bigram eval_loss
nonempty_generation_rate >= 0.98
utf8_valid_generation_rate >= 0.80
empty_output_rate <= 0.02
static_output_rate <= 0.15
repetition_rate <= 0.25
copy_prompt_rate <= 0.20
bounded_chat_slot_binding_accuracy >= 0.80
finite_label_anchorroute_retention_accuracy >= 0.90
unsupported_refusal_accuracy >= 0.80
packaged_winner_hash_unchanged = true
fineweb_source_hash_unchanged = true
```

Reject mean-only, best-seed, and 2/3 base-gate pass:

```text
MULTI_SEED_OPEN_VOCAB_INSTABILITY_DETECTED
```

Margin gate:

```text
delta_vs_char_bigram_loss >= 0.03 on at least 2/3 seeds
min_delta_vs_char_bigram_loss > 0.00
mean_delta_vs_char_bigram_loss >= 0.03
```

If this fails:

```text
OPEN_VOCAB_MARGIN_TOO_WEAK
```

Do not proceed to Chat SFT if margin fails.

Track char-trigram honestly:

```text
delta_vs_char_trigram_loss
CHAR_TRIGRAM_BASELINE_BEATEN
```

Trigram win is optional in smoke and must be reported plainly.

## Required Artifacts

Write:

```text
queue.json
progress.jsonl
training_config.json
upstream_092_manifest.json
fineweb_source_manifest.json
dataset_manifest.json
tokenizer_manifest.json
seed_run_manifest.json
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
multi_seed_aggregate.json
failure_case_samples.jsonl
summary.json
report.md
```

`progress.jsonl`, `summary.json`, and `report.md` must be written from start and refreshed after upstream verification, FineWeb source verification, dataset split, every seed start, training heartbeat, every seed eval, retention eval, control comparison, aggregate verdict, and final verdict.

Human samples must include each seed and:

```text
short English continuation
unseen word continuation
mixed punctuation continuation
number/text continuation
simple dialogue continuation
unsupported-domain refusal continuation
```

Each row:

```text
seed
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

## Verdicts

Positive:

```text
OPEN_VOCAB_FINEWEB_MARGIN_SCALE_CONFIRM_POSITIVE
UPSTREAM_092_FINEWEB_CONFIRM_VERIFIED
FINEWEB_SOURCE_IMMUTABILITY_PASSES
BYTE_LEVEL_TOKENIZER_REUSED
OPEN_VOCAB_NEXT_BYTE_TRAINING_MULTI_SEED_COMPLETED
TOKEN_OBJECTIVE_LEARNED
MAIN_BEATS_RANDOM_AND_SHUFFLED_CONTROLS
CHAR_BIGRAM_MARGIN_PASSES
COLLAPSE_REJECTED_ALL_SEEDS
BOUNDED_CHAT_RETENTION_PASSES_ALL_SEEDS
FINITE_LABEL_ANCHORROUTE_RETENTION_PASSES_ALL_SEEDS
PACKAGED_WINNER_UNCHANGED
PRODUCTION_CHAT_NOT_CLAIMED
GPT_LIKE_READINESS_NOT_CLAIMED
```

Optional positive:

```text
CHAR_TRIGRAM_BASELINE_BEATEN
```

Failure:

```text
OPEN_VOCAB_FINEWEB_MARGIN_SCALE_CONFIRM_FAILS
UPSTREAM_092_ARTIFACT_MISSING
UPSTREAM_092_NOT_POSITIVE
FINEWEB_SLICE_MISSING
FINEWEB_SOURCE_MUTATION_DETECTED
DATASET_BUILD_FAILS
DATASET_MIX_CONTAMINATION_DETECTED
TRAIN_EVAL_LEAKAGE_DETECTED
BASELINE_EVAL_MISMATCH
NO_ACTUAL_TRAINING_UPDATE_DETECTED
TOKEN_OBJECTIVE_NOT_LEARNED
CONTROL_DELTA_INSUFFICIENT
OPEN_VOCAB_MARGIN_TOO_WEAK
MULTI_SEED_OPEN_VOCAB_INSTABILITY_DETECTED
OPEN_VOCAB_GENERATION_SMOKE_FAILS
BOUNDED_CHAT_RETENTION_REGRESSION_DETECTED
FINITE_LABEL_RETENTION_REGRESSION_DETECTED
PACKAGED_CHECKPOINT_MUTATION_DETECTED
TRAINING_SIDE_EFFECT_ON_PACKAGED_CHECKPOINT
ORACLE_SHORTCUT_DETECTED
LLM_JUDGE_USED
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
GPT_LIKE_READINESS_FALSE_CLAIM
PRODUCTION_CHAT_CLAIM_DETECTED
PUBLIC_RELEASE_CLAIM_DETECTED
ARCHITECTURE_WINNER_FALSE_CLAIM
```

## Validation

```powershell
python -m py_compile scripts/probes/run_stable_loop_phase_lock_093_open_vocab_fineweb_margin_scale_confirm.py
python scripts/probes/run_stable_loop_phase_lock_093_open_vocab_fineweb_margin_scale_confirm.py --out target/pilot_wave/stable_loop_phase_lock_093_open_vocab_fineweb_margin_scale_confirm/smoke --upstream-092-root target/pilot_wave/stable_loop_phase_lock_092_open_vocab_fineweb_slice_confirm/smoke --fineweb-source "S:\AI\MESSY TRAINING DATA - INPUT ONLY\Fineweb edu 10B\fineweb_edu_30m.txt" --seeds 2026,2027,2028 --train-tokens 1000000 --eval-tokens 200000 --seq-len 128 --batch-size 32 --steps 3000 --heartbeat-sec 20
python -m py_compile scripts/probes/run_stable_loop_phase_lock_093_open_vocab_fineweb_margin_scale_confirm_check.py
python scripts/probes/run_stable_loop_phase_lock_093_open_vocab_fineweb_margin_scale_confirm_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_092_open_vocab_fineweb_slice_confirm_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_091_open_vocab_chat_lm_foundation_check.py --check-only
git diff --check
```

## Assumptions

093 reuses the runner-local PyTorch byte-LM setup from 092 and does not change runtime/service/deploy architecture.

If 093 passes, next milestone is `094_OPEN_VOCAB_CHAT_SFT_MIX_POC`.

If 093 fails only on char-bigram margin, next milestone is `093B_OPEN_VOCAB_MARGIN_FAILURE_ANALYSIS`.

If 093 fails on collapse or retention, open-vocab scaling stops until diagnosis.

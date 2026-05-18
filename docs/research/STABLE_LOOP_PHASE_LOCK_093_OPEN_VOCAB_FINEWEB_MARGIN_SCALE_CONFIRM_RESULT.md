# STABLE_LOOP_PHASE_LOCK_093_OPEN_VOCAB_FINEWEB_MARGIN_SCALE_CONFIRM Result

## Status

093 implements a multi-seed FineWeb margin/scale confirmation gate after 092.

092 was positive but weak over char-bigram:

```text
delta_vs_char_bigram_loss = 0.00309
```

093 requires a stronger margin before Chat SFT:

```text
delta_vs_char_bigram_loss >= 0.03 on at least 2/3 seeds
min_delta_vs_char_bigram_loss > 0.00
mean_delta_vs_char_bigram_loss >= 0.03
```

This is FineWeb margin/scale confirmation only for a runner-local PyTorch byte-LM. It is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public release, not deployment, not safety alignment, and not proof INSTNCT/AnchorRoute is open-domain LM winner.

## Expected Evidence

Smoke output:

```text
target/pilot_wave/stable_loop_phase_lock_093_open_vocab_fineweb_margin_scale_confirm/smoke
```

Required top-level artifacts:

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

Every seed has a fresh target-only subdirectory:

```text
seed_2026/
seed_2027/
seed_2028/
```

Per-seed arms:

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

All arms must share:

```text
eval_token_hash
eval_token_count
eval_row_hash
```

## Positive Verdicts

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

## Failure Verdicts

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

## Interpretation

Passing 093 means the FineWeb byte-LM signal beats char-bigram with a meaningful multi-seed margin while avoiding collapse and preserving bounded retention references.

Passing 093 still does not mean good chat inference. It only says there is enough FineWeb LM margin evidence to justify the next Chat SFT PoC.

If 093 fails only on char-bigram margin, the next step is `093B_OPEN_VOCAB_MARGIN_FAILURE_ANALYSIS`, not Chat SFT.

If 093 fails on collapse or retention, open-vocab scaling stops until diagnosis.

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

## Next

If 093 passes, next milestone is `094_OPEN_VOCAB_CHAT_SFT_MIX_POC`.

If 093 fails only on char-bigram margin, next milestone is `093B_OPEN_VOCAB_MARGIN_FAILURE_ANALYSIS`.

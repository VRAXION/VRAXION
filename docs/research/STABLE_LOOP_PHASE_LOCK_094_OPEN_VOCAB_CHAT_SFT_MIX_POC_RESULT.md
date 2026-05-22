# STABLE_LOOP_PHASE_LOCK_094_OPEN_VOCAB_CHAT_SFT_MIX_POC Result

## Status

094 implements a Chat SFT mix PoC after the positive 093 FineWeb margin/scale gate.

The expected smoke root is:

```text
target/pilot_wave/stable_loop_phase_lock_094_open_vocab_chat_sft_mix_poc/smoke
```

This is Chat SFT mix PoC only for a runner-local PyTorch byte-LM. It is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public release, not deployment, not safety alignment, and not proof INSTNCT/AnchorRoute is an open-domain LM winner.

## Expected Evidence

Required artifacts:

```text
queue.json
progress.jsonl
training_config.json
upstream_093_manifest.json
source_checkpoint_manifest.json
sft_dataset_manifest.json
fineweb_replay_manifest.json
sft_train_examples_sample.jsonl
sft_eval_examples_sample.jsonl
training_metrics.jsonl
checkpoint_manifest.json
checkpoint_hashes.json
pre_sft_eval_metrics.json
post_sft_eval_metrics.json
chat_sft_metrics.json
fineweb_retention_metrics.json
bounded_retention_metrics.json
collapse_metrics.json
control_comparison.json
generation_samples.jsonl
human_readable_samples.jsonl
failure_case_samples.jsonl
summary.json
report.md
```

The human sample report must include paired `PRE_SFT_093_BEST_CHECKPOINT` and `POST_SFT_MIX_CHECKPOINT` rows for:

```text
short instruction
simple dialogue
bounded active slot
context carry
unsupported open-domain refusal
boundary/injection refusal
```

## Positive Verdicts

```text
OPEN_VOCAB_CHAT_SFT_MIX_POC_POSITIVE
UPSTREAM_093_FINEWEB_MARGIN_VERIFIED
BEST_093_CHECKPOINT_LOADED_READ_ONLY
CHAT_SFT_DATASET_BUILT
CHAT_SFT_TRAINING_COMPLETED
SFT_OBJECTIVE_LEARNED
CHAT_FORMAT_SIGNAL_IMPROVES
FINEWEB_RETENTION_WITHIN_LIMITS
BOUNDED_CHAT_RETENTION_PASSES
FINITE_LABEL_ANCHORROUTE_RETENTION_PASSES
COLLAPSE_REJECTED
SOURCE_093_CHECKPOINT_UNCHANGED
PRODUCTION_CHAT_NOT_CLAIMED
GPT_LIKE_READINESS_NOT_CLAIMED
```

Optional warning:

```text
WARMSTART_ADVANTAGE_NOT_PROVEN
```

This warning means the post-SFT model passed, but the random-init SFT control was too close to prove that the 093 warm-start is necessary.

## Failure Verdicts

```text
OPEN_VOCAB_CHAT_SFT_MIX_POC_FAILS
UPSTREAM_093_ARTIFACT_MISSING
UPSTREAM_093_NOT_POSITIVE
BEST_SEED_SELECTION_UNDOCUMENTED
SOURCE_093_CHECKPOINT_MUTATION_DETECTED
NO_ACTUAL_SFT_UPDATE_DETECTED
SFT_OBJECTIVE_NOT_LEARNED
TRAIN_EVAL_LEAKAGE_DETECTED
DATASET_MIX_CONTAMINATION_DETECTED
BASELINE_EVAL_MISMATCH
CHAT_FORMAT_SIGNAL_NOT_IMPROVED
COPY_OR_STATIC_CONTROL_UNEXPECTED_PASS
CHAT_SFT_TEMPLATE_COPY_REGRESSION_DETECTED
FINEWEB_RETENTION_REGRESSION_DETECTED
BOUNDED_CHAT_RETENTION_REGRESSION_DETECTED
FINITE_LABEL_RETENTION_REGRESSION_DETECTED
UNSUPPORTED_REFUSAL_REGRESSION_DETECTED
EMPTY_OUTPUT_COLLAPSE_DETECTED
STATIC_RESPONSE_COLLAPSE_DETECTED
REPETITION_COLLAPSE_DETECTED
HUMAN_SAMPLE_REPORT_MISSING
LLM_JUDGE_USED
CHAT_EVAL_RUBRIC_MISSING
FINEWEB_SOURCE_MUTATION_DETECTED
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
GPT_LIKE_READINESS_FALSE_CLAIM
PRODUCTION_CHAT_CLAIM_DETECTED
PUBLIC_RELEASE_CLAIM_DETECTED
ARCHITECTURE_WINNER_FALSE_CLAIM
```

## Interpretation

Passing 094 means a target-local SFT copy of the best 093 FineWeb byte-LM checkpoint improved on deterministic chat/instruction SFT rows while keeping FineWeb retention, bounded behavior, finite-label retention, and collapse metrics inside thresholds.

Passing 094 still does not mean GPT-like assistant readiness, open-domain assistant readiness, production chat, deployment readiness, public release, or safety alignment.

## Validation

```powershell
python -m py_compile scripts/probes/run_stable_loop_phase_lock_094_open_vocab_chat_sft_mix_poc.py
python scripts/probes/run_stable_loop_phase_lock_094_open_vocab_chat_sft_mix_poc.py --out target/pilot_wave/stable_loop_phase_lock_094_open_vocab_chat_sft_mix_poc/smoke --upstream-093-root target/pilot_wave/stable_loop_phase_lock_093_open_vocab_fineweb_margin_scale_confirm/smoke --fineweb-source "S:\AI\MESSY TRAINING DATA - INPUT ONLY\Fineweb edu 10B\fineweb_edu_30m.txt" --seed 2028 --sft-examples 12000 --lm-replay-tokens 200000 --seq-len 128 --batch-size 32 --sft-steps 1200 --control-steps 400 --heartbeat-sec 20
python -m py_compile scripts/probes/run_stable_loop_phase_lock_094_open_vocab_chat_sft_mix_poc_check.py
python scripts/probes/run_stable_loop_phase_lock_094_open_vocab_chat_sft_mix_poc_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_093_open_vocab_fineweb_margin_scale_confirm_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_092_open_vocab_fineweb_slice_confirm_check.py --check-only
git diff --check
```

## Next

If 094 passes, next milestone is `095_FRESH_OPEN_DOMAIN_CHAT_EVAL`.

If 094 fails due to retention, use `094B_CHAT_SFT_RETENTION_FAILURE_ANALYSIS`.

If 094 fails due to SFT signal, use `094B_CHAT_SFT_SIGNAL_FAILURE_ANALYSIS`.

If 094 fails due to generation collapse/template copy, use `094B_CHAT_SFT_GENERATION_FAILURE_ANALYSIS`.

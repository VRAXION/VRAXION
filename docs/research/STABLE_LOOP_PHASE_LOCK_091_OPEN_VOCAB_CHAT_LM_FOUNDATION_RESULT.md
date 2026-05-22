# STABLE_LOOP_PHASE_LOCK_091_OPEN_VOCAB_CHAT_LM_FOUNDATION Result

## Status

091 implements a runner-local open-vocab next-byte LM foundation sanity gate after the 089B bounded winner proof.

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

## Expected Positive Evidence

Smoke output under:

```text
target/pilot_wave/stable_loop_phase_lock_091_open_vocab_chat_lm_foundation/smoke
```

must include:

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

The positive verdicts are:

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

## Interpretation

Passing 091 means a runner-local PyTorch byte-level next-token foundation sanity gate learned above the char-bigram, random-byte, and shuffled-target controls on a deterministic local text mix, generated non-empty UTF-8 continuations without trivial collapse, and kept the packaged bounded winner checkpoint untouched as the bounded retention reference.

Passing 091 does not mean the packaged bounded winner itself is now GPT-like. It does not prove INSTNCT as open-domain LM, and it is not production chat, not public release, not safety alignment, and not deployment.

If 091 passes, the next milestone is `092_OPEN_VOCAB_FINEWEB_SLICE_CONFIRM`. If 091 fails, the next milestone is `091B_OPEN_VOCAB_LM_FAILURE_ANALYSIS`.

# STABLE_LOOP_PHASE_LOCK_094_OPEN_VOCAB_CHAT_SFT_MIX_POC Contract

## Summary

094 is the first Chat SFT mix PoC after the positive 093 FineWeb margin/scale gate.

It loads the best 093 runner-local PyTorch byte-LM checkpoint read-only, trains only a target-local 094 SFT copy, and checks whether chat/instruction behavior improves while FineWeb LM retention, bounded chat behavior, and finite-label AnchorRoute retention remain inside hard gates.

094 is Chat SFT mix PoC only. It is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public release, not deployment, not safety alignment, and not proof INSTNCT/AnchorRoute is an open-domain LM winner.

## Key Changes

Add only:

```text
scripts/probes/run_stable_loop_phase_lock_094_open_vocab_chat_sft_mix_poc.py
scripts/probes/run_stable_loop_phase_lock_094_open_vocab_chat_sft_mix_poc_check.py
docs/research/STABLE_LOOP_PHASE_LOCK_094_OPEN_VOCAB_CHAT_SFT_MIX_POC_CONTRACT.md
docs/research/STABLE_LOOP_PHASE_LOCK_094_OPEN_VOCAB_CHAT_SFT_MIX_POC_RESULT.md
```

Generated outputs stay under:

```text
target/pilot_wave/stable_loop_phase_lock_094_open_vocab_chat_sft_mix_poc/
```

Do not modify runtime/service/deploy code, SDK/public exports, product/release docs, root LICENSE, existing checkpoints, 083/089 packages, or the FineWeb source file.

## Required Upstream

Require positive 093:

```text
OPEN_VOCAB_FINEWEB_MARGIN_SCALE_CONFIRM_POSITIVE
all_seed_base_gates_pass = true
char_bigram_margin_pass = true
retention_pass_all_seeds = true
fineweb_source_hash_unchanged_all_seeds = true
packaged_winner_hash_unchanged_all_seeds = true
architecture_winner_for_open_vocab_claimed = false
```

Best-seed selection must be documented:

```text
selected_093_seed = 2028
selected_by = lowest_eval_loss
all_093_seed_eval_losses
all_093_seed_delta_vs_bigram
selection_rule_fixed_before_094_training = true
```

Failure:

```text
BEST_SEED_SELECTION_UNDOCUMENTED
UPSTREAM_093_ARTIFACT_MISSING
UPSTREAM_093_NOT_POSITIVE
```

## Runner Behavior

Default smoke:

```powershell
python scripts/probes/run_stable_loop_phase_lock_094_open_vocab_chat_sft_mix_poc.py --out target/pilot_wave/stable_loop_phase_lock_094_open_vocab_chat_sft_mix_poc/smoke --upstream-093-root target/pilot_wave/stable_loop_phase_lock_093_open_vocab_fineweb_margin_scale_confirm/smoke --fineweb-source "S:\AI\MESSY TRAINING DATA - INPUT ONLY\Fineweb edu 10B\fineweb_edu_30m.txt" --seed 2028 --sft-examples 12000 --lm-replay-tokens 200000 --seq-len 128 --batch-size 32 --sft-steps 1200 --control-steps 400 --heartbeat-sec 20
```

Use the same runner-local PyTorch byte-LM setup:

```text
byte ids 0..255 plus BOS/EOS/PAD
decoder_path = causal_next_byte
response_table_used_for_main_prediction = false
prediction_oracle_used = false
llm_judge_used = false
```

The source 093 checkpoint is read-only:

```text
source_093_checkpoint_hash_before
source_093_checkpoint_hash_after
source_093_checkpoint_unchanged = true
```

The target SFT checkpoint must change:

```text
target_sft_checkpoint_before_hash
target_sft_checkpoint_after_hash
target_sft_checkpoint_changed = true
sft_train_step_count > 0
```

## Gates

SFT leakage and dataset-mix gates:

```text
sft_train_eval_exact_prompt_overlap_count = 0
sft_train_eval_exact_response_overlap_count = 0
max_sft_train_eval_prompt_jaccard < 0.90
fineweb_rows_in_sft_eval = 0
bounded_rows_in_fineweb_eval = 0
```

Compare identical SFT eval rows across:

```text
PRE_SFT_093_BEST_CHECKPOINT
POST_SFT_MIX_CHECKPOINT
SFT_ONLY_FROM_RANDOM_INIT_CONTROL
NO_FINEWEB_REPLAY_CONTROL
STATIC_OUTPUT_CONTROL
COPY_PROMPT_CONTROL
```

Positive requires:

```text
post_sft_sft_eval_loss < pre_sft_sft_eval_loss by >= 0.15
post_sft_prompt_response_accuracy >= pre_sft_prompt_response_accuracy + 0.20
post_sft_prompt_response_accuracy >= 0.70
post_sft beats STATIC_OUTPUT_CONTROL
post_sft beats COPY_PROMPT_CONTROL
exact_sft_train_response_copy_rate <= 0.25
response_skeleton_reuse_rate <= 0.50
novel_response_rate >= 0.50
fineweb_eval_loss_regression <= 0.35
fineweb_next_byte_accuracy_drop <= 0.08
bounded_chat_slot_binding_accuracy >= 0.80
finite_label_anchorroute_retention_accuracy >= 0.90
unsupported_refusal_accuracy >= 0.80
nonempty_generation_rate >= 0.98
utf8_valid_generation_rate >= 0.80
empty_output_rate <= 0.02
static_output_rate <= 0.15
repetition_rate <= 0.25
copy_prompt_rate <= 0.20
llm_judge_used = false
```

If random-init control is too close to post-SFT, emit:

```text
WARMSTART_ADVANTAGE_NOT_PROVEN
```

and do not claim the 093 warm-start is necessary.

## Verdicts

Positive:

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

Failures include:

```text
OPEN_VOCAB_CHAT_SFT_MIX_POC_FAILS
BEST_SEED_SELECTION_UNDOCUMENTED
SOURCE_093_CHECKPOINT_MUTATION_DETECTED
NO_ACTUAL_SFT_UPDATE_DETECTED
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

## Next

If 094 passes, proceed to `095_FRESH_OPEN_DOMAIN_CHAT_EVAL`.

If retention fails, run `094B_CHAT_SFT_RETENTION_FAILURE_ANALYSIS`.

If SFT signal fails, run `094B_CHAT_SFT_SIGNAL_FAILURE_ANALYSIS`.

If generation collapses or template-copy returns, run `094B_CHAT_SFT_GENERATION_FAILURE_ANALYSIS`.

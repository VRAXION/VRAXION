# STABLE_LOOP_PHASE_LOCK_100_OPEN_VOCAB_ASSISTANT_CAPABILITY_SCALE Contract

100 starts the GPT-like capability track after the bounded local/private release-ready stop condition, but it is not a GPT-like readiness claim.

The 099 bounded release stack is frozen. 100 trains only a new target-only runner-local PyTorch byte-LM/SFT checkpoint under `target/pilot_wave/stable_loop_phase_lock_100_open_vocab_assistant_capability_scale/`.

Allowed claim:

```text
runner-local open-vocab assistant capability scale signal improved under bounded retention gates
```

Forbidden claims:

```text
GPT-like assistant readiness
open-domain assistant readiness
production chat
public API
hosted SaaS
deployment readiness
safety alignment
INSTNCT/AnchorRoute proven as open-domain LM winner
```

## Required Upstreams

```text
099 BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_POSITIVE
098 PRIVATE_EVAL_RC_REFRESH_WITH_GENERATION_REPAIR_POSITIVE
097 CHAT_DECODER_MULTI_SEED_OOD_RETENTION_CONFIRM_POSITIVE
094B CHAT_SFT_FREE_GENERATION_GAP_ANALYSIS_POSITIVE
094 OPEN_VOCAB_CHAT_SFT_MIX_POC_POSITIVE
093 OPEN_VOCAB_FINEWEB_MARGIN_SCALE_CONFIRM_POSITIVE
```

## Frozen Surfaces

Do not modify:

```text
099 target artifacts
083/089/098 packages
packaged bounded winner checkpoint
service API
deployment harness
SDK/public exports
root LICENSE
FineWeb source file
```

Required:

```text
bounded_release_artifact_unchanged = true
packaged_winner_hash_unchanged = true
no_training_on_bounded_release = true
```

## Training And Eval

Train a new target-only checkpoint from the 094 target SFT checkpoint copy.

Default smoke:

```text
seed = 2026
train_tokens = 3000000
eval_tokens = 300000
sft_examples = 24000
lm_replay_tokens = 300000
seq_len = 128
batch_size = 32
steps = 6000
heartbeat_sec = 20
```

Eval families:

```text
FRESH_SHORT_INSTRUCTION
FRESH_SHORT_EXPLANATION
FRESH_OPEN_DOMAIN_SIMPLE_QA
FRESH_UNSUPPORTED_REFUSAL
FRESH_MULTI_TURN_CONTEXT_CARRY
FRESH_HUNGARIAN_BASIC_CHAT
FRESH_ENGLISH_BASIC_CHAT
FRESH_BOUNDARY_REFUSAL
FRESH_ANTI_REPETITION
BOUNDED_CHAT_RETENTION
FINITE_LABEL_ANCHORROUTE_RETENTION
```

Every long run must refresh:

```text
progress.jsonl
summary.json
report.md
training_metrics.jsonl
```

## Positive Gates

Positive requires:

```text
target_100_checkpoint_changed = true
train_step_count > 0
train_loss_final < train_loss_initial
eval_loss_after < eval_loss_before
generated_prompt_response_accuracy > 094 generated_prompt_response_accuracy
generated_prompt_response_accuracy >= 0.25
instruction_following_accuracy >= 0.50
short_explanation_accuracy >= 0.50
multi_turn_context_accuracy >= 0.40
unsupported_refusal_accuracy >= 0.80
bounded_chat_slot_binding_accuracy >= 0.80
finite_label_anchorroute_retention_accuracy >= 0.90
nonempty_generation_rate >= 0.98
utf8_valid_generation_rate >= 0.80
empty_output_rate <= 0.02
static_output_rate <= 0.15
repetition_rate <= 0.25
copy_prompt_rate <= 0.20
llm_judge_used = false
prediction_oracle_used = false
response_table_used_for_main_prediction = false
```

## Verdicts

Positive:

```text
OPEN_VOCAB_ASSISTANT_CAPABILITY_SCALE_POSITIVE
BOUNDED_RELEASE_BASELINE_FROZEN
OPEN_VOCAB_TRAINING_COMPLETED
ASSISTANT_GENERATION_IMPROVES
MULTI_TURN_SMOKE_RECORDED
HUNGARIAN_ENGLISH_SMOKE_RECORDED
RETENTION_PASSES
COLLAPSE_REJECTED
PRODUCTION_CHAT_NOT_CLAIMED
GPT_LIKE_READINESS_NOT_CLAIMED
```

Failure:

```text
OPEN_VOCAB_ASSISTANT_CAPABILITY_SCALE_FAILS
BOUNDED_RELEASE_MUTATION_DETECTED
PACKAGED_CHECKPOINT_MUTATION_DETECTED
TOKEN_OBJECTIVE_NOT_LEARNED
ASSISTANT_GENERATION_NOT_IMPROVED
MULTI_TURN_CONTEXT_FAILS
HUNGARIAN_BASIC_FAILS
RETENTION_REGRESSION_DETECTED
STATIC_RESPONSE_COLLAPSE_DETECTED
REPETITION_COLLAPSE_DETECTED
GPT_LIKE_READINESS_FALSE_CLAIM
PRODUCTION_CHAT_CLAIM_DETECTED
```

If positive, continue to `101_FRESH_ASSISTANT_EVAL_AND_FAILURE_MAP`. If failed, continue to `100B_ASSISTANT_SCALE_FAILURE_ANALYSIS`.

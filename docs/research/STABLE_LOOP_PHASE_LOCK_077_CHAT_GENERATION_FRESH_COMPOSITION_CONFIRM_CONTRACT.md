# STABLE_LOOP_PHASE_LOCK_077_CHAT_GENERATION_FRESH_COMPOSITION_CONFIRM Contract

Status: contract for eval-only fresh composition confirmation of the 076
runner-local chat checkpoint.

077 tests whether the 076 chat PoC composes useful multi-token responses on
fresh prompts, or mostly selects/copies controlled SFT response templates.

This is bounded fresh composition confirm only.

no training
no resume
no checkpoint repair
no checkpoint mutation
no replacement checkpoint
not GPT-like assistant readiness
not full English LM
not language grounding
not production chat
not public beta / GA / hosted SaaS
no service API change
no deployment harness change
no SDK/public export change
no release docs change
no root LICENSE change

## Implementation Scope

077 adds only:

```text
instnct-core/examples/phase_lane_chat_generation_fresh_composition_confirm.rs
scripts/probes/run_stable_loop_phase_lock_077_chat_generation_fresh_composition_confirm_check.py
docs/research/STABLE_LOOP_PHASE_LOCK_077_CHAT_GENERATION_FRESH_COMPOSITION_CONFIRM_CONTRACT.md
docs/research/STABLE_LOOP_PHASE_LOCK_077_CHAT_GENERATION_FRESH_COMPOSITION_CONFIRM_RESULT.md
```

Generated artifacts are written only under:

```text
target/pilot_wave/stable_loop_phase_lock_077_chat_generation_fresh_composition_confirm/
```

## Upstream Inputs

Default checkpoint:

```text
target/pilot_wave/stable_loop_phase_lock_076_chat_generation_poc/smoke/checkpoints/chat_generation_poc/model_checkpoint.json
```

Upstream roots:

```text
target/pilot_wave/stable_loop_phase_lock_076_chat_generation_poc/smoke
target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke
```

Required upstream verification:

```text
upstream_076_summary_present = true
upstream_076_positive = true
checkpoint_exists = true
child_eval_started_after_077_start = true
checkpoint_hash_before
checkpoint_hash_after
checkpoint_hash_unchanged = true
train_step_count = 0
prediction_oracle_used = false
response_uses_decoder_loop = true
```

Failure verdicts:

```text
UPSTREAM_076_ARTIFACT_MISSING
STALE_CHECKPOINT_ARTIFACT_USED
CHECKPOINT_MUTATION_DETECTED
TRAINING_SIDE_EFFECT_DETECTED
ORACLE_SHORTCUT_DETECTED
```

## Fresh Eval Families

```text
FRESH_SIMPLE_INSTRUCTION
FRESH_SHORT_EXPLANATION
FRESH_CONTEXT_CARRY_CHAT
FRESH_TWO_TURN_DIALOGUE
FRESH_BOUNDARY_REFUSAL_MINI
FRESH_COMPOSITION_NOVELTY
ANTI_TEMPLATE_COPY
ANTI_REPETITION
FINITE_LABEL_ANCHORROUTE_RETENTION
```

Fresh prompts must use:

```text
new wording
new entities
new instruction shapes
new context-carry variants
new refusal/boundary phrasing
no exact prompt overlap with 076 train/eval prompts
```

## Required Artifacts

```text
queue.json
progress.jsonl
benchmark_config.json
upstream_076_manifest.json
upstream_074_manifest.json
checkpoint_manifest.json
fresh_chat_eval_dataset.jsonl
generation_samples.jsonl
human_readable_samples.jsonl
composition_metrics.json
novelty_metrics.json
collapse_metrics.json
finite_label_retention_metrics.json
summary.json
report.md
```

`progress.jsonl`, `summary.json`, and `report.md` are written from start and
refreshed during major phases.

## Template-Copy And Novelty Metrics

Compare generated output against:

```text
076 train responses
076 eval outputs
checkpoint response_table responses
exact template responses
```

Record:

```text
exact_train_response_copy_rate
exact_eval_response_copy_rate
response_table_copy_rate
template_copy_rate
train_response_ngram_overlap
novel_response_rate
prompt_ngram_overlap_stats
train_eval_exact_prompt_overlap_count = 0
```

Human-readable samples must include:

```text
eval_family
prompt
expected_behavior
required_keywords
forbidden_outputs
model_output
output_classification
pass_fail
short_diagnosis
template_copy_flag
novelty_flag
```

## Composition, Collapse, And Retention Metrics

Record:

```text
multi_token_response_rate
non_empty_response_rate
fresh_instruction_accuracy
fresh_context_carry_accuracy
two_turn_dialogue_accuracy
boundary_refusal_accuracy
label_only_response_rate
generated_token_count_mean
generated_token_count_min
unique_response_count
empty_output_rate
space_output_rate
top_response_rate
static_response_rate
repetition_rate
copy_prompt_rate
finite_label_retention_accuracy
```

Finite-label retention must include real 074-style rows:

```text
active scenario binding
distractor scenario rejection
old/stale/inactive suppression
answer-only scenario binding
```

## Gates

Positive gate:

```text
multi_token_response_rate >= 0.90
non_empty_response_rate >= 0.98
fresh_instruction_accuracy >= 0.70
fresh_context_carry_accuracy >= 0.65
novel_response_rate >= 0.60
template_copy_rate <= 0.30
label_only_response_rate <= 0.20
generated_token_count_min >= 2
empty_output_rate <= 0.02
space_output_rate <= 0.02
static_response_rate <= 0.15
repetition_rate <= 0.20
finite_label_retention_accuracy >= 0.90
checkpoint_hash_unchanged = true
train_step_count = 0
prediction_oracle_used = false
```

Positive verdicts:

```text
CHAT_GENERATION_FRESH_COMPOSITION_CONFIRM_POSITIVE
FRESH_MULTI_TOKEN_RESPONSES_PASS
FRESH_INSTRUCTION_FOLLOWING_PASSES
FRESH_CONTEXT_CARRY_CHAT_PASSES
TEMPLATE_COPY_REJECTED
STATIC_RESPONSE_COLLAPSE_REJECTED
FINITE_LABEL_RETENTION_PASSES
NO_TRAINING_PERFORMED
PRODUCTION_CHAT_NOT_CLAIMED
```

Failure verdicts:

```text
CHAT_GENERATION_FRESH_COMPOSITION_CONFIRM_FAILS
FRESH_INSTRUCTION_FOLLOWING_FAILS
FRESH_CONTEXT_CARRY_CHAT_FAILS
TEMPLATE_COPY_DETECTED
LABEL_ONLY_RESPONSE_COLLAPSE_DETECTED
CHAT_GENERATION_SURFACE_STILL_TOO_TABLE_LIKE
STATIC_RESPONSE_COLLAPSE_DETECTED
REPETITION_COLLAPSE_DETECTED
EMPTY_OUTPUT_COLLAPSE_DETECTED
FINITE_LABEL_RETENTION_REGRESSION_DETECTED
TRAIN_EVAL_LEAKAGE_DETECTED
HUMAN_SAMPLE_REPORT_MISSING
OPEN_ENDED_ASSISTANT_CLAIM_DETECTED
GPT_LIKE_READINESS_FALSE_CLAIM
PRODUCTION_CHAT_CLAIM_DETECTED
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
```

If any core gate fails, do not emit
`CHAT_GENERATION_FRESH_COMPOSITION_CONFIRM_POSITIVE`.

## Validation Commands

```powershell
cargo check -p instnct-core --example phase_lane_chat_generation_fresh_composition_confirm
cargo run -p instnct-core --example phase_lane_chat_generation_fresh_composition_confirm -- --out target/pilot_wave/stable_loop_phase_lock_077_chat_generation_fresh_composition_confirm/smoke --checkpoint target/pilot_wave/stable_loop_phase_lock_076_chat_generation_poc/smoke/checkpoints/chat_generation_poc/model_checkpoint.json --upstream-076-root target/pilot_wave/stable_loop_phase_lock_076_chat_generation_poc/smoke --upstream-074-root target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke --seed 2027 --heartbeat-sec 20
python -m py_compile scripts/probes/run_stable_loop_phase_lock_077_chat_generation_fresh_composition_confirm_check.py
python scripts/probes/run_stable_loop_phase_lock_077_chat_generation_fresh_composition_confirm_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_076_chat_generation_poc_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm_check.py --check-only
git diff --check
```

If 077 passes, the next milestone is `078_CHAT_GENERATION_MULTI_SEED_CONFIRM`.
If 077 fails, the next milestone is `077B_CHAT_GENERATION_FAILURE_ANALYSIS`.

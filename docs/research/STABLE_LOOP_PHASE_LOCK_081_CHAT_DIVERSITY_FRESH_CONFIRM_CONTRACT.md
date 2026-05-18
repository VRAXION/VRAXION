# STABLE_LOOP_PHASE_LOCK_081_CHAT_DIVERSITY_FRESH_CONFIRM Contract

081 is an eval-only fresh confirmation gate for the 080
`TOKEN_COMPOSITION_DIVERSITY_REPAIR` checkpoint.

It checks whether the 080 chat-diversity repair generalizes to fresh,
non-080-shaped prompts without falling back to template copy, response-table
copy, skeleton reuse, low-vocabulary collapse, or finite-label-only output.

This is bounded fresh chat diversity confirmation only.

not GPT-like assistant readiness
not full English LM
not language grounding
not production chat
not safety alignment
not public beta / GA / hosted SaaS

no service API change
no deployment harness change
no SDK/public export change
no release docs change
no root LICENSE change
no upstream checkpoint mutation

## Files

```text
instnct-core/examples/phase_lane_chat_diversity_fresh_confirm.rs
scripts/probes/run_stable_loop_phase_lock_081_chat_diversity_fresh_confirm_check.py
docs/research/STABLE_LOOP_PHASE_LOCK_081_CHAT_DIVERSITY_FRESH_CONFIRM_CONTRACT.md
docs/research/STABLE_LOOP_PHASE_LOCK_081_CHAT_DIVERSITY_FRESH_CONFIRM_RESULT.md
```

Generated artifacts are written only under:

```text
target/pilot_wave/stable_loop_phase_lock_081_chat_diversity_fresh_confirm/
```

## Upstream And Eval-Only Guard

Required upstream roots:

```text
target/pilot_wave/stable_loop_phase_lock_080_chat_composition_diversity_repair/smoke
target/pilot_wave/stable_loop_phase_lock_079b_chat_composition_fresh_failure_analysis/smoke
target/pilot_wave/stable_loop_phase_lock_079_chat_composition_fresh_confirm/smoke
target/pilot_wave/stable_loop_phase_lock_078_chat_composition_repair/smoke
target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke
```

Default checkpoint:

```text
target/pilot_wave/stable_loop_phase_lock_080_chat_composition_diversity_repair/smoke/checkpoints/chat_composition_diversity_repair/model_checkpoint.json
```

If upstream artifacts are missing, emit:

```text
UPSTREAM_080_ARTIFACT_MISSING
```

Do not rerun 078, 079, 079B, or 080. Do not train or repair a replacement
checkpoint.

Record:

```text
train_step_count = 0
checkpoint_hash_before
checkpoint_hash_after
checkpoint_hash_unchanged = true
prediction_oracle_used = false
llm_judge_used = false
decoder_path = token_level_next_token
response_table_used_for_main_prediction = false
eval_started_after_081_start = true
```

Failures include:

```text
TRAINING_SIDE_EFFECT_DETECTED
CHECKPOINT_MUTATION_DETECTED
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
```

## Fresh Evaluation

Fresh families:

```text
FRESH_DIVERSITY_SIMPLE_INSTRUCTION
FRESH_DIVERSITY_SHORT_EXPLANATION
FRESH_DIVERSITY_CONTEXT_SLOT
FRESH_DIVERSITY_TWO_TURN
FRESH_DIVERSITY_BOUNDARY_MINI
FRESH_DIVERSITY_SEMANTIC_RECOMBINATION
FRESH_ANTI_TEMPLATE_COPY
FRESH_ANTI_SKELETON_REUSE
FRESH_ANTI_REPETITION
FINITE_LABEL_ANCHORROUTE_RETENTION
```

Fresh prompt audit compares against 080 train/eval prompts, 079 prompts, 078
train/eval/generated prompts, and 076 train/eval prompts.

Record:

```text
overlap_with_080_train_prompt_count
overlap_with_080_eval_prompt_count
overlap_with_079_prompt_count
overlap_with_078_prompt_count
overlap_with_076_prompt_count
near_duplicate_prompt_count
```

Exact prompt overlap or token-Jaccard near duplicate >= 0.90 fails:

```text
FRESH_PROMPT_LEAKAGE_DETECTED
```

Template and skeleton audit compares generated outputs against 080
train/eval/generated outputs, 079 generated outputs, 078 train/eval/generated
outputs, and 076 response-table outputs.

Required metrics:

```text
multi_token_response_rate
non_empty_response_rate
fresh_instruction_accuracy
fresh_context_carry_accuracy
slot_binding_accuracy
two_turn_dialogue_accuracy
boundary_refusal_accuracy
novel_response_rate
template_copy_rate
exact_train_response_copy_rate
exact_eval_response_copy_rate
response_table_copy_rate
semantic_template_overlap_rate
slot_only_skeleton_reuse_rate
response_skeleton_reuse_rate
top_skeleton_rate
response_skeleton_diversity
generated_to_train_vocab_ratio
unique_bigram_count
unique_trigram_count
token_entropy
response_entropy
label_only_response_rate
empty_output_rate
space_output_rate
static_response_rate
repetition_rate
copy_prompt_rate
finite_label_retention_accuracy
slot_value_expected
slot_value_emitted
wrong_slot_rate
missing_slot_rate
stale_slot_rate
```

## Gates And Artifacts

Positive requires:

```text
multi_token_response_rate >= 0.90
non_empty_response_rate >= 0.98
fresh_instruction_accuracy >= 0.75
fresh_context_carry_accuracy >= 0.75
slot_binding_accuracy >= 0.75
two_turn_dialogue_accuracy >= 0.70
boundary_refusal_accuracy >= 0.70
novel_response_rate >= 0.65
template_copy_rate <= 0.25
exact_train_response_copy_rate <= 0.15
exact_eval_response_copy_rate <= 0.15
response_table_copy_rate <= 0.20
semantic_template_overlap_rate <= 0.50
slot_only_skeleton_reuse_rate <= 0.25
response_skeleton_reuse_rate <= 0.50
top_skeleton_rate <= 0.35
response_skeleton_diversity >= 0.50
generated_to_train_vocab_ratio >= 0.35
unique_bigram_count >= 30
unique_trigram_count >= 30
token_entropy > 2.0
response_entropy > 2.0
label_only_response_rate <= 0.15
empty_output_rate <= 0.02
space_output_rate <= 0.02
static_response_rate <= 0.15
repetition_rate <= 0.20
copy_prompt_rate <= 0.15
finite_label_retention_accuracy >= 0.90
checkpoint_hash_unchanged = true
train_step_count = 0
prediction_oracle_used = false
llm_judge_used = false
response_table_used_for_main_prediction = false
```

Required artifacts:

```text
queue.json
progress.jsonl
benchmark_config.json
upstream_080_manifest.json
checkpoint_manifest.json
fresh_chat_eval_dataset.jsonl
generation_samples.jsonl
human_readable_samples.jsonl
composition_metrics.json
novelty_metrics.json
skeleton_diversity_metrics.json
vocabulary_entropy_metrics.json
context_slot_metrics.json
finite_label_retention_metrics.json
collapse_metrics.json
summary.json
report.md
```

`progress.jsonl`, `summary.json`, and `report.md` must be written from start
and refreshed by phase.

`human_readable_samples.jsonl` must include:

```text
eval_family
prompt
model_output
expected_behavior
required_keywords
forbidden_outputs
pass_fail
output_classification
novelty_flag
template_copy_flag
skeleton_reuse_flag
semantic_template_overlap_score
slot_binding_diagnosis
short_diagnosis
```

## Verdicts

Positive verdicts:

```text
CHAT_DIVERSITY_FRESH_CONFIRM_POSITIVE
NO_TRAINING_PERFORMED
CHECKPOINT_UNCHANGED
FRESH_DIVERSITY_MULTI_TOKEN_RESPONSES_PASS
FRESH_DIVERSITY_INSTRUCTION_PASSES
FRESH_DIVERSITY_CONTEXT_SLOT_BINDING_PASSES
FRESH_DIVERSITY_TEMPLATE_COPY_REJECTED
FRESH_DIVERSITY_SKELETON_REUSE_REJECTED
FRESH_DIVERSITY_VOCAB_ENTROPY_PASSES
FINITE_LABEL_ANCHORROUTE_RETENTION_PASSES
STATIC_RESPONSE_COLLAPSE_REJECTED
PRODUCTION_CHAT_NOT_CLAIMED
```

Failure verdicts:

```text
CHAT_DIVERSITY_FRESH_CONFIRM_FAILS
UPSTREAM_080_ARTIFACT_MISSING
TRAINING_SIDE_EFFECT_DETECTED
CHECKPOINT_MUTATION_DETECTED
FRESH_PROMPT_LEAKAGE_DETECTED
TEMPLATE_COPY_DETECTED
SKELETON_REUSE_REGRESSION_DETECTED
VOCAB_DIVERSITY_REGRESSION_DETECTED
CONTEXT_SLOT_BINDING_FAILS
FINITE_LABEL_RETENTION_REGRESSION_DETECTED
STATIC_RESPONSE_COLLAPSE_DETECTED
REPETITION_COLLAPSE_DETECTED
EMPTY_OUTPUT_COLLAPSE_DETECTED
HUMAN_SAMPLE_REPORT_MISSING
GPT_LIKE_READINESS_FALSE_CLAIM
PRODUCTION_CHAT_CLAIM_DETECTED
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
```

## Validation

```powershell
cargo check -p instnct-core --example phase_lane_chat_diversity_fresh_confirm
cargo run -p instnct-core --example phase_lane_chat_diversity_fresh_confirm -- --out target/pilot_wave/stable_loop_phase_lock_081_chat_diversity_fresh_confirm/smoke --upstream-080-root target/pilot_wave/stable_loop_phase_lock_080_chat_composition_diversity_repair/smoke --upstream-079b-root target/pilot_wave/stable_loop_phase_lock_079b_chat_composition_fresh_failure_analysis/smoke --upstream-079-root target/pilot_wave/stable_loop_phase_lock_079_chat_composition_fresh_confirm/smoke --upstream-078-root target/pilot_wave/stable_loop_phase_lock_078_chat_composition_repair/smoke --upstream-074-root target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke --seed 2027 --heartbeat-sec 20
python -m py_compile scripts/probes/run_stable_loop_phase_lock_081_chat_diversity_fresh_confirm_check.py
python scripts/probes/run_stable_loop_phase_lock_081_chat_diversity_fresh_confirm_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_080_chat_composition_diversity_repair_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_079b_chat_composition_fresh_failure_analysis_check.py --check-only
git diff --check
```

If 081 passes, next milestone is `082_CHAT_DIVERSITY_MULTI_SEED_CONFIRM`.

If 081 fails, next milestone is `081B_CHAT_DIVERSITY_FRESH_FAILURE_ANALYSIS`.

# STABLE_LOOP_PHASE_LOCK_080_CHAT_COMPOSITION_DIVERSITY_REPAIR Result

Status: implementation result for bounded runner-local chat composition diversity repair.

080 targets the 079B diagnosis:

```text
exact_078_train_response_copy_rate = 1.0
semantic_078_template_overlap_rate = 1.0
response_skeleton_reuse_rate = 1.0
context_slot_binding_accuracy = 1.0
context_composition_novelty_rate = 0.0
finite_label_retention_accuracy = 1.0
```

080 trains a new target-only research checkpoint under:

```text
target/pilot_wave/stable_loop_phase_lock_080_chat_composition_diversity_repair/
```

This is bounded runner-local chat composition diversity repair only.

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

## Implemented Files

```text
instnct-core/examples/phase_lane_chat_composition_diversity_repair.rs
scripts/probes/run_stable_loop_phase_lock_080_chat_composition_diversity_repair_check.py
docs/research/STABLE_LOOP_PHASE_LOCK_080_CHAT_COMPOSITION_DIVERSITY_REPAIR_CONTRACT.md
docs/research/STABLE_LOOP_PHASE_LOCK_080_CHAT_COMPOSITION_DIVERSITY_REPAIR_RESULT.md
```

## Runner Behavior

Main arm:

```text
TOKEN_COMPOSITION_DIVERSITY_REPAIR
```

Main prediction path:

```text
decoder_path = token_level_next_token
response_table_used_for_main_prediction = false
response_table_path_available_but_disabled = true
skeleton_dropout_enabled = true
lexical_dropout_enabled = true
clause_order_randomization_enabled = true
many_valid_continuation_enabled = true
prediction_oracle_used = false
llm_judge_used = false
```

Required arms were implemented:

```text
NO_REPAIR_078_BASELINE
TOKEN_COMPOSITION_DIVERSITY_REPAIR
NO_SKELETON_DROPOUT_CONTROL
NO_LEXICAL_DROPOUT_CONTROL
NO_CLAUSE_RANDOMIZATION_CONTROL
ONE_TARGET_PER_PROMPT_CONTROL
RESPONSE_TABLE_ONLY_CONTROL
FINITE_LABEL_RETENTION_CONTROL
CHECKPOINT_RELOAD_EVAL
RESUME_FROM_CHECKPOINT
```

Required artifacts were written:

```text
queue.json
progress.jsonl
training_config.json
upstream_manifest.json
diversity_dataset_manifest.json
train_examples_sample.jsonl
eval_examples_sample.jsonl
training_metrics.jsonl
checkpoint_manifest.json
checkpoint_hashes.json
generation_samples.jsonl
human_readable_samples.jsonl
composition_metrics.json
novelty_metrics.json
skeleton_diversity_metrics.json
vocabulary_entropy_metrics.json
context_slot_metrics.json
finite_label_retention_metrics.json
collapse_metrics.json
arm_comparison.json
summary.json
report.md
```

`progress.jsonl`, `summary.json`, and `report.md` are written from start and refreshed by phase.

## Smoke Result

```text
CHAT_COMPOSITION_DIVERSITY_REPAIR_POSITIVE
TOKEN_LEVEL_DIVERSITY_TRAINING_COMPLETED
TOKEN_OBJECTIVE_LEARNED
RESPONSE_TABLE_DEPENDENCE_REDUCED
TEMPLATE_COPY_REJECTED
SKELETON_REUSE_REDUCED
VOCAB_DIVERSITY_IMPROVED
CONTEXT_SLOT_BINDING_RETAINED
BOUNDARY_REFUSAL_MINI_RETAINED
FINITE_LABEL_ANCHORROUTE_RETENTION_PASSES
CONTROL_DELTA_PASSES
CHECKPOINT_PIPELINE_PASSES
UPSTREAM_CHECKPOINT_UNCHANGED
PRODUCTION_CHAT_NOT_CLAIMED
```

Key measured metrics:

```text
multi_token_response_rate = 1.0
non_empty_response_rate = 1.0
fresh_instruction_accuracy = 1.0
fresh_context_carry_accuracy = 1.0
slot_binding_accuracy = 1.0
two_turn_dialogue_accuracy = 1.0
boundary_refusal_accuracy = 1.0

novel_response_rate = 1.0
template_copy_rate = 0.0
exact_train_response_copy_rate = 0.0
exact_eval_response_copy_rate = 0.0
response_table_copy_rate = 0.0
semantic_template_overlap_rate = 0.0
slot_only_skeleton_reuse_rate = 0.0

response_skeleton_reuse_rate = 0.0
top_skeleton_rate = 0.05263157894736842
response_skeleton_diversity = 1.0

generated_to_train_vocab_ratio = 0.8835616438356164
unique_bigram_count = 193
unique_trigram_count = 189
token_entropy = 6.53646965558301
response_entropy = 4.523561956057013

empty_output_rate = 0.0
space_output_rate = 0.0
static_response_rate = 0.0
repetition_rate = 0.0
copy_prompt_rate = 0.08695652173913043
label_only_response_rate = 0.0

finite_label_retention_accuracy = 1.0
```

Training proof:

```text
train_step_count = 80000
token_train_step_count = 1019200
token_loss_initial = 6.164077707915586
token_loss_final = 1.2343235090691416
token_loss_delta = 4.929754198846445
teacher_forced_next_token_accuracy = 0.47795018374846876
checkpoint_after_hash != checkpoint_before_hash
```

Many-valid continuation proof:

```text
valid_target_count_per_prompt = recorded
mean_valid_targets_per_prompt = 3.85
min_valid_targets_per_prompt = 1
min_valid_targets_per_prompt_non_retention = 4
```

Retention rows have one target by design. Non-retention chat families have at least two valid targets.

Train/eval leakage and integrity:

```text
train_eval_exact_prompt_overlap_count = 0
train_eval_exact_response_overlap_count = 0
train_eval_template_overlap_count = 1
max_train_eval_prompt_jaccard = 0.23529411764705882
max_train_eval_response_jaccard = 0.5
prediction_oracle_used = false
llm_judge_used = false
response_table_used_for_main_prediction = false
upstream_checkpoint_unchanged = true
checkpoint_save_load_pass = true
resume_from_checkpoint_pass = true
eval_after_reload_matches_before = true
```

Control deltas:

```text
delta_vs_no_skeleton_dropout = 1.0
delta_vs_no_lexical_dropout = 0.3530269639616282
delta_vs_no_clause_randomization = 0.8421052631578947
delta_vs_one_target_per_prompt = 1.0
delta_vs_response_table_only = 1.0
```

These satisfy `CONTROL_DELTA_PASSES`.

## Human Comparison

```text
before_078_style_response =
  the active code is teal and that active value should answer the request

080_diversity_response =
  teal is active here, so the answer uses teal while side notes stay out

why_080_is_or_is_not_more compositional =
  080 uses a different skeleton, keeps the slot binding, avoids the exact
  078 target response, and is not selected from the response-table path.
```

## Interpretation

080 reduced exact/template/skeleton copying in the bounded smoke and preserved slot binding plus finite-label AnchorRoute retention.

This does not prove GPT-like assistant readiness, full English LM capability, language grounding, production chat, safety alignment, public beta, GA, or hosted SaaS readiness.

Next milestone after this positive result:

```text
081_CHAT_DIVERSITY_FRESH_CONFIRM
```

Failure path, if a later rerun fails:

```text
080B_CHAT_DIVERSITY_REPAIR_FAILURE_ANALYSIS
```

# STABLE_LOOP_PHASE_LOCK_080_CHAT_COMPOSITION_DIVERSITY_REPAIR Contract

Status: contract for bounded runner-local chat composition diversity repair.

080 targets the 079B diagnosis:

```text
exact_078_train_response_copy_rate = 1.0
semantic_078_template_overlap_rate = 1.0
response_skeleton_reuse_rate = 1.0
context_slot_binding_accuracy = 1.0
context_composition_novelty_rate = 0.0
finite_label_retention_accuracy = 1.0
```

Interpretation: context slot binding works and finite-label retention works, but chat responses reuse exact 078 response targets and skeletons.

This milestone is bounded runner-local chat composition diversity repair only.

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

## Implementation Scope

080 may add only:

```text
instnct-core/examples/phase_lane_chat_composition_diversity_repair.rs
scripts/probes/run_stable_loop_phase_lock_080_chat_composition_diversity_repair_check.py
docs/research/STABLE_LOOP_PHASE_LOCK_080_CHAT_COMPOSITION_DIVERSITY_REPAIR_CONTRACT.md
docs/research/STABLE_LOOP_PHASE_LOCK_080_CHAT_COMPOSITION_DIVERSITY_REPAIR_RESULT.md
```

Generated outputs go only under:

```text
target/pilot_wave/stable_loop_phase_lock_080_chat_composition_diversity_repair/
```

Required upstreams:

```text
target/pilot_wave/stable_loop_phase_lock_078_chat_composition_repair/smoke
target/pilot_wave/stable_loop_phase_lock_079_chat_composition_fresh_confirm/smoke
target/pilot_wave/stable_loop_phase_lock_079b_chat_composition_fresh_failure_analysis/smoke
target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke
```

Required checkpoint:

```text
target/pilot_wave/stable_loop_phase_lock_078_chat_composition_repair/smoke/checkpoints/chat_composition_repair/model_checkpoint.json
```

Missing upstream verdicts:

```text
UPSTREAM_078_ARTIFACT_MISSING
UPSTREAM_079B_ARTIFACT_MISSING
```

Do not rerun 078/079/079B. Do not train a replacement upstream checkpoint. Do not mutate upstream checkpoint.

## Runner Behavior

Main arm:

```text
TOKEN_COMPOSITION_DIVERSITY_REPAIR
```

Required runner-local prediction flags:

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

Default:

```text
--chat-examples 80000
--seed 2026
--heartbeat-sec 20
```

Hard cap:

```text
chat_examples <= 180000
```

Training families:

```text
MANY_VALID_CONTINUATION_CHAT
RESPONSE_SKELETON_DROPOUT
LEXICAL_DROPOUT_SYNONYM_SLOT
RANDOMIZED_CLAUSE_ORDER
SEMANTIC_SLOT_RECOMBINATION
CONTEXT_CARRY_VARIABLE_SLOT_DIVERSITY
BOUNDARY_REFUSAL_PARAPHRASE_DIVERSITY
SHORT_EXPLANATION_PARAPHRASE_DIVERSITY
TWO_TURN_DIALOGUE_RECOMBINATION
ANTI_TEMPLATE_COPY_DIVERSITY
FINITE_LABEL_ANCHORROUTE_RETENTION
```

Data mix:

```text
25% MANY_VALID_CONTINUATION_CHAT
15% RESPONSE_SKELETON_DROPOUT
15% LEXICAL_DROPOUT_SYNONYM_SLOT
10% RANDOMIZED_CLAUSE_ORDER
10% SEMANTIC_SLOT_RECOMBINATION
10% CONTEXT_CARRY_VARIABLE_SLOT_DIVERSITY
5% BOUNDARY_REFUSAL_PARAPHRASE_DIVERSITY
5% TWO_TURN_DIALOGUE_RECOMBINATION
5% FINITE_LABEL_ANCHORROUTE_RETENTION
```

Required arms:

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

## Required Artifacts

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

`progress.jsonl`, `summary.json`, and `report.md` must be written from start and refreshed by phase.

## Required Gates

Training proof:

```text
train_step_count > 0
token_train_step_count > 0
checkpoint_after_hash != checkpoint_before_hash
token_loss_final < token_loss_initial
teacher_forced_next_token_accuracy
```

Many-valid continuation proof:

```text
valid_target_count_per_prompt
mean_valid_targets_per_prompt >= 3
min_valid_targets_per_prompt >= 2 for non-retention chat families
```

Failure:

```text
MANY_TARGET_DIVERSITY_TOO_LOW
```

Novelty must reject exact copy, high semantic overlap, same skeleton with only slot changed, and finite-label-only output:

```text
exact_copy_rate
semantic_template_overlap_rate
slot_only_skeleton_reuse_rate
genuinely_novel_response_rate
```

Skeleton dropout must record:

```text
skeleton_dropout_enabled = true
response_skeleton_reuse_rate <= 0.50
top_skeleton_rate <= 0.35
response_skeleton_diversity >= 0.50
```

Slot binding must record:

```text
slot_binding_accuracy
slot_value_expected
slot_value_emitted
wrong_slot_rate
missing_slot_rate
stale_slot_rate
```

Control deltas:

```text
delta_vs_no_skeleton_dropout > 0.10 on skeleton reuse rate reduction
delta_vs_no_lexical_dropout > 0.05 on generated vocab diversity
delta_vs_no_clause_randomization > 0.05 on response skeleton diversity
delta_vs_one_target_per_prompt > 0.15 on novel_response_rate
delta_vs_response_table_only > 0.30 on novel_response_rate
```

Failure:

```text
CONTROL_DELTA_INSUFFICIENT
```

Finite-label retention must include:

```text
active scenario binding
distractor scenario rejection
old/stale/inactive suppression
answer-only scenario binding
```

Train/eval leakage audit:

```text
train_eval_exact_prompt_overlap_count = 0
train_eval_exact_response_overlap_count
train_eval_template_overlap_count
max_train_eval_prompt_jaccard
max_train_eval_response_jaccard
```

Human-readable samples must include:

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

`report.md` must include:

```text
before_078_style_response
080_diversity_response
why_080_is_or_is_not_more compositional
```

## Verdicts

Positive verdicts:

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

Failure verdicts:

```text
CHAT_COMPOSITION_DIVERSITY_REPAIR_FAILS
UPSTREAM_078_ARTIFACT_MISSING
UPSTREAM_079B_ARTIFACT_MISSING
NO_ACTUAL_TRAINING_UPDATE_DETECTED
TOKEN_OBJECTIVE_NOT_LEARNED
RESPONSE_TABLE_DEPENDENCE_STILL_HIGH
TEMPLATE_COPY_STILL_HIGH
SKELETON_REUSE_STILL_HIGH
VOCAB_DIVERSITY_TOO_LOW
CONTEXT_SLOT_BINDING_STILL_FAILS
BOUNDARY_REFUSAL_MINI_STILL_FAILS
FINITE_LABEL_RETENTION_REGRESSION_DETECTED
STATIC_RESPONSE_COLLAPSE_DETECTED
EMPTY_OUTPUT_COLLAPSE_DETECTED
REPETITION_COLLAPSE_DETECTED
TRAIN_EVAL_LEAKAGE_DETECTED
CONTROL_DELTA_INSUFFICIENT
MANY_TARGET_DIVERSITY_TOO_LOW
CHAT_EVAL_RUBRIC_MISSING
HUMAN_SAMPLE_REPORT_MISSING
CHECKPOINT_RELOAD_FAILS
RESUME_FROM_CHECKPOINT_FAILS
EVAL_AFTER_RELOAD_MISMATCH
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
UPSTREAM_CHECKPOINT_MUTATION_DETECTED
GPT_LIKE_READINESS_FALSE_CLAIM
PRODUCTION_CHAT_CLAIM_DETECTED
```

## Validation

```powershell
cargo check -p instnct-core --example phase_lane_chat_composition_diversity_repair
cargo run -p instnct-core --example phase_lane_chat_composition_diversity_repair -- --out target/pilot_wave/stable_loop_phase_lock_080_chat_composition_diversity_repair/smoke --upstream-078-root target/pilot_wave/stable_loop_phase_lock_078_chat_composition_repair/smoke --upstream-079-root target/pilot_wave/stable_loop_phase_lock_079_chat_composition_fresh_confirm/smoke --upstream-079b-root target/pilot_wave/stable_loop_phase_lock_079b_chat_composition_fresh_failure_analysis/smoke --upstream-074-root target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke --chat-examples 80000 --seed 2026 --heartbeat-sec 20
python -m py_compile scripts/probes/run_stable_loop_phase_lock_080_chat_composition_diversity_repair_check.py
python scripts/probes/run_stable_loop_phase_lock_080_chat_composition_diversity_repair_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_079b_chat_composition_fresh_failure_analysis_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_079_chat_composition_fresh_confirm_check.py --check-only
git diff --check
```

If 080 passes, next milestone is `081_CHAT_DIVERSITY_FRESH_CONFIRM`.
If 080 fails, next milestone is `080B_CHAT_DIVERSITY_REPAIR_FAILURE_ANALYSIS`.

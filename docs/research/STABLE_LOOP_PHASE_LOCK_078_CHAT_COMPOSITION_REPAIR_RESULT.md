# STABLE_LOOP_PHASE_LOCK_078_CHAT_COMPOSITION_REPAIR Result

Status: implementation result for bounded runner-local chat composition repair.

078 targets the 077/077B chat failure where the 076 chat PoC produced
multi-token outputs but copied response-table/template responses instead of
composing fresh responses.

This milestone is runner-local and uses token-level next-token repair only.

response_table_used_for_main_prediction = false
decoder_path = token_level_next_token
response_table_path_available_but_disabled = true
llm_judge_used = false

not GPT-like assistant readiness
not full English LM
not production chat
not safety alignment
not public beta
not GA
not hosted SaaS

no service API change
no deployment harness change
no SDK/public export change
no release docs change
no root LICENSE change
no upstream checkpoint mutation

## Implemented Files

078 adds only:

```text
instnct-core/examples/phase_lane_chat_composition_repair.rs
scripts/probes/run_stable_loop_phase_lock_078_chat_composition_repair_check.py
docs/research/STABLE_LOOP_PHASE_LOCK_078_CHAT_COMPOSITION_REPAIR_CONTRACT.md
docs/research/STABLE_LOOP_PHASE_LOCK_078_CHAT_COMPOSITION_REPAIR_RESULT.md
```

Generated outputs are written only under:

```text
target/pilot_wave/stable_loop_phase_lock_078_chat_composition_repair/
```

The runner requires:

```text
076 chat root/checkpoint
077 fresh composition failure root
077B failure analysis root
074 finite-label scenario-state confirmation root
```

Failure verdicts for missing or invalid upstreams:

```text
UPSTREAM_076_ARTIFACT_MISSING
UPSTREAM_077B_ARTIFACT_MISSING
UPSTREAM_CHECKPOINT_MUTATION_DETECTED
```

## Runner Behavior

The runner trains a new experimental checkpoint under `target/` and keeps the
076 upstream checkpoint read-only.

Main arm:

```text
TOKEN_COMPOSITION_REPAIR
```

Main prediction does not use direct response-table lookup:

```text
response_table_used_for_main_prediction = false
decoder_path = token_level_next_token
response_table_path_available_but_disabled = true
```

Token objective proof is recorded:

```text
token_train_step_count
token_loss_initial
token_loss_final
token_loss_delta
teacher_forced_next_token_accuracy
checkpoint_before_hash
checkpoint_after_hash
```

Required proof:

```text
train_step_count > 0
checkpoint_after_hash != checkpoint_before_hash
token_loss_final < token_loss_initial
```

Failure verdicts:

```text
NO_ACTUAL_TRAINING_UPDATE_DETECTED
TOKEN_OBJECTIVE_NOT_LEARNED
ORACLE_SHORTCUT_DETECTED
```

Default command uses:

```text
--chat-examples 60000
--seed 2026
--heartbeat-sec 20
```

Hard cap:

```text
chat_examples <= 150000
```

Training mix:

```text
SIMPLE_INSTRUCTION_PARAPHRASE = 30%
CONTEXT_CARRY_VARIABLE_SLOT = 20%
SHORT_EXPLANATION_MANY_TARGET = 15%
TWO_TURN_DIALOGUE_STATE = 10%
BOUNDARY_REFUSAL_PARAPHRASE_MINI = 10%
ANCHORROUTE_FINITE_LABEL_RETENTION = 10%
ANTI_TEMPLATE_COPY_DROPOUT = 5%
```

Required arms:

```text
NO_REPAIR_076_BASELINE
RESPONSE_TABLE_ONLY_CONTROL
TOKEN_COMPOSITION_REPAIR
TOKEN_COMPOSITION_NO_DROPOUT_CONTROL
NO_CONTEXT_SLOT_CONTROL
FINITE_LABEL_RETENTION_CONTROL
CHECKPOINT_RELOAD_EVAL
RESUME_FROM_CHECKPOINT
```

## Artifacts

Required artifacts:

```text
queue.json
progress.jsonl
training_config.json
upstream_manifest.json
repair_dataset_manifest.json
train_examples_sample.jsonl
eval_examples_sample.jsonl
training_metrics.jsonl
checkpoint_manifest.json
checkpoint_hashes.json
generation_samples.jsonl
human_readable_samples.jsonl
composition_metrics.json
novelty_metrics.json
context_slot_metrics.json
finite_label_retention_metrics.json
collapse_metrics.json
arm_comparison.json
summary.json
report.md
```

`progress.jsonl`, `summary.json`, and `report.md` are written from start and
refreshed during major phases.

## Result Metrics

Smoke result:

```text
CHAT_COMPOSITION_REPAIR_POSITIVE
```

Key measured metrics:

```text
multi_token_response_rate = 1.0
non_empty_response_rate = 1.0
fresh_instruction_accuracy = 0.75
fresh_context_carry_accuracy = 1.0
slot_binding_accuracy = 1.0
two_turn_dialogue_accuracy = 1.0
boundary_refusal_accuracy = 1.0
novel_response_rate = 0.8518518518518519
template_copy_rate = 0.14814814814814814
response_table_copy_rate = 0.14814814814814814
exact_train_response_copy_rate = 0.14814814814814814
label_only_response_rate = 0.14814814814814814
finite_label_retention_accuracy = 1.0
empty_output_rate = 0.0
space_output_rate = 0.0
static_response_rate = 0.0
repetition_rate = 0.0
copy_prompt_rate = 0.07407407407407407
train_eval_exact_prompt_overlap_count = 0
```

Token objective proof:

```text
token_train_step_count = 810000
token_loss_initial = 0.6931471805600268
token_loss_final = 0.39368021734109904
token_loss_delta = 0.29946696321892774
teacher_forced_next_token_accuracy = 0.8142135642135642
checkpoint_after_hash != checkpoint_before_hash
```

Control deltas:

```text
delta_vs_response_table_only_control = 0.8518518518518519
delta_vs_no_context_slot_control = 1.0
delta_vs_no_dropout_control = 0.8518518518518519
```

Control failure verdict if insufficient:

```text
CONTROL_DELTA_INSUFFICIENT
```

Context slot report includes:

```text
slot_value_expected
slot_value_emitted
slot_binding_accuracy
wrong_slot_rate
missing_slot_rate
stale_slot_rate
```

Novelty report includes:

```text
exact_template_copy_rate
semantic_template_overlap_rate
response_table_copy_rate
novel_response_rate
train_response_ngram_overlap
```

Human-readable samples include:

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
slot_binding_diagnosis
short_diagnosis
```

Boundary refusal is controlled and small:

```text
boundary_refusal_accuracy is not safety alignment
no production safety claim
no clinical/high-stakes readiness
```

## Verdicts

Positive verdicts:

```text
CHAT_COMPOSITION_REPAIR_POSITIVE
TOKEN_LEVEL_COMPOSITION_TRAINING_COMPLETED
TOKEN_OBJECTIVE_LEARNED
RESPONSE_TABLE_DEPENDENCE_REDUCED
TEMPLATE_COPY_REJECTED
CONTEXT_SLOT_BINDING_REPAIRED
BOUNDARY_REFUSAL_MINI_REPAIRED
FINITE_LABEL_ANCHORROUTE_RETENTION_PASSES
CONTROL_DELTA_PASSES
CHECKPOINT_PIPELINE_PASSES
UPSTREAM_CHECKPOINT_UNCHANGED
PRODUCTION_CHAT_NOT_CLAIMED
```

Failure verdicts:

```text
CHAT_COMPOSITION_REPAIR_FAILS
RESPONSE_TABLE_DEPENDENCE_STILL_HIGH
TEMPLATE_COPY_STILL_HIGH
CONTEXT_SLOT_BINDING_STILL_FAILS
BOUNDARY_REFUSAL_MINI_STILL_FAILS
FINITE_LABEL_RETENTION_REGRESSION_DETECTED
STATIC_RESPONSE_COLLAPSE_DETECTED
EMPTY_OUTPUT_COLLAPSE_DETECTED
REPETITION_COLLAPSE_DETECTED
TRAIN_EVAL_LEAKAGE_DETECTED
CHAT_EVAL_RUBRIC_MISSING
HUMAN_SAMPLE_REPORT_MISSING
CHECKPOINT_RELOAD_FAILS
RESUME_FROM_CHECKPOINT_FAILS
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
```

## Validation Commands

```powershell
cargo check -p instnct-core --example phase_lane_chat_composition_repair
cargo run -p instnct-core --example phase_lane_chat_composition_repair -- --out target/pilot_wave/stable_loop_phase_lock_078_chat_composition_repair/smoke --upstream-076-root target/pilot_wave/stable_loop_phase_lock_076_chat_generation_poc/smoke --upstream-077-root target/pilot_wave/stable_loop_phase_lock_077_chat_generation_fresh_composition_confirm/smoke --upstream-077b-root target/pilot_wave/stable_loop_phase_lock_077b_chat_generation_failure_analysis/smoke --upstream-074-root target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke --chat-examples 60000 --seed 2026 --heartbeat-sec 20
python -m py_compile scripts/probes/run_stable_loop_phase_lock_078_chat_composition_repair_check.py
python scripts/probes/run_stable_loop_phase_lock_078_chat_composition_repair_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_077b_chat_generation_failure_analysis_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_077_chat_generation_fresh_composition_confirm_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_076_chat_generation_poc_check.py --check-only
git diff --check
```

## Next Milestones

If 078 passes:

```text
079_CHAT_COMPOSITION_FRESH_CONFIRM
```

If 078 fails:

```text
078B_CHAT_COMPOSITION_REPAIR_FAILURE_ANALYSIS
```

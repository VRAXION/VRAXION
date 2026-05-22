# STABLE_LOOP_PHASE_LOCK_079_CHAT_COMPOSITION_FRESH_CONFIRM Contract

079 is a bounded fresh chat composition confirm only. It evaluates the existing 078
`TOKEN_COMPOSITION_REPAIR` checkpoint on fresh prompts and checks whether the
token-level repair generalizes without falling back to response-table or
template-copy behavior.

079 is eval-only: no training, no resume, no checkpoint repair, no checkpoint
mutation, no replacement checkpoint, and no decoder weight update. The decoder
path remains runner-local in `phase_lane_chat_composition_fresh_confirm.rs`.
This milestone makes no service API change, no deployment harness change,
no SDK/public export change, no release docs change, no root LICENSE change,
and no upstream checkpoint mutation.

This milestone is not GPT-like assistant readiness, not full English LM,
not language grounding, not production chat, not safety alignment,
not public beta, not GA, and not hosted SaaS.

## Scope

Add:

```text
instnct-core/examples/phase_lane_chat_composition_fresh_confirm.rs
scripts/probes/run_stable_loop_phase_lock_079_chat_composition_fresh_confirm_check.py
docs/research/STABLE_LOOP_PHASE_LOCK_079_CHAT_COMPOSITION_FRESH_CONFIRM_CONTRACT.md
docs/research/STABLE_LOOP_PHASE_LOCK_079_CHAT_COMPOSITION_FRESH_CONFIRM_RESULT.md
```

Generated artifacts are written only under:

```text
target/pilot_wave/stable_loop_phase_lock_079_chat_composition_fresh_confirm/
```

## Required Upstreams

079 requires:

```text
078 chat composition repair root
078 TOKEN_COMPOSITION_REPAIR checkpoint
078 summary.json
078 checkpoint_manifest.json
078 train/eval samples
078 generation_samples.jsonl
077B failure analysis root
076 chat generation PoC root
074 finite-label scenario-state confirm root
```

If any required upstream is missing, 079 emits `UPSTREAM_078_ARTIFACT_MISSING`.
It does not rerun 076, 077, 077B, or 078, and it does not train a replacement
checkpoint.

## Eval-Only Hard Wall

079 records:

```text
upstream_078_summary_present = true
upstream_078_positive = true
checkpoint_exists = true
eval_started_after_079_start = true
checkpoint_hash_before
checkpoint_hash_after
checkpoint_hash_unchanged = true
train_step_count = 0
prediction_oracle_used = false
llm_judge_used = false
decoder_path = token_level_next_token
response_table_used_for_main_prediction = false
```

Failures include:

```text
STALE_CHECKPOINT_ARTIFACT_USED
TRAINING_SIDE_EFFECT_DETECTED
CHECKPOINT_MUTATION_DETECTED
ORACLE_SHORTCUT_DETECTED
RESPONSE_TABLE_PATH_USED_IN_EVAL
```

## Fresh Prompt Audit

Fresh prompts use new wording, entities, instruction shapes, context-carry
variants, and refusal phrasing. They are compared against 078 train prompts, 078
eval prompts, 077 prompts, and 076 train/eval prompts.

079 records:

```text
overlap_with_078_train_prompt_count
overlap_with_078_eval_prompt_count
overlap_with_077_prompt_count
overlap_with_076_prompt_count
max_prompt_token_jaccard_vs_078_train
max_prompt_token_jaccard_vs_078_eval
max_prompt_token_jaccard_vs_077
max_prompt_token_jaccard_vs_076
near_duplicate_prompt_count
```

Exact prompt overlap or any near duplicate with token Jaccard >= 0.90 fails with
`FRESH_PROMPT_LEAKAGE_DETECTED`.

## Template-Copy Audit

Generated outputs are compared against:

```text
078 train responses
078 eval outputs
078 generated outputs
077 generated outputs
076 response table outputs
076 train/eval outputs
```

079 records:

```text
exact_train_response_copy_rate
exact_eval_response_copy_rate
response_table_copy_rate
semantic_template_overlap_rate
template_copy_rate
novel_response_rate
```

If the fresh eval still copies templates, 079 emits `TEMPLATE_COPY_DETECTED`.

## Required Artifacts

```text
queue.json
progress.jsonl
benchmark_config.json
upstream_078_manifest.json
checkpoint_manifest.json
fresh_chat_eval_dataset.jsonl
generation_samples.jsonl
human_readable_samples.jsonl
composition_metrics.json
novelty_metrics.json
context_slot_metrics.json
finite_label_retention_metrics.json
collapse_metrics.json
summary.json
report.md
```

`progress.jsonl`, `summary.json`, and `report.md` are written from start and
refreshed by phase.

## Required Metrics

```text
multi_token_response_rate
non_empty_response_rate
fresh_instruction_accuracy
fresh_context_carry_accuracy
slot_binding_accuracy
two_turn_dialogue_accuracy
boundary_refusal_accuracy
label_only_response_rate
generated_token_count_mean
generated_token_count_min
sentence_like_response_rate
empty_output_rate
space_output_rate
static_response_rate
repetition_rate
copy_prompt_rate
finite_label_retention_accuracy
```

The context slot report includes:

```text
slot_value_expected
slot_value_emitted
wrong_slot_rate
missing_slot_rate
stale_slot_rate
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
novelty_flag
template_copy_flag
semantic_template_overlap_score
slot_binding_diagnosis
short_diagnosis
```

Finite-label retention includes 074-style rows:

```text
active scenario binding
distractor scenario rejection
old/stale/inactive suppression
answer-only scenario binding
```

## Verdicts

Positive verdicts:

```text
CHAT_COMPOSITION_FRESH_CONFIRM_POSITIVE
NO_TRAINING_PERFORMED
CHECKPOINT_UNCHANGED
FRESH_MULTI_TOKEN_RESPONSES_PASS
FRESH_INSTRUCTION_COMPOSITION_PASSES
FRESH_CONTEXT_SLOT_BINDING_PASSES
FRESH_TEMPLATE_COPY_REJECTED
FRESH_NOVEL_RESPONSES_PASS
FINITE_LABEL_ANCHORROUTE_RETENTION_PASSES
STATIC_RESPONSE_COLLAPSE_REJECTED
PRODUCTION_CHAT_NOT_CLAIMED
```

Failure verdicts:

```text
CHAT_COMPOSITION_FRESH_CONFIRM_FAILS
UPSTREAM_078_ARTIFACT_MISSING
FRESH_PROMPT_LEAKAGE_DETECTED
TEMPLATE_COPY_DETECTED
LABEL_ONLY_RESPONSE_COLLAPSE_DETECTED
FRESH_INSTRUCTION_COMPOSITION_FAILS
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
cargo check -p instnct-core --example phase_lane_chat_composition_fresh_confirm
cargo run -p instnct-core --example phase_lane_chat_composition_fresh_confirm -- --out target/pilot_wave/stable_loop_phase_lock_079_chat_composition_fresh_confirm/smoke --upstream-078-root target/pilot_wave/stable_loop_phase_lock_078_chat_composition_repair/smoke --upstream-077b-root target/pilot_wave/stable_loop_phase_lock_077b_chat_generation_failure_analysis/smoke --upstream-076-root target/pilot_wave/stable_loop_phase_lock_076_chat_generation_poc/smoke --upstream-074-root target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke --seed 2027 --heartbeat-sec 20
python -m py_compile scripts/probes/run_stable_loop_phase_lock_079_chat_composition_fresh_confirm_check.py
python scripts/probes/run_stable_loop_phase_lock_079_chat_composition_fresh_confirm_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_078_chat_composition_repair_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_077b_chat_generation_failure_analysis_check.py --check-only
git diff --check
```

If 079 passes, next is `080_CHAT_COMPOSITION_MULTI_SEED_CONFIRM`.
If 079 fails, next is `079B_CHAT_COMPOSITION_FRESH_FAILURE_ANALYSIS`.

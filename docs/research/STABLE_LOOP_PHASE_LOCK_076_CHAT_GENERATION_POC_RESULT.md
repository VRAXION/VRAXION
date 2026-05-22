# STABLE_LOOP_PHASE_LOCK_076_CHAT_GENERATION_POC Result

Status: implementation result for the first bounded, runner-local chat
generation proof of concept after 075 confirmed that the 072/074 checkpoint had
no chat/free-form surface.

076 creates a small experimental decoder/generation loop inside the research
runner only, trains a controlled chat SFT surface, and checks that the
confirmed finite-label AnchorRoute scenario-state capability is retained.

This is runner-local chat generation PoC only.

no product API
no SDK surface
no service API change
no deployment harness change
no release docs change
no public crate export change
no root LICENSE change
no full English LM training
no production chat
no ChatGPT-like assistant readiness
no language grounding
no safety alignment
no public beta
no GA
no hosted SaaS

## Implementation

Runner:

```text
instnct-core/examples/phase_lane_chat_generation_poc.rs
```

Static checker:

```text
scripts/probes/run_stable_loop_phase_lock_076_chat_generation_poc_check.py
```

Research docs:

```text
docs/research/STABLE_LOOP_PHASE_LOCK_076_CHAT_GENERATION_POC_CONTRACT.md
docs/research/STABLE_LOOP_PHASE_LOCK_076_CHAT_GENERATION_POC_RESULT.md
```

Generated checkpoints and artifacts are written only under:

```text
target/pilot_wave/stable_loop_phase_lock_076_chat_generation_poc/
```

The decoder/generation loop is runner-local. It is not exposed through public
crate exports, service API, deployment harness, release docs, SDK candidate, or
root `LICENSE`.

## Runner-Local Chat Stack

The implemented runner contains:

```text
fixed vocabulary from controlled chat SFT examples
simple token-level decoder/generation loop
deterministic greedy decode
seeded sampling allowed only as an audit mode
default max response length 64 tokens
stop token support
runner-local decoder
```

Required records:

```text
upstream_checkpoint_hash_before
upstream_checkpoint_hash_after
upstream_checkpoint_unchanged = true
train_step_count > 0
checkpoint_before_hash
checkpoint_after_hash
checkpoint_after_hash != checkpoint_before_hash
checkpoint_save_load_pass = true
resume_from_checkpoint_pass = true
eval_after_reload_matches_before = true
prediction_oracle_used = false
response_uses_decoder_loop = true
```

The prediction path does not use expected answer, answer template ID, task
family as direct answer, solution metadata, or oracle parser.

## Data Mix

The controlled chat SFT dataset uses:

```text
SIMPLE_INSTRUCTION_CHAT = 50%
SHORT_ANSWER_EXPLANATION = 20%
CONTEXT_CARRY_CHAT = 15%
ANCHORROUTE_RETENTION = 10%
BOUNDARY_REFUSAL = 5%
```

Leakage audit fields:

```text
train_eval_exact_prompt_overlap_count = 0
train_eval_exact_response_overlap_count
train_eval_template_overlap_count
eval_prompt_hash
train_prompt_hash
```

## Required Artifacts

The runner writes:

```text
queue.json
progress.jsonl
training_config.json
upstream_manifest.json
chat_sft_dataset_manifest.json
train_examples_sample.jsonl
eval_examples_sample.jsonl
training_metrics.jsonl
checkpoint_manifest.json
checkpoint_hashes.json
generation_samples.jsonl
human_readable_samples.jsonl
chat_eval_metrics.json
finite_label_retention_metrics.json
collapse_metrics.json
summary.json
report.md
```

`progress.jsonl`, `summary.json`, and `report.md` are written from start and
refreshed during major phases.

## Rubric-Bounded Eval

No LLM judge is used. Each eval row records:

```text
prompt
expected_behavior
required_keywords
forbidden_outputs
model_output
pass_fail
diagnosis
```

`human_readable_samples.jsonl` includes:

```text
eval_family
prompt
model_output
expected_behavior
pass_fail
output_classification
short_diagnosis
```

Finite-label retention includes real 074-style rows:

```text
active scenario binding
distractor scenario rejection
inactive/stale pocket suppression
answer-only scenario binding
```

Boundary refusal remains limited:

```text
boundary_refusal_accuracy is a controlled mini-eval only
no production safety claim
no clinical/high-stakes readiness
```

## Positive Gates

```text
chat_generation_supported = true
free_form_answering_supported = true
response_uses_decoder_loop = true
multi_token_response_rate >= 0.80
label_only_response_rate <= 0.20
generated_token_count_mean > 3.0
generated_token_count_min >= 2
non_empty_response_rate >= 0.95
instruction_following_accuracy >= 0.65
context_carry_chat_accuracy >= 0.60
boundary_refusal_accuracy >= 0.70
finite_label_retention_accuracy >= 0.90
empty_output_rate <= 0.02
space_output_rate <= 0.02
top_response_rate <= 0.35
static_response_rate <= 0.20
repetition_rate <= 0.20
train_eval_exact_prompt_overlap_count = 0
prediction_oracle_used = false
checkpoint_save_load_pass = true
resume_from_checkpoint_pass = true
eval_after_reload_matches_before = true
upstream_checkpoint_unchanged = true
```

Collapse metrics include:

```text
empty_output_rate
space_output_rate
top_response_rate
static_response_rate
repetition_rate
copy_prompt_rate
unique_response_count
```

## Observed Smoke Result

The smoke run completed with:

```text
CHAT_GENERATION_POC_POSITIVE
RUNNER_LOCAL_DECODER_LOOP_CREATED
RUNNER_LOCAL_DECODER_SURFACE_CONFIRMED
CONTROLLED_CHAT_SFT_COMPLETED
MULTI_TOKEN_CHAT_OUTPUT_PRODUCED
LABEL_ONLY_CHAT_REJECTED
INSTRUCTION_FOLLOWING_CHAT_BASELINE_PASSES
CONTEXT_CARRY_CHAT_BASELINE_PASSES
RUBRIC_BOUNDED_CHAT_EVAL_RECORDED
FINITE_LABEL_ANCHORROUTE_RETENTION_PASSES
STATIC_RESPONSE_COLLAPSE_REJECTED
TRAIN_EVAL_LEAKAGE_REJECTED
UPSTREAM_CHECKPOINT_UNCHANGED
CHECKPOINT_PIPELINE_PASSES
PRODUCTION_TRAINING_NOT_CLAIMED
```

Measured smoke metrics:

```text
chat_generation_supported = true
free_form_answering_supported = true
multi_token_response_rate = 1.0
label_only_response_rate = 0.20
generated_token_count_mean = 9.8125
generated_token_count_min = 7
non_empty_response_rate = 1.0
instruction_following_accuracy = 1.0
context_carry_chat_accuracy = 1.0
boundary_refusal_accuracy = 1.0
finite_label_retention_accuracy = 1.0
empty_output_rate = 0.0
space_output_rate = 0.0
top_response_rate = 0.20
static_response_rate = 0.0
repetition_rate = 0.0
unique_response_count = 12
train_eval_exact_prompt_overlap_count = 0
train_step_count = 21280
checkpoint_save_load_pass = true
resume_from_checkpoint_pass = true
eval_after_reload_matches_before = true
upstream_checkpoint_unchanged = true
prediction_oracle_used = false
```

## Failure Verdict Coverage

The runner and checker preserve these failure verdicts:

```text
CHAT_GENERATION_POC_FAILS
CHAT_GENERATION_SURFACE_STILL_UNSUPPORTED
NO_ACTUAL_TRAINING_UPDATE_DETECTED
UPSTREAM_CHECKPOINT_MUTATION_DETECTED
ORACLE_SHORTCUT_DETECTED
TRAIN_EVAL_LEAKAGE_DETECTED
LABEL_ONLY_RESPONSE_COLLAPSE_DETECTED
STATIC_RESPONSE_COLLAPSE_DETECTED
EMPTY_OUTPUT_COLLAPSE_DETECTED
INSTRUCTION_FOLLOWING_CHAT_FAILS
CONTEXT_CARRY_CHAT_FAILS
FINITE_LABEL_RETENTION_REGRESSION_DETECTED
CHAT_EVAL_RUBRIC_MISSING
CHECKPOINT_RELOAD_FAILS
RESUME_FROM_CHECKPOINT_FAILS
EVAL_AFTER_RELOAD_MISMATCH
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
```

If any core gate fails, the runner does not emit `CHAT_GENERATION_POC_POSITIVE`.

## Validation Commands

```powershell
cargo check -p instnct-core --example phase_lane_chat_generation_poc
cargo run -p instnct-core --example phase_lane_chat_generation_poc -- --out target/pilot_wave/stable_loop_phase_lock_076_chat_generation_poc/smoke --upstream-checkpoint target/pilot_wave/stable_loop_phase_lock_072_counterfactual_scenario_binding_repair/smoke/checkpoints/scenario_gated_sidepacket_repair/model_checkpoint.json --upstream-074-root target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke --upstream-075-root target/pilot_wave/stable_loop_phase_lock_075_chat_surface_baseline_gap_analysis/smoke --seed 2026 --chat-examples 20000 --heartbeat-sec 20
python -m py_compile scripts/probes/run_stable_loop_phase_lock_076_chat_generation_poc_check.py
python scripts/probes/run_stable_loop_phase_lock_076_chat_generation_poc_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_075_chat_surface_baseline_gap_analysis_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm_check.py --check-only
git diff --check
```

# STABLE_LOOP_PHASE_LOCK_076_CHAT_GENERATION_POC Contract

Status: contract for the first bounded, runner-local chat generation proof of
concept after 075 confirmed that the 072/074 checkpoint had no chat/free-form
surface.

076 creates a small experimental decoder/generation loop inside the research
runner only. It trains a controlled chat SFT surface and checks that the
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

## Implementation Scope

076 adds only:

```text
instnct-core/examples/phase_lane_chat_generation_poc.rs
scripts/probes/run_stable_loop_phase_lock_076_chat_generation_poc_check.py
docs/research/STABLE_LOOP_PHASE_LOCK_076_CHAT_GENERATION_POC_CONTRACT.md
docs/research/STABLE_LOOP_PHASE_LOCK_076_CHAT_GENERATION_POC_RESULT.md
```

Generated checkpoints and artifacts are written only under:

```text
target/pilot_wave/stable_loop_phase_lock_076_chat_generation_poc/
```

The decoder/generation logic must remain runner-local. It must not be exposed
through public crate exports, service API, deployment harness, release docs, SDK
candidate, or root `LICENSE`.

## Upstream Inputs

Default upstream checkpoint:

```text
target/pilot_wave/stable_loop_phase_lock_072_counterfactual_scenario_binding_repair/smoke/checkpoints/scenario_gated_sidepacket_repair/model_checkpoint.json
```

Default upstream proof roots:

```text
target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke
target/pilot_wave/stable_loop_phase_lock_075_chat_surface_baseline_gap_analysis/smoke
```

The upstream checkpoint is read-only. Record:

```text
upstream_checkpoint_hash_before
upstream_checkpoint_hash_after
upstream_checkpoint_unchanged = true
```

Failure:

```text
UPSTREAM_CHECKPOINT_MUTATION_DETECTED
```

## Runner-Local Chat Stack

The runner adds a minimal isolated chat stack:

```text
fixed vocabulary from controlled chat SFT examples
simple token-level decoder/generation loop
deterministic greedy decode
seeded sampling allowed only as an audit mode
default max response length 64 tokens
stop token support
runner-local decoder
```

The prediction path may use only the learned runner-local checkpoint and input
text. It must not use expected answer, answer template ID, task family as direct
answer, solution metadata, or oracle parser.

Required records:

```text
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

## Data Mix

Required controlled chat SFT mix:

```text
SIMPLE_INSTRUCTION_CHAT = 50%
SHORT_ANSWER_EXPLANATION = 20%
CONTEXT_CARRY_CHAT = 15%
ANCHORROUTE_RETENTION = 10%
BOUNDARY_REFUSAL = 5%
```

Leakage audit records:

```text
train_eval_exact_prompt_overlap_count = 0
train_eval_exact_response_overlap_count
train_eval_template_overlap_count
eval_prompt_hash
train_prompt_hash
```

Failure:

```text
TRAIN_EVAL_LEAKAGE_DETECTED
```

## Required Artifacts

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

No LLM judge is allowed. Each eval row records:

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

Finite-label retention eval must include real 074-style rows:

```text
active scenario binding
distractor scenario rejection
inactive/stale pocket suppression
answer-only scenario binding
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

Boundary refusal is limited:

```text
boundary_refusal_accuracy is a controlled mini-eval only
no production safety claim
no clinical/high-stakes readiness
```

## Verdicts

Positive verdicts include:

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

Failure verdicts include:

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

If any core gate fails, do not emit `CHAT_GENERATION_POC_POSITIVE`.

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

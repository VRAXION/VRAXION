# STABLE_LOOP_PHASE_LOCK_069_MODEL_CAPABILITY_BENCHMARK_GATE Contract

Status: contract for the 069 eval-only model capability benchmark gate.

069 measures the current 068 `MIXED_WITH_ROUTE_GRAMMAR_ON` checkpoint as it
exists today. The checkpoint is a finite-label/extractive classifier-style
surface, not an open-ended assistant and not a general language model.

no retraining
no production training
no full English LM capability
no perplexity support
no free-form generation
no language grounding
no GA
no public beta
no hosted SaaS
no clinical use
no high-stakes education use
no full VRAXION
no consciousness
no biological/FlyWire equivalence
no physical quantum behavior

## Target Artifacts

Default checkpoint:

```text
target/pilot_wave/stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale/confirm/seed_2028/checkpoints/MIXED_WITH_ROUTE_GRAMMAR_ON/model_checkpoint.json
```

Default upstream summary:

```text
target/pilot_wave/stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale/confirm/summary.json
```

If either artifact is missing, the runner must emit
`UPSTREAM_068_ARTIFACT_MISSING`, stop, and not rerun 067, not rerun 068, and
not train a replacement model.

## Required Commands

```powershell
cargo check -p instnct-core --example phase_lane_model_capability_benchmark_gate
cargo run -p instnct-core --example phase_lane_model_capability_benchmark_gate -- --out target/pilot_wave/stable_loop_phase_lock_069_model_capability_benchmark_gate/smoke --checkpoint target/pilot_wave/stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale/confirm/seed_2028/checkpoints/MIXED_WITH_ROUTE_GRAMMAR_ON/model_checkpoint.json --upstream-summary target/pilot_wave/stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale/confirm/summary.json --seed 2026 --heartbeat-sec 20
python -m py_compile scripts/probes/run_stable_loop_phase_lock_069_model_capability_benchmark_check.py
python scripts/probes/run_stable_loop_phase_lock_069_model_capability_benchmark_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_067_real_text_anchorcell_training_poc_check.py --check-only
cargo test -p instnct-core sdk_candidate
git diff --check
```

## Eval-Only Guardrails

The runner must record:

```text
train_step_count = 0
checkpoint_hash_before
checkpoint_hash_after
checkpoint_hash_unchanged = true
prediction_oracle_used = false
```

If the checkpoint changes, emit `CHECKPOINT_MUTATION_DETECTED`. If any
training occurs, emit `TRAINING_SIDE_EFFECT_DETECTED`.

Prediction may use only the loaded checkpoint, input text, and fixed benchmark
config. Prediction must not use `expected_output`, the task family as answer
hint, an oracle parser, generated route answer, or benchmark solution metadata.
If violated, emit `ORACLE_SHORTCUT_DETECTED`.

## Finite-Label Honesty

Because the 068 checkpoint has a finite output label set, the runner must
record:

```text
checkpoint_label_count
benchmark_label_count
labels_not_in_checkpoint_count
unsupported_label_cases
open_ended_generation_supported = false
perplexity_supported = false
free_form_answering_supported = false
```

Unsupported open-label cases are limitation rows, not hidden failures inside
the declared finite-label task surface.

## Required Families

Per-family metrics are required for:

```text
FINEWEB_CLOSED_CONTINUATION_SELECTION
CONTEXT_ENTITY_EXTRACTION
INSTRUCTION_FOLLOWING_CLOSED
MULTI_HOP_KEY_VALUE_BINDING
COUNTERFACTUAL_BINDING
DISTRACTOR_RESISTANCE
LONG_CONTEXT_NEEDLE_BINDING
SYMBOLIC_RULE_CLOSED_CHOICE
NON_ROUTE_TEXT_CONTROL
OPEN_ENDED_INTERFACE_LIMITATION
```

Positive verdict cannot be aggregate-only. Missing or weak family metrics emit
`CAPABILITY_FAMILY_GATE_FAILS`.

## Baselines

Required baselines:

```text
MAJORITY_LABEL
ANSWER_PRIOR_ONLY
COPY_LAST_TOKEN
COPY_FIRST_MATCH
UNIGRAM_LABEL_PRIOR
SHUFFLED_CONTEXT
SHUFFLED_LABELS
NO_ROUTE_FEATURES
```

The model and baselines must evaluate the exact same supported rows:

```text
eval_row_hash_model
eval_row_hash_baselines
baseline_eval_mismatch = false
```

Mismatch emits `BASELINE_EVAL_MISMATCH`.

## Required Artifacts

```text
queue.json
progress.jsonl
benchmark_config.json
upstream_068_manifest.json
checkpoint_manifest.json
capability_dataset_manifest.json
benchmark_examples_sample.jsonl
baseline_metrics.json
capability_metrics.json
per_family_metrics.json
limitation_report.json
human_readable_samples.jsonl
failure_case_samples.jsonl
collapse_metrics.json
summary.json
report.md
```

`human_readable_samples.jsonl` must include:

```text
task_family
input
expected_output
model_output
baseline_outputs
pass_fail
limitation_flag
```

If the sample report is missing, emit `HUMAN_SAMPLE_REPORT_MISSING`.

## Capability Gates

Finite-label thresholds:

```text
context_entity_extraction_accuracy >= 0.85
instruction_following_closed_accuracy >= 0.75
multi_hop_key_value_accuracy >= 0.70
counterfactual_binding_accuracy >= 0.85
distractor_resistance_accuracy >= 0.80
long_context_needle_accuracy >= 0.65
symbolic_rule_closed_choice_accuracy >= 0.60
family_min_accuracy >= 0.60
delta_vs_majority > 0.05
delta_vs_copy_first_match > 0.05
top_output_rate <= 0.45
space_output_rate <= 0.02
empty_output_rate <= 0.02
collapse_detected = false
```

Collapse metrics must include top output rate, space output rate, empty output
rate, unique output count, output entropy, and `collapse_detected`. Collapse
failure emits `STATIC_OUTPUT_COLLAPSE_DETECTED`.

## Verdicts

Positive verdicts include:

```text
MODEL_CAPABILITY_BENCHMARK_GATE_POSITIVE
CURRENT_CHECKPOINT_CAPABILITY_PROFILE_WRITTEN
UPSTREAM_068_CHECKPOINT_VERIFIED
FINITE_LABEL_SURFACE_MEASURED
OPEN_ENDED_LIMITATION_RECORDED
INSTRUCTION_FOLLOWING_CLOSED_MEASURED
REASONING_CLOSED_CHOICE_MEASURED
LONG_CONTEXT_NEEDLE_MEASURED
BASELINE_COMPARISON_RECORDED
HUMAN_READABLE_SAMPLES_WRITTEN
NO_TRAINING_PERFORMED
PRODUCTION_TRAINING_NOT_CLAIMED
```

Failure verdicts include:

```text
MODEL_CAPABILITY_BENCHMARK_GATE_FAILS
UPSTREAM_068_ARTIFACT_MISSING
CHECKPOINT_MUTATION_DETECTED
TRAINING_SIDE_EFFECT_DETECTED
ORACLE_SHORTCUT_DETECTED
BASELINE_EVAL_MISMATCH
CAPABILITY_FAMILY_GATE_FAILS
OPEN_ENDED_CLAIM_DETECTED
PERPLEXITY_CLAIM_DETECTED
STATIC_OUTPUT_COLLAPSE_DETECTED
HUMAN_SAMPLE_REPORT_MISSING
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
```

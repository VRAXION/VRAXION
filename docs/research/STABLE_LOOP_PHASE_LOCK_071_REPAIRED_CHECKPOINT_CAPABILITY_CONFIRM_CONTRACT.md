# STABLE_LOOP_PHASE_LOCK_071_REPAIRED_CHECKPOINT_CAPABILITY_CONFIRM Contract

Status: contract for eval-only fresh confirmation of the repaired 070
checkpoint.

071 verifies whether the 070 `FINETUNE_068_TARGETED_REPAIR` checkpoint
generalizes to fresh hard-distractor, counterfactual, near-miss, pocket
suppression, negative-route, and long-context rows.

This is eval-only.

no training
no checkpoint repair
no open-ended assistant
no free-form generation
no perplexity
no full English LM
no language grounding
no production training
no GA
no public beta
no hosted SaaS
no service API change
no deployment harness change
no release docs change
no public crate export change
no root LICENSE change

## Runner And Checker

Runner:

```text
instnct-core/examples/phase_lane_repaired_checkpoint_capability_confirm.rs
```

Static checker:

```text
scripts/probes/run_stable_loop_phase_lock_071_repaired_checkpoint_capability_confirm_check.py
```

071 may add only the runner, checker, and 071 research docs.

## Required Inputs

```text
target/pilot_wave/stable_loop_phase_lock_070_distractor_resistant_anchorroute_training/smoke/checkpoints/finetune_068_targeted_repair/model_checkpoint.json
target/pilot_wave/stable_loop_phase_lock_070_distractor_resistant_anchorroute_training/smoke/summary.json
target/pilot_wave/stable_loop_phase_lock_070_distractor_resistant_anchorroute_training/smoke/checkpoint_manifest.json
target/pilot_wave/stable_loop_phase_lock_070_distractor_resistant_anchorroute_training/smoke/arm_comparison.json
target/pilot_wave/stable_loop_phase_lock_069_model_capability_benchmark_gate/smoke
```

If any required upstream artifact is missing, emit
`UPSTREAM_070_ARTIFACT_MISSING`. Do not rerun 069 or 070. Do not train or
repair a replacement checkpoint.

## Eval-Only Hard Wall

The runner must record:

```text
train_step_count = 0
checkpoint_hash_before
checkpoint_hash_after
checkpoint_hash_unchanged = true
prediction_oracle_used = false
```

Failure verdicts:

```text
TRAINING_SIDE_EFFECT_DETECTED
CHECKPOINT_MUTATION_DETECTED
ORACLE_SHORTCUT_DETECTED
```

## Benchmark Families

```text
FRESH_CONTEXT_ENTITY_EXTRACTION
FRESH_COUNTERFACTUAL_BINDING
FRESH_DISTRACTOR_RESISTANCE
FRESH_LONG_CONTEXT_NEEDLE_BINDING
FRESH_NEAR_MISS_ANCHOR_SELECTION
FRESH_IRRELEVANT_POCKET_SUPPRESSION
FRESH_NEGATIVE_ROUTE_REJECTION
RETENTION_INSTRUCTION_FOLLOWING_CLOSED
RETENTION_MULTI_HOP_KEY_VALUE_BINDING
RETENTION_SYMBOLIC_RULE_CLOSED_CHOICE
RETENTION_NON_ROUTE_TEXT_CONTROL
OPEN_ENDED_INTERFACE_LIMITATION
```

Rows must use fresh templates, new key/value schedules, reordered distractors,
longer context variants, and near-miss anchor variants.

Record:

```text
overlap_with_069_samples_count
overlap_with_070_samples_count
upstream_exact_overlap_audit_limited
```

Fail with `BENCHMARK_LEAKAGE_DETECTED` if obvious exact overlap is detected.

## Baselines And Controls

Model predictions, baselines, and `NO_ROUTE_FEATURE_CONTROL` must evaluate the
same supported rows.

Record:

```text
eval_row_hash_model
eval_row_hash_baselines
eval_row_hash_no_route_control
baseline_eval_mismatch = false
delta_vs_no_route_control
```

Failure verdicts:

```text
BASELINE_EVAL_MISMATCH
NO_ROUTE_CONTROL_MISSING
```

## Required Artifacts

Generated artifacts are written only under `target/`:

```text
queue.json
progress.jsonl
benchmark_config.json
upstream_070_manifest.json
checkpoint_manifest.json
capability_dataset_manifest.json
benchmark_examples_sample.jsonl
baseline_metrics.json
no_route_feature_control_metrics.json
capability_metrics.json
per_family_metrics.json
limitation_report.json
human_readable_samples.jsonl
failure_case_samples.jsonl
collapse_metrics.json
summary.json
report.md
```

`progress.jsonl`, `summary.json`, and `report.md` are written from the start so
071 is not a black-box run.

## Gates

```text
fresh_context_entity_extraction_accuracy >= 0.85
fresh_counterfactual_binding_accuracy >= 0.85
fresh_distractor_resistance_accuracy >= 0.85
fresh_long_context_needle_accuracy >= 0.75
fresh_near_miss_anchor_selection_accuracy >= 0.80
fresh_irrelevant_pocket_suppression_accuracy >= 0.80
fresh_negative_route_rejection_accuracy >= 0.75
retention_instruction_following_closed_accuracy >= 0.90
retention_multi_hop_key_value_accuracy >= 0.75
retention_symbolic_rule_closed_choice_accuracy >= 0.85
retention_non_route_text_control_accuracy >= 0.90
family_min_accuracy >= 0.70
supported_accuracy >= 0.85
delta_vs_majority > 0.10
delta_vs_copy_first_match > 0.10
top_output_rate <= 0.45
space_output_rate <= 0.02
empty_output_rate <= 0.02
collapse_detected = false
```

## Limitation Honesty

The runner and docs must state:

```text
open_ended_generation_supported = false
free_form_answering_supported = false
perplexity_supported = false
finite_label_surface = true
closed-label success does not imply language grounding
this is not an open-ended assistant
```

Failure verdicts:

```text
OPEN_ENDED_CLAIM_DETECTED
PERPLEXITY_CLAIM_DETECTED
```

## Validation Commands

```powershell
cargo check -p instnct-core --example phase_lane_repaired_checkpoint_capability_confirm
cargo run -p instnct-core --example phase_lane_repaired_checkpoint_capability_confirm -- --out target/pilot_wave/stable_loop_phase_lock_071_repaired_checkpoint_capability_confirm/smoke --checkpoint target/pilot_wave/stable_loop_phase_lock_070_distractor_resistant_anchorroute_training/smoke/checkpoints/finetune_068_targeted_repair/model_checkpoint.json --upstream-070-root target/pilot_wave/stable_loop_phase_lock_070_distractor_resistant_anchorroute_training/smoke --benchmark-069-root target/pilot_wave/stable_loop_phase_lock_069_model_capability_benchmark_gate/smoke --seed 2027 --heartbeat-sec 20
python -m py_compile scripts/probes/run_stable_loop_phase_lock_071_repaired_checkpoint_capability_confirm_check.py
python scripts/probes/run_stable_loop_phase_lock_071_repaired_checkpoint_capability_confirm_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_070_distractor_resistant_anchorroute_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_069_model_capability_benchmark_check.py --check-only
cargo test -p instnct-core sdk_candidate
git diff --check
```

## Verdicts

Positive verdicts include:

```text
REPAIRED_CHECKPOINT_CAPABILITY_CONFIRM_POSITIVE
UPSTREAM_070_CHECKPOINT_VERIFIED
NO_TRAINING_PERFORMED
CHECKPOINT_UNCHANGED
FRESH_HARD_DISTRACTOR_GENERALIZATION_PASSES
FRESH_COUNTERFACTUAL_GENERALIZATION_PASSES
FRESH_LONG_CONTEXT_GENERALIZATION_PASSES
RETENTION_CONFIRM_PASSES
NO_ROUTE_CONTROL_RECORDED
BASELINE_COMPARISON_RECORDED
HUMAN_READABLE_SAMPLES_WRITTEN
OPEN_ENDED_LIMITATION_RECORDED
PRODUCTION_TRAINING_NOT_CLAIMED
```

Failure verdicts include:

```text
REPAIRED_CHECKPOINT_CAPABILITY_CONFIRM_FAILS
UPSTREAM_070_ARTIFACT_MISSING
CHECKPOINT_MUTATION_DETECTED
TRAINING_SIDE_EFFECT_DETECTED
ORACLE_SHORTCUT_DETECTED
BENCHMARK_LEAKAGE_DETECTED
BASELINE_EVAL_MISMATCH
NO_ROUTE_CONTROL_MISSING
FRESH_HARD_DISTRACTOR_GENERALIZATION_FAILS
FRESH_COUNTERFACTUAL_GENERALIZATION_FAILS
FRESH_LONG_CONTEXT_GENERALIZATION_FAILS
RETENTION_CONFIRM_FAILS
CAPABILITY_FAMILY_GATE_FAILS
STATIC_OUTPUT_COLLAPSE_DETECTED
HUMAN_SAMPLE_REPORT_MISSING
OPEN_ENDED_CLAIM_DETECTED
PERPLEXITY_CLAIM_DETECTED
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
```

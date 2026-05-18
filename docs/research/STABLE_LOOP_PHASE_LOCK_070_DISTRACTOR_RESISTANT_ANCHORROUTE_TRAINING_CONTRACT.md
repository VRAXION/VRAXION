# STABLE_LOOP_PHASE_LOCK_070_DISTRACTOR_RESISTANT_ANCHORROUTE_TRAINING Contract

Status: contract for targeted hard-distractor AnchorRoute repair training.

070 targets the weak 069 capability families:

```text
CONTEXT_ENTITY_EXTRACTION
COUNTERFACTUAL_BINDING
DISTRACTOR_RESISTANCE
LONG_CONTEXT_NEEDLE_BINDING
```

This milestone is finite-label repair training only.

no open-ended assistant
no free-form generation
no perplexity
no full English LM
no language grounding
no production training
no production-scale training
no GA
no public beta
no hosted SaaS
no clinical use
no high-stakes education use
no service API change
no deployment harness change
no public crate export change
no root LICENSE change

## Runner

```text
instnct-core/examples/phase_lane_distractor_resistant_anchorroute_training.rs
```

The runner compares:

```text
FRESH_TARGETED_MIX_TRAINING
FINETUNE_068_TARGETED_REPAIR
```

The warm-start arm loads the upstream 068 checkpoint read-only. 070 records the
checkpoint hash before and after the run and fails with
`CHECKPOINT_MUTATION_DETECTED` if the upstream file changes.

## Required Inputs

```text
target/pilot_wave/stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale/confirm/seed_2028/checkpoints/MIXED_WITH_ROUTE_GRAMMAR_ON/model_checkpoint.json
target/pilot_wave/stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale/confirm/summary.json
target/pilot_wave/stable_loop_phase_lock_069_model_capability_benchmark_gate/smoke
```

If required upstream artifacts are missing, 070 emits
`UPSTREAM_068_ARTIFACT_MISSING`. It does not rerun 067, 068, or 069 and does not
train a replacement checkpoint.

## Required Arms

```text
NO_TRAIN_069_BASELINE
FRESH_TARGETED_MIX_TRAINING
FINETUNE_068_TARGETED_REPAIR
FRESH_SHUFFLED_LABEL_CONTROL
FINETUNE_SHUFFLED_LABEL_CONTROL
NO_ROUTE_FEATURE_CONTROL
CHECKPOINT_RELOAD_EVAL
ROLLBACK_REHEARSAL
RESUME_FROM_CHECKPOINT
```

## Required Training Families

```text
HARD_DISTRACTOR_ANCHOR_BINDING
NEAR_MISS_ANCHOR_SELECTION
SAME_KEY_DIFFERENT_CONTEXT
LONG_CONTEXT_NEEDLE_RETRIEVAL
IRRELEVANT_POCKET_SUPPRESSION
NEGATIVE_ROUTE_REJECTION
ANSWER_ONLY_HARD_BINDING
TRACE_MIXED_HARD_BINDING
RETENTION_FINEWEB_CONTINUATION
RETENTION_NON_ROUTE_CONTROL
```

## Integrity Rules

The runner must record:

```text
warm_start_hash
initialized_hash
final_hash
prediction_oracle_used = false
overlap_with_069_eval_count
best_arm
fresh_pass
finetune_pass
fresh_vs_finetune_delta
recommended_next_strategy
```

Hard fail conditions:

```text
CHECKPOINT_MUTATION_DETECTED
NO_ACTUAL_TRAINING_UPDATE_DETECTED
TRAIN_BENCHMARK_LEAKAGE_DETECTED
BASELINE_EVAL_MISMATCH
RETENTION_REGRESSION_DETECTED
ARM_COMPARISON_MISSING
OPEN_ENDED_CLAIM_DETECTED
FAILURE_CASE_REPORT_MISSING
STATIC_OUTPUT_COLLAPSE_DETECTED
CHECKPOINT_RELOAD_FAILS
ROLLBACK_REHEARSAL_FAILS
RESUME_FROM_CHECKPOINT_FAILS
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
```

Training data must not contain exact 069 benchmark evaluation rows.
`overlap_with_069_eval_count` must be zero.

## Required Artifacts

Generated artifacts are written only under `target/`:

```text
queue.json
progress.jsonl
training_config.json
upstream_068_manifest.json
baseline_069_reference.json
targeted_dataset_manifest.json
train_examples_sample.jsonl
heldout_examples_sample.jsonl
ood_examples_sample.jsonl
training_metrics.jsonl
checkpoint_manifest.json
checkpoint_hashes.json
post_training_capability_metrics.json
per_family_metrics.json
retention_metrics.json
regression_report.json
baseline_knockout_report.json
arm_comparison.json
human_readable_samples.jsonl
failure_case_samples.jsonl
collapse_metrics.json
summary.json
report.md
```

`progress.jsonl`, `summary.json`, and `report.md` are written from the start of
the run and updated after major phases. This prevents black-box runs.

## Positive Gates

```text
context_entity_extraction_accuracy >= 0.85
counterfactual_binding_accuracy >= 0.85
distractor_resistance_accuracy >= 0.80
long_context_needle_accuracy >= 0.65
family_min_accuracy >= 0.70
delta_vs_majority > 0.10
delta_vs_copy_first_match > 0.10
top_output_rate <= 0.45
space_output_rate <= 0.02
empty_output_rate <= 0.02
collapse_detected = false
```

Fine-tune retention must reject more than 0.05 absolute regression on:

```text
INSTRUCTION_FOLLOWING_CLOSED
MULTI_HOP_KEY_VALUE_BINDING
SYMBOLIC_RULE_CLOSED_CHOICE
NON_ROUTE_TEXT_CONTROL
```

## Verdicts

Positive verdicts include:

```text
DISTRACTOR_RESISTANT_ANCHORROUTE_TRAINING_POSITIVE
FRESH_TARGETED_MIX_TRAINING_COMPLETED
FINETUNE_068_TARGETED_REPAIR_COMPLETED
DISTRACTOR_RESISTANCE_IMPROVED
CONTEXT_ENTITY_EXTRACTION_IMPROVED
COUNTERFACTUAL_BINDING_IMPROVED
LONG_CONTEXT_NEEDLE_IMPROVED
RETENTION_GATE_PASSES
RETENTION_REGRESSION_REJECTED
ARM_COMPARISON_WRITTEN
BEST_ARM_SELECTED
TRAIN_BENCHMARK_LEAKAGE_REJECTED
UPSTREAM_068_CHECKPOINT_UNCHANGED
BASELINE_KNOCKOUT_STABLE
CHECKPOINT_PIPELINE_STRICT_PASS
ORACLE_SHORTCUT_REJECTED
PRODUCTION_TRAINING_NOT_CLAIMED
```

Failure verdicts include:

```text
DISTRACTOR_RESISTANT_ANCHORROUTE_TRAINING_FAILS
UPSTREAM_068_ARTIFACT_MISSING
FULL_CORPUS_TRAINING_ATTEMPTED
TARGETED_SCALE_LIMIT_EXCEEDED
CHECKPOINT_MUTATION_DETECTED
NO_ACTUAL_TRAINING_UPDATE_DETECTED
ORACLE_SHORTCUT_DETECTED
BASELINE_EVAL_MISMATCH
TRAIN_BENCHMARK_LEAKAGE_DETECTED
DISTRACTOR_RESISTANCE_STILL_FAILS
CONTEXT_ENTITY_EXTRACTION_STILL_FAILS
COUNTERFACTUAL_BINDING_STILL_FAILS
LONG_CONTEXT_NEEDLE_STILL_FAILS
RETENTION_REGRESSION_DETECTED
ARM_COMPARISON_MISSING
OPEN_ENDED_CLAIM_DETECTED
FAILURE_CASE_REPORT_MISSING
STATIC_OUTPUT_COLLAPSE_DETECTED
CHECKPOINT_RELOAD_FAILS
ROLLBACK_REHEARSAL_FAILS
RESUME_FROM_CHECKPOINT_FAILS
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
```

## Validation Commands

```powershell
cargo check -p instnct-core --example phase_lane_distractor_resistant_anchorroute_training
cargo run -p instnct-core --example phase_lane_distractor_resistant_anchorroute_training -- --out target/pilot_wave/stable_loop_phase_lock_070_distractor_resistant_anchorroute_training/smoke --upstream-checkpoint target/pilot_wave/stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale/confirm/seed_2028/checkpoints/MIXED_WITH_ROUTE_GRAMMAR_ON/model_checkpoint.json --upstream-summary target/pilot_wave/stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale/confirm/summary.json --benchmark-069-root target/pilot_wave/stable_loop_phase_lock_069_model_capability_benchmark_gate/smoke --targeted-examples 120000 --seed 2026 --heartbeat-sec 20
python -m py_compile scripts/probes/run_stable_loop_phase_lock_070_distractor_resistant_anchorroute_check.py
python scripts/probes/run_stable_loop_phase_lock_070_distractor_resistant_anchorroute_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_069_model_capability_benchmark_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale_check.py --check-only
cargo test -p instnct-core sdk_candidate
git diff --check
```

# STABLE_LOOP_PHASE_LOCK_073_SCENARIO_GATED_REPAIR_FRESH_CONFIRM Contract

Status: contract for eval-only fresh confirmation of the 072 scenario-gated
repair checkpoint.

073 verifies whether the 072 `SCENARIO_GATED_SIDEPACKET_REPAIR` checkpoint
generalizes to fresh, non-trained scenario, pocket, counterfactual,
answer-only, and retention rows.

This is finite-label scenario-state confirmation only.

no training
no checkpoint repair
no checkpoint mutation
no upstream rerun
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
instnct-core/examples/phase_lane_scenario_gated_repair_fresh_confirm.rs
```

Static checker:

```text
scripts/probes/run_stable_loop_phase_lock_073_scenario_gated_repair_fresh_confirm_check.py
```

073 may add only the eval runner, static checker, and 073 research docs.
Generated outputs are written only under `target/`.

## Required Upstream Artifacts

Default checkpoint:

```text
target/pilot_wave/stable_loop_phase_lock_072_counterfactual_scenario_binding_repair/smoke/checkpoints/scenario_gated_sidepacket_repair/model_checkpoint.json
```

Required upstream artifacts:

```text
072 checkpoint
072 summary.json
072 checkpoint_manifest.json
072 arm_comparison.json
072 targeted_dataset_manifest.json
071B analysis root
071 confirm root
070 repair root
```

If any required artifact is missing, emit:

```text
UPSTREAM_072_ARTIFACT_MISSING
```

Do not rerun 070/071/071B/072 and do not train or repair a replacement
checkpoint.

## Eval-Only Hard Wall

Required fields:

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

## Freshness And Same-Row Controls

Fresh rows must not copy upstream rows:

```text
overlap_with_070_eval_count = 0
overlap_with_071_eval_count = 0
overlap_with_071b_failure_digest_count = 0
overlap_with_072_train_count = 0
overlap_with_072_eval_count = 0
```

Any exact overlap fails with:

```text
BENCHMARK_LEAKAGE_DETECTED
FRESH_EVAL_LEAKAGE_DETECTED
```

Same-row guarantee:

```text
eval_row_hash_model
eval_row_hash_baselines
eval_row_hash_no_route_control
eval_row_hash_ungated_control
eval_row_hash_shuffled_control
baseline_eval_mismatch = false
```

Failure verdict:

```text
BASELINE_EVAL_MISMATCH
```

## Evaluation Families And Controls

Fresh benchmark families:

```text
FRESH_ACTIVE_SCENARIO_BINDING
FRESH_COUNTERFACTUAL_SCENARIO_SWITCH
FRESH_DISTRACTOR_SCENARIO_REJECTION
FRESH_OLD_SCENARIO_SUPPRESSION
FRESH_INACTIVE_POCKET_SUPPRESSION
FRESH_STALE_POCKET_SUPPRESSION
FRESH_FIRST_LEDGER_BIAS_SUPPRESSION
FRESH_SIDE_NOTE_SUPPRESSION
FRESH_ANSWER_ONLY_SCENARIO_BINDING
FRESH_TRACE_MIXED_SCENARIO_BINDING
RETENTION_INSTRUCTION_FOLLOWING_CLOSED
RETENTION_MULTI_HOP_KEY_VALUE_BINDING
RETENTION_SYMBOLIC_RULE_CLOSED_CHOICE
RETENTION_NON_ROUTE_TEXT_CONTROL
OPEN_ENDED_INTERFACE_LIMITATION
```

Controls and baselines:

```text
MAJORITY_LABEL
COPY_FIRST_MATCH
COPY_LAST_TOKEN
NO_ROUTE_FEATURE_CONTROL
UNGATED_SIDEPACKET_SIMULATED_CONTROL
SHUFFLED_SCENARIO_LABEL_CONTROL
```

`SHUFFLED_SCENARIO_LABEL_CONTROL` must fail or stay clearly below threshold:

```text
SHUFFLED_SCENARIO_CONTROL_UNEXPECTED_PASS
```

## Required Attribution

Scenario source attribution must include:

```text
active_scenario_selection_accuracy
distractor_scenario_selection_rate
old_scenario_selection_rate
inactive_pocket_selection_rate
stale_pocket_selection_rate
first_ledger_bias_rate
side_note_leak_rate
```

Failure verdict:

```text
SCENARIO_SOURCE_ATTRIBUTION_MISSING
```

## Positive Gates

```text
fresh_active_scenario_binding_accuracy >= 0.90
fresh_counterfactual_scenario_switch_accuracy >= 0.85
fresh_distractor_scenario_rejection_accuracy >= 0.90
fresh_old_scenario_suppression_accuracy >= 0.90
fresh_inactive_pocket_suppression_accuracy >= 0.85
fresh_stale_pocket_suppression_accuracy >= 0.85
fresh_first_ledger_bias_suppression_accuracy >= 0.85
fresh_side_note_suppression_accuracy >= 0.85
fresh_answer_only_scenario_binding_accuracy >= 0.85
family_min_accuracy >= 0.75
supported_accuracy >= 0.88
delta_vs_no_route_control > 0.10
delta_vs_ungated_sidepacket_control > 0.03
delta_vs_copy_first_match > 0.10
shuffled_scenario_label_control_accuracy < 0.70
retention families pass
top_output_rate <= 0.45
space_output_rate <= 0.02
empty_output_rate <= 0.02
collapse_detected = false
```

If gated passes but ungated is too close, emit:

```text
GATED_WRITEBACK_NOT_UNIQUELY_CONFIRMED_ON_FRESH_EVAL
```

Answer-only must pass. Trace-mixed success alone fails with:

```text
TRACE_DEPENDENCE_DETECTED
```

Retention failure verdict:

```text
RETENTION_CONFIRM_FAILS
```

## Required Artifacts

Generated artifacts are written only under:

```text
target/pilot_wave/stable_loop_phase_lock_073_scenario_gated_repair_fresh_confirm/
```

Artifact names:

```text
queue.json
progress.jsonl
benchmark_config.json
upstream_072_manifest.json
checkpoint_manifest.json
capability_dataset_manifest.json
benchmark_examples_sample.jsonl
baseline_metrics.json
no_route_feature_control_metrics.json
ungated_sidepacket_control_metrics.json
shuffled_scenario_control_metrics.json
capability_metrics.json
per_family_metrics.json
scenario_selection_metrics.json
pocket_suppression_metrics.json
retention_metrics.json
limitation_report.json
human_readable_samples.jsonl
failure_case_samples.jsonl
collapse_metrics.json
summary.json
report.md
```

`progress.jsonl`, `summary.json`, and `report.md` are written from the start.

`human_readable_samples.jsonl` must include:

```text
task_family
input
expected_output
model_output
baseline_outputs
no_route_output
ungated_control_output
shuffled_control_output
pass_fail
limitation_flag
```

`failure_case_samples.jsonl` must exist even if empty.

Failure verdict:

```text
HUMAN_SAMPLE_REPORT_MISSING
```

## Validation Commands

```powershell
cargo check -p instnct-core --example phase_lane_scenario_gated_repair_fresh_confirm
cargo run -p instnct-core --example phase_lane_scenario_gated_repair_fresh_confirm -- --out target/pilot_wave/stable_loop_phase_lock_073_scenario_gated_repair_fresh_confirm/smoke --checkpoint target/pilot_wave/stable_loop_phase_lock_072_counterfactual_scenario_binding_repair/smoke/checkpoints/scenario_gated_sidepacket_repair/model_checkpoint.json --upstream-072-root target/pilot_wave/stable_loop_phase_lock_072_counterfactual_scenario_binding_repair/smoke --upstream-071b-root target/pilot_wave/stable_loop_phase_lock_071b_repair_overfit_failure_analysis/smoke --upstream-071-root target/pilot_wave/stable_loop_phase_lock_071_repaired_checkpoint_capability_confirm/smoke --upstream-070-root target/pilot_wave/stable_loop_phase_lock_070_distractor_resistant_anchorroute_training/smoke --seed 2027 --heartbeat-sec 20
python -m py_compile scripts/probes/run_stable_loop_phase_lock_073_scenario_gated_repair_fresh_confirm_check.py
python scripts/probes/run_stable_loop_phase_lock_073_scenario_gated_repair_fresh_confirm_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_072_counterfactual_scenario_binding_repair_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_071b_repair_overfit_failure_analysis_check.py --check-only
git diff --check
```

## Verdicts

Positive verdicts:

```text
SCENARIO_GATED_REPAIR_FRESH_CONFIRM_POSITIVE
UPSTREAM_072_CHECKPOINT_VERIFIED
NO_TRAINING_PERFORMED
CHECKPOINT_UNCHANGED
FRESH_ACTIVE_SCENARIO_BINDING_PASSES
FRESH_COUNTERFACTUAL_GENERALIZATION_PASSES
FRESH_POCKET_SUPPRESSION_PASSES
FRESH_GATED_ADVANTAGE_CONFIRMED
SCENARIO_SOURCE_ATTRIBUTION_RECORDED
ANSWER_ONLY_FRESH_SCENARIO_BINDING_PASSES
FRESH_EVAL_LEAKAGE_REJECTED
NO_ROUTE_CONTROL_RECORDED
UNGATED_CONTROL_RECORDED
SHUFFLED_SCENARIO_CONTROL_FAILS
RETENTION_CONFIRM_PASSES
BASELINE_COMPARISON_RECORDED
HUMAN_READABLE_SAMPLES_WRITTEN
OPEN_ENDED_LIMITATION_RECORDED
PRODUCTION_TRAINING_NOT_CLAIMED
```

Failure verdicts:

```text
SCENARIO_GATED_REPAIR_FRESH_CONFIRM_FAILS
UPSTREAM_072_ARTIFACT_MISSING
CHECKPOINT_MUTATION_DETECTED
TRAINING_SIDE_EFFECT_DETECTED
ORACLE_SHORTCUT_DETECTED
BENCHMARK_LEAKAGE_DETECTED
FRESH_EVAL_LEAKAGE_DETECTED
BASELINE_EVAL_MISMATCH
NO_ROUTE_CONTROL_MISSING
UNGATED_CONTROL_MISSING
SHUFFLED_SCENARIO_CONTROL_UNEXPECTED_PASS
GATED_WRITEBACK_NOT_UNIQUELY_CONFIRMED_ON_FRESH_EVAL
SCENARIO_SOURCE_ATTRIBUTION_MISSING
FRESH_ACTIVE_SCENARIO_BINDING_FAILS
FRESH_COUNTERFACTUAL_GENERALIZATION_FAILS
FRESH_POCKET_SUPPRESSION_FAILS
TRACE_DEPENDENCE_DETECTED
RETENTION_CONFIRM_FAILS
CAPABILITY_FAMILY_GATE_FAILS
STATIC_OUTPUT_COLLAPSE_DETECTED
HUMAN_SAMPLE_REPORT_MISSING
OPEN_ENDED_CLAIM_DETECTED
PERPLEXITY_CLAIM_DETECTED
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
```

073 success means fresh bounded scenario-state confirmation, not general
intelligence or production readiness.

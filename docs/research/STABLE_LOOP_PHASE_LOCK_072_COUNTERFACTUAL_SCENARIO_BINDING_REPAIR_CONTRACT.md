# STABLE_LOOP_PHASE_LOCK_072_COUNTERFACTUAL_SCENARIO_BINDING_REPAIR Contract

Status: contract for bounded finite-label scenario-state repair training.

072 targets the 071B failure mode where the model selected
`distractor_scenario_value`, `old_scenario_value`, `inactive_pocket_value`, or
`stale_pocket_value` instead of the active scenario value.

The hypothesis is protected highway plus gated sidepocket:

```text
base route stays protected
old/active/distractor sidepockets stay represented
only active scenario sidepacket may write back to readout
inactive/stale/distractor pockets must remain non-winning
```

This hypothesis must be tested against standard targeted repair, ungated
sidepacket, no-route, and shuffled-scenario controls. It is not assumed true.

This is finite-label scenario-state repair only.

no open-ended assistant
no free-form generation
no perplexity
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
instnct-core/examples/phase_lane_counterfactual_scenario_binding_repair.rs
```

Static checker:

```text
scripts/probes/run_stable_loop_phase_lock_072_counterfactual_scenario_binding_repair_check.py
```

072 may add only the bounded Rust runner, the 072 checker, and these 072
research docs. Generated checkpoints and artifacts are written only under
`target/`.

## Upstream Inputs

Read-only upstream warm start:

```text
target/pilot_wave/stable_loop_phase_lock_070_distractor_resistant_anchorroute_training/smoke/checkpoints/finetune_068_targeted_repair/model_checkpoint.json
```

Required 071B analysis root:

```text
target/pilot_wave/stable_loop_phase_lock_071b_repair_overfit_failure_analysis/smoke
```

The upstream checkpoint hash is recorded before and after 072. If it changes,
fail with:

```text
CHECKPOINT_MUTATION_DETECTED
```

## Required Arms

```text
NO_TRAIN_071_BASELINE
STANDARD_TARGETED_REPAIR_BASELINE
SCENARIO_GATED_SIDEPACKET_REPAIR
UNGATED_SIDEPACKET_REPAIR_CONTROL
NO_ROUTE_FEATURE_CONTROL
SHUFFLED_SCENARIO_LABEL_CONTROL
CHECKPOINT_RELOAD_EVAL
ROLLBACK_REHEARSAL
RESUME_FROM_CHECKPOINT
```

## Required Curriculum Families

```text
ACTIVE_SCENARIO_MARKER_BINDING
SAME_KEY_DIFFERENT_SCENARIO_SWITCH
DISTRACTOR_SCENARIO_REJECTION
STALE_SCENARIO_SUPPRESSION
INACTIVE_POCKET_NEGATIVE_ROUTE
FIRST_LEDGER_BIAS_SUPPRESSION
SIDE_NOTE_SUPPRESSION
ANSWER_ONLY_SCENARIO_BINDING
TRACE_MIXED_SCENARIO_BINDING
RETENTION_INSTRUCTION_FOLLOWING_CLOSED
RETENTION_MULTI_HOP_KEY_VALUE_BINDING
RETENTION_SYMBOLIC_RULE_CLOSED_CHOICE
RETENTION_NON_ROUTE_TEXT_CONTROL
```

Trace fields:

```text
scenario:active
scenario:old
scenario:distractor
pocket:active
pocket:inactive
writeback:active_only
```

The runner separately reports:

```text
trace_mixed_accuracy
answer_only_accuracy
answer_only_active_scenario_accuracy
```

## Positive Gates

```text
fresh_counterfactual_binding_accuracy >= 0.85
active_scenario_selection_accuracy >= 0.90
distractor_scenario_selection_rate <= 0.10
old_scenario_selection_rate <= 0.10
fresh_context_entity_extraction_accuracy >= 0.85
fresh_irrelevant_pocket_suppression_accuracy >= 0.80
answer_only_active_scenario_accuracy >= 0.85
family_min_accuracy >= 0.70
collapse_detected = false
```

Gated writeback comparison:

```text
delta_vs_standard_targeted > 0.03
delta_vs_ungated_sidepacket > 0.03
delta_vs_no_route > 0.10
```

If `SCENARIO_GATED_SIDEPACKET_REPAIR` passes but ungated is within `0.03`,
emit:

```text
GATED_WRITEBACK_NOT_UNIQUELY_PROVEN
```

and do not claim gated uniqueness.

Protected route metrics:

```text
base_route_retention_accuracy
protected_highway_regression
sidepocket_writeback_accuracy
inactive_sidepocket_not_readout_rate
```

Fail if protected route regresses by more than `0.05` absolute:

```text
PROTECTED_HIGHWAY_REGRESSION_DETECTED
```

Retention gates require no regression greater than `0.05` absolute on:

```text
RETENTION_INSTRUCTION_FOLLOWING_CLOSED
RETENTION_MULTI_HOP_KEY_VALUE_BINDING
RETENTION_SYMBOLIC_RULE_CLOSED_CHOICE
RETENTION_NON_ROUTE_TEXT_CONTROL
```

Failure verdict:

```text
RETENTION_REGRESSION_DETECTED
```

## Leakage And Control Gates

```text
overlap_with_071_eval_count = 0
overlap_with_071b_failure_digest_count = 0
overlap_with_070_eval_count = 0
baseline_eval_mismatch = false
prediction_oracle_used = false
```

Training rows must not overlap with 071 eval rows, 071B failure digest rows, or
070 eval rows:

```text
TRAIN_BENCHMARK_LEAKAGE_DETECTED
```

All arms and controls must evaluate exact same rows:

```text
BASELINE_EVAL_MISMATCH
```

`SHUFFLED_SCENARIO_LABEL_CONTROL` must fail or remain clearly below pass
threshold:

```text
SHUFFLED_SCENARIO_CONTROL_UNEXPECTED_PASS
```

Answer-only generalization is required:

```text
TRACE_DEPENDENCE_DETECTED
```

## Required Artifacts

Generated artifacts are written only under:

```text
target/pilot_wave/stable_loop_phase_lock_072_counterfactual_scenario_binding_repair/
```

Required artifact names:

```text
queue.json
progress.jsonl
training_config.json
upstream_checkpoint_manifest.json
upstream_071b_manifest.json
targeted_dataset_manifest.json
train_examples_sample.jsonl
eval_examples_sample.jsonl
training_metrics.jsonl
checkpoint_manifest.json
checkpoint_hashes.json
arm_comparison.json
scenario_selection_metrics.json
pocket_writeback_metrics.json
protected_highway_metrics.json
retention_metrics.json
wrong_answer_source_after_repair.json
baseline_knockout_report.json
failure_case_samples.jsonl
collapse_metrics.json
summary.json
report.md
```

`progress.jsonl`, `summary.json`, and `report.md` are written from the start, so
072 is not a black-box run.

`failure_case_samples.jsonl` must exist even on pass. Failed rows include:

```text
arm
task_family
input
expected
predicted
wrong_answer_source
scenario_state
pocket_state
short_diagnosis
```

## Validation Commands

```powershell
cargo check -p instnct-core --example phase_lane_counterfactual_scenario_binding_repair
cargo run -p instnct-core --example phase_lane_counterfactual_scenario_binding_repair -- --out target/pilot_wave/stable_loop_phase_lock_072_counterfactual_scenario_binding_repair/smoke --upstream-checkpoint target/pilot_wave/stable_loop_phase_lock_070_distractor_resistant_anchorroute_training/smoke/checkpoints/finetune_068_targeted_repair/model_checkpoint.json --upstream-071b-root target/pilot_wave/stable_loop_phase_lock_071b_repair_overfit_failure_analysis/smoke --targeted-examples 120000 --seed 2026 --heartbeat-sec 20
python -m py_compile scripts/probes/run_stable_loop_phase_lock_072_counterfactual_scenario_binding_repair_check.py
python scripts/probes/run_stable_loop_phase_lock_072_counterfactual_scenario_binding_repair_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_071b_repair_overfit_failure_analysis_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_071_repaired_checkpoint_capability_confirm_check.py --check-only
git diff --check
```

## Verdicts

Positive verdicts:

```text
COUNTERFACTUAL_SCENARIO_BINDING_REPAIR_POSITIVE
SCENARIO_GATED_SIDEPACKET_REPAIR_COMPLETED
ACTIVE_SCENARIO_SELECTION_IMPROVED
DISTRACTOR_SCENARIO_REJECTED
STALE_SCENARIO_SUPPRESSED
INACTIVE_POCKET_SUPPRESSED
GATED_WRITEBACK_ADVANTAGE_SHOWN
PROTECTED_HIGHWAY_PRESERVED
ANSWER_ONLY_SCENARIO_BINDING_PASSES
SHUFFLED_SCENARIO_CONTROL_FAILS
TRAIN_BENCHMARK_LEAKAGE_REJECTED
RETENTION_GATE_PASSES
BEST_ARM_SELECTED
UPSTREAM_CHECKPOINT_UNCHANGED
ORACLE_SHORTCUT_REJECTED
PRODUCTION_TRAINING_NOT_CLAIMED
```

Failure verdicts:

```text
COUNTERFACTUAL_SCENARIO_BINDING_REPAIR_FAILS
ACTIVE_SCENARIO_SELECTION_STILL_FAILS
DISTRACTOR_SCENARIO_STILL_SELECTED
STALE_SCENARIO_STILL_SELECTED
INACTIVE_POCKET_STILL_SELECTED
GATED_WRITEBACK_NOT_UNIQUELY_PROVEN
PROTECTED_HIGHWAY_REGRESSION_DETECTED
TRAIN_BENCHMARK_LEAKAGE_DETECTED
BASELINE_EVAL_MISMATCH
SHUFFLED_SCENARIO_CONTROL_UNEXPECTED_PASS
TRACE_DEPENDENCE_DETECTED
RETENTION_REGRESSION_DETECTED
FAILURE_CASE_REPORT_MISSING
CHECKPOINT_MUTATION_DETECTED
ORACLE_SHORTCUT_DETECTED
OPEN_ENDED_CLAIM_DETECTED
RUNTIME_SURFACE_MUTATION_DETECTED
ROOT_LICENSE_CHANGED
```

072 success means active scenario readout wins while old, distractor, inactive,
and stale pockets can remain represented without becoming winning readout.

072 does not mean open-ended assistant capability.
no full English LM capability
no production training
no language grounding
no GA
no public beta
no hosted SaaS

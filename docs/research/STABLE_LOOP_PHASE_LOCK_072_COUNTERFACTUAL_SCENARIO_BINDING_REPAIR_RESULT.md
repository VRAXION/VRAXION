# STABLE_LOOP_PHASE_LOCK_072_COUNTERFACTUAL_SCENARIO_BINDING_REPAIR Result

Status: implementation result document for bounded finite-label scenario-state
repair training.

072 implements the `SCENARIO_GATED_SIDEPACKET_REPAIR` hypothesis from the 071B
failure attribution: active scenario state should write back to the finite-label
readout, while old, distractor, inactive, and stale sidepockets remain
represented but non-winning.

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

## Implementation

Runner:

```text
instnct-core/examples/phase_lane_counterfactual_scenario_binding_repair.rs
```

Static checker:

```text
scripts/probes/run_stable_loop_phase_lock_072_counterfactual_scenario_binding_repair_check.py
```

The runner requires the 070 `FINETUNE_068_TARGETED_REPAIR` checkpoint as a
read-only warm start and the 071B analysis root as upstream input. It writes
generated checkpoints and reports only below:

```text
target/pilot_wave/stable_loop_phase_lock_072_counterfactual_scenario_binding_repair/
```

It records upstream checkpoint hash before and after the run. If the upstream
checkpoint changes, the run fails with:

```text
CHECKPOINT_MUTATION_DETECTED
```

## Arms Compared

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

The report must state:

```text
best_arm
delta_vs_standard_targeted
delta_vs_ungated_sidepacket
delta_vs_no_route
```

If the gated sidepacket arm passes but ungated sidepacket is within `0.03`,
the result must emit:

```text
GATED_WRITEBACK_NOT_UNIQUELY_PROVEN
```

and avoid claiming unique gated-writeback proof.

## Curriculum

Training families:

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

Separate accuracy fields:

```text
trace_mixed_accuracy
answer_only_accuracy
answer_only_active_scenario_accuracy
```

## Required Metrics

Scenario source attribution after repair:

```text
active_scenario_selection_accuracy
distractor_scenario_selection_rate
old_scenario_selection_rate
inactive_pocket_selection_rate
stale_pocket_selection_rate
first_ledger_bias_rate
side_note_leak_rate
```

Protected highway metrics:

```text
base_route_retention_accuracy
protected_highway_regression
sidepocket_writeback_accuracy
inactive_sidepocket_not_readout_rate
```

Positive gates:

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
delta_vs_standard_targeted > 0.03
delta_vs_ungated_sidepacket > 0.03
delta_vs_no_route > 0.10
```

Retention gates:

```text
RETENTION_INSTRUCTION_FOLLOWING_CLOSED regression <= 0.05
RETENTION_MULTI_HOP_KEY_VALUE_BINDING regression <= 0.05
RETENTION_SYMBOLIC_RULE_CLOSED_CHOICE regression <= 0.05
RETENTION_NON_ROUTE_TEXT_CONTROL regression <= 0.05
```

Failure verdict:

```text
RETENTION_REGRESSION_DETECTED
```

## Leakage And Controls

The runner records:

```text
overlap_with_071_eval_count
overlap_with_071b_failure_digest_count
overlap_with_070_eval_count
baseline_eval_mismatch
prediction_oracle_used
```

Failure verdicts:

```text
TRAIN_BENCHMARK_LEAKAGE_DETECTED
BASELINE_EVAL_MISMATCH
ORACLE_SHORTCUT_DETECTED
```

`SHUFFLED_SCENARIO_LABEL_CONTROL` must fail or stay clearly below pass
threshold:

```text
SHUFFLED_SCENARIO_CONTROL_UNEXPECTED_PASS
```

Answer-only scenario binding must pass:

```text
TRACE_DEPENDENCE_DETECTED
```

## Required Artifacts

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

`progress.jsonl`, `summary.json`, and `report.md` are written from the start.

`failure_case_samples.jsonl` must exist even on pass. Failure rows include:

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

Failure verdict:

```text
FAILURE_CASE_REPORT_MISSING
```

## Required Commands

```powershell
cargo check -p instnct-core --example phase_lane_counterfactual_scenario_binding_repair
cargo run -p instnct-core --example phase_lane_counterfactual_scenario_binding_repair -- --out target/pilot_wave/stable_loop_phase_lock_072_counterfactual_scenario_binding_repair/smoke --upstream-checkpoint target/pilot_wave/stable_loop_phase_lock_070_distractor_resistant_anchorroute_training/smoke/checkpoints/finetune_068_targeted_repair/model_checkpoint.json --upstream-071b-root target/pilot_wave/stable_loop_phase_lock_071b_repair_overfit_failure_analysis/smoke --targeted-examples 120000 --seed 2026 --heartbeat-sec 20
python -m py_compile scripts/probes/run_stable_loop_phase_lock_072_counterfactual_scenario_binding_repair_check.py
python scripts/probes/run_stable_loop_phase_lock_072_counterfactual_scenario_binding_repair_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_071b_repair_overfit_failure_analysis_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_071_repaired_checkpoint_capability_confirm_check.py --check-only
git diff --check
```

## Observed Smoke Result

The 072 smoke run completed with:

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

Observed metrics:

```text
best_arm = SCENARIO_GATED_SIDEPACKET_REPAIR
supported_accuracy = 0.9923076923076923
fresh_counterfactual_binding_accuracy = 0.9833333333333333
active_scenario_selection_accuracy = 0.9833333333333333
distractor_scenario_selection_rate = 0.0
old_scenario_selection_rate = 0.0
fresh_context_entity_extraction_accuracy = 1.0
fresh_irrelevant_pocket_suppression_accuracy = 1.0
answer_only_active_scenario_accuracy = 0.975
trace_mixed_accuracy = 1.0
family_min_accuracy = 0.95
delta_vs_standard_targeted = 0.20000000000000007
delta_vs_ungated_sidepacket = 0.13076923076923075
delta_vs_no_route = 0.7615384615384615
base_route_retention_accuracy = 1.0
protected_highway_regression = 0.0
inactive_sidepocket_not_readout_rate = 1.0
collapse_detected = false
```

Integrity checks:

```text
upstream checkpoint unchanged = true
overlap_with_071_eval_count = 0
overlap_with_071b_failure_digest_count = 0
overlap_with_070_eval_count = 0
baseline_eval_mismatch = false
prediction_oracle_used = false
checkpoint_reload_pass = true
rollback_success = true
resume_from_checkpoint_pass = true
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

Observed smoke metrics are recorded in generated `summary.json` and
`report.md`. A positive 072 result means only bounded scenario-state repair on
fresh finite-label rows. It does not mean open-ended assistant capability.
no full English LM capability
no production training
no language grounding
no GA
no public beta
no hosted SaaS

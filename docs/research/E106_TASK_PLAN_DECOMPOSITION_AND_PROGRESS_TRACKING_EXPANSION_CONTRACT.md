# E106 Task Plan Decomposition And Progress Tracking Expansion Contract

## Purpose

E106 expands the Operator Library with controlled task-state skills. These
Operators decompose a requested task into required evidence-backed steps, map
deliverables to proof artifacts, maintain step status, preserve blockers and
regressions, and block completion until every required item is proven.

This is not open-domain project management.

## Required Boundary

```text
controlled task plan/progress tracking proxy
not open-domain project management
not autonomous deployment
not direct completion without evidence
not a model-scale claim
```

## Stable Candidate Targets

```text
task_requirement_decomposition_lens
deliverable_evidence_mapping_scribe
step_status_transition_guard
blocked_dependency_tracker_t_stab
progress_ledger_update_scribe
completion_gate_all_requirements_guard
regression_recheck_step_guard
next_action_selector_scribe
```

## Controls

```text
first_step_done_means_complete_control
popularity_step_order_control
ignore_blocked_dependency_control
missing_evidence_complete_control
stale_progress_reuse_control
always_continue_without_status_control
overbroad_done_summary_control
plan_decomposer_echo_clone
```

## Required Artifacts

```text
run_manifest.json
operator_library_manifest.json
task_generation_report.json
progress.jsonl
partial_aggregate_snapshot.json
seed_results.json
aggregate_metrics.json
selection_frequency_report.json
counterfactual_report.json
operator_lifecycle_report.json
mutation_summary.json
deterministic_replay.json
decision.json
summary.json
report.md
row_level_samples.jsonl
operator_evolution_history.jsonl
```

Sample pack:

```text
docs/research/artifact_samples/e106_task_plan_decomposition_and_progress_tracking_expansion/
```

## Metrics

```text
plan_tracking_success
decomposition_validity
evidence_mapping_validity
status_transition_validity
blocked_dependency_preservation
completion_gate_validity
next_action_accuracy
premature_complete_rate
missed_blocker_rate
stale_done_reuse_rate
wrong_next_action_rate
counterfactual plan-tracking loss
accepted/rejected/rollback mutation counts
deterministic replay hash match
checker failure count
```

## Pass Requirements

```text
validation_plan_tracking_success_min = 1.000000
adversarial_plan_tracking_success_min = 1.000000
validation_decomposition_validity_min = 1.000000
validation_evidence_mapping_validity_min = 1.000000
validation_status_transition_validity_min = 1.000000
validation_blocked_dependency_preservation_min = 1.000000
validation_completion_gate_validity_min = 1.000000
validation_next_action_accuracy_min = 1.000000
adversarial_premature_complete_rate_max = 0.000000
adversarial_missed_blocker_rate_max = 0.000000
adversarial_stale_done_reuse_rate_max = 0.000000
adversarial_wrong_next_action_rate_max = 0.000000
checker_failure_count = 0
sample_only_checker_failure_count = 0
deterministic replay passes
```

## Decisions

```text
e106_task_plan_progress_tracking_expansion_confirmed
e106_task_plan_progress_tracking_incomplete
e106_decomposition_failure
e106_evidence_mapping_failure
e106_status_transition_failure
e106_blocker_preservation_failure
e106_completion_gate_failure
e106_artifact_or_replay_failure
```

## Interpretation Rule

A positive result means the library gained scoped task-plan/progress-tracking
Operators for controlled work-state traces.

It does not mean open-domain project management works.

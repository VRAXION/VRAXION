# E97 Operator Route/Composition Planning Expansion Contract

## Purpose

E97 expands the Operator Library with controlled route/composition planning
Operators. The goal is to choose a small active Operator set and an ordered
call sequence for already-scoped Flow/Agency/Proposal tasks.

This is not open-domain planning. It is not a claim about autonomous agents,
language reasoning, model-scale behavior, AGI, or consciousness.

## Required Boundary

```text
controlled Operator route/composition planning proxy
not open-domain planning
not full-library scan
not direct Flow write
not model-scale agent behavior
```

## Systems

Stable candidates must cover these mechanical roles:

```text
route_intent_classifier_lens
active_operator_set_selector_guard
ordered_operator_sequence_scribe
adapter_requirement_detector_lens
loop_prevention_route_guard
route_budget_guard
fallback_to_ask_route_scribe
composition_completion_t_stab
```

Controls:

```text
full_library_scan_router
random_operator_caller
looping_route_runner
budgetless_route_expander
adapterless_cross_abi_caller
always_call_more_control
route_selector_clone
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
archived_public_artifact_sample_removed
```

## Metrics

```text
validation_route_success
adversarial_route_success
active_set_precision
sequence_accuracy
loop_rate
overcall_rate
over_budget_rate
adapter_miss_rate
unsafe_operator_selected
counterfactual route success loss
accepted/rejected/rollback mutation counts
deterministic replay hash match
checker failure count
```

## Pass Requirements

```text
validation_route_success_min = 1.000000
adversarial_route_success_min = 1.000000
validation_active_set_precision_min = 1.000000
validation_sequence_accuracy_min = 1.000000
adversarial_loop_rate_max = 0.000000
adversarial_overcall_rate_max = 0.000000
adversarial_over_budget_max = 0.000000
adversarial_adapter_miss_max = 0.000000
unsafe_final_selected = 0
checker_failure_count = 0
sample_only_checker_failure_count = 0
deterministic replay passes
```

## Decisions

```text
e97_operator_route_composition_planning_expansion_confirmed
e97_operator_route_planning_incomplete
e97_full_library_scan_regression
e97_loop_or_budget_guard_failure
e97_adapter_planning_failure
e97_artifact_or_replay_failure
```

## Interpretation Rule

A positive result means the library gained scoped Operator orchestration
skills: selecting the relevant active Operators, rendering the route order,
detecting adapter needs, and halting or asking when appropriate.

It does not mean the system can perform arbitrary planning.

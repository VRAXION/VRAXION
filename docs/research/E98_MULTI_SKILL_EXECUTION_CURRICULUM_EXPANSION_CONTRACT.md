# E98 Multi-Skill Execution Curriculum Expansion Contract

## Purpose

E98 expands the Operator Library with controlled multi-skill execution hygiene.
The target is not a new domain solver. It is the ability to compose already
scoped Operators into a dependency-correct route with trace continuity,
intermediate state carry, checkpoints, scope boundaries, and safe final answer
assembly.

## Required Boundary

```text
controlled multi-skill Operator composition proxy
not open-domain reasoning
not direct answer shortcut
not full agent behavior
not model-scale claim
```

## Stable Candidate Targets

```text
composite_task_decomposer_lens
dependency_ordering_scribe
intermediate_state_carry_t_stab
cross_skill_trace_join_guard
capability_scope_boundary_guard
partial_route_checkpoint_scribe
composition_error_recovery_scribe
final_response_integrity_guard
```

## Controls

```text
single_skill_shortcut_solver
answer_only_composer
unordered_skill_bag_runner
drop_intermediate_state_control
trace_join_omission_control
scope_bleed_word_solver_control
always_successful_completion_control
decomposer_echo_clone
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
docs/research/artifact_samples/e98_multi_skill_execution_curriculum_expansion/
```

## Metrics

```text
composition_success
dependency_order_accuracy
trace_join_validity
intermediate_carry_validity
checkpoint_accuracy
scope_bleed_rate
dropped_intermediate_rate
unordered_execution_rate
unsafe_answer_rate
counterfactual composition success loss
accepted/rejected/rollback mutation counts
deterministic replay hash match
checker failure count
```

## Pass Requirements

```text
validation_composition_success_min = 1.000000
adversarial_composition_success_min = 1.000000
validation_dependency_order_accuracy_min = 1.000000
validation_trace_join_validity_min = 1.000000
validation_intermediate_carry_validity_min = 1.000000
validation_checkpoint_accuracy_min = 1.000000
adversarial_scope_bleed_rate_max = 0.000000
adversarial_dropped_intermediate_rate_max = 0.000000
adversarial_unordered_execution_rate_max = 0.000000
adversarial_unsafe_answer_rate_max = 0.000000
checker_failure_count = 0
sample_only_checker_failure_count = 0
deterministic replay passes
```

## Decisions

```text
e98_multi_skill_execution_curriculum_expansion_confirmed
e98_multi_skill_execution_curriculum_incomplete
e98_dependency_ordering_failure
e98_trace_join_failure
e98_intermediate_state_carry_failure
e98_scope_bleed_failure
e98_artifact_or_replay_failure
```

## Interpretation Rule

A positive result means the library gained controlled multi-skill execution
hygiene: staged decomposition, dependency ordering, intermediate carry, trace
join, checkpoints, scope boundaries, and final integrity checks.

It does not mean arbitrary text reasoning or open-domain planning works.

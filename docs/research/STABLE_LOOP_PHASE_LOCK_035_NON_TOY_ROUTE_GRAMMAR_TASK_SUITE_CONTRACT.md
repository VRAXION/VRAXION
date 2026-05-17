# STABLE_LOOP_PHASE_LOCK_035_NON_TOY_ROUTE_GRAMMAR_TASK_SUITE Contract

## Summary

034 integrated the route-grammar subsystem as an `instnct-core` experimental
API. 035 hardens that API against less toy-like route tasks.

This is not a new mechanism. It is a non-toy task-suite and regression probe
over:

```text
instnct_core::experimental_route_grammar
```

No production API readiness, full VRAXION, language grounding, consciousness,
biological, FlyWire, or physical quantum claim.

## Required Arms

```text
HAND_PIPELINE_REFERENCE
API_BASELINE_034
NOISY_GRAPH_ROUTE_TASK
MULTI_REQUEST_BATCH
DYNAMIC_EDGE_DELETION_REPAIR
DISTRACTOR_SAME_SOURCE_TARGET
COMPOSITIONAL_ROUTE_LABELS
VARIABLE_PHASE_GATE_POLICY
REACHABLE_SEED_BUG_REGRESSION
NO_GRAMMAR_API_CONTROL
RANDOM_ROUTE_TASK_CONTROL
RANDOM_PHASE_RULE_CONTROL
```

## Additional API Check

```text
BOUNDED_API_ERROR_HANDLING
```

Must reject malformed API inputs with bounded errors:

```text
SourceOutOfBounds
TargetOutOfBounds
EdgeOutOfBounds
```

## Metrics

```text
sufficient_tick_final_accuracy
long_path_accuracy
family_min_accuracy
wrong_if_delivered_rate
route_order_accuracy
retained_successor_accuracy
missing_successor_count
duplicate_successor_count
branch_count
cycle_count
source_to_target_reachability
gate_shuffle_collapse
same_target_counterfactual_accuracy
random_control_accuracy
bounded_api_error_handling_pass
forbidden_private_field_leak
nonlocal_edge_count
direct_output_leak_rate
```

## Verdicts

```text
NON_TOY_ROUTE_GRAMMAR_SUITE_POSITIVE
NOISY_GRAPH_ROUTE_TASK_WORKS
MULTI_REQUEST_BATCH_WORKS
DYNAMIC_EDGE_DELETION_REPAIR_WORKS
DISTRACTOR_SAME_SOURCE_TARGET_WORKS
COMPOSITIONAL_ROUTE_LABELS_WORKS
VARIABLE_PHASE_GATE_POLICY_WORKS
REACHABLE_SEED_BUG_REGRESSION_WORKS
BOUNDED_API_ERROR_HANDLING_WORKS
NO_GRAMMAR_API_CONTROL_FAILS
RANDOM_ROUTE_TASK_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
PRODUCTION_API_NOT_READY
```

## Decision Gate

```text
NON_TOY_ROUTE_GRAMMAR_SUITE_POSITIVE if a non-hand API-backed hardening arm reaches:

sufficient_tick_final_accuracy >= 0.95
long_path_accuracy >= 0.95
family_min_accuracy >= 0.85
wrong_if_delivered_rate <= 0.10
route_order_accuracy >= 0.90
retained_successor_accuracy >= 0.90
missing_successor_count <= 0.05
branch/cycle near zero
same_target_counterfactual_accuracy >= 0.85
gate_shuffle_collapse >= 0.50
random controls fail
bounded API error handling passes
wall/private/nonlocal/direct leaks = 0
```

## Required Outputs

```text
queue.json
progress.jsonl
metrics.jsonl
non_toy_task_metrics.jsonl
api_metrics.jsonl
api_error_metrics.jsonl
task_family_metrics.jsonl
loop_metrics.jsonl
grammar_metrics.jsonl
delivery_metrics.jsonl
routing_metrics.jsonl
family_metrics.jsonl
counterfactual_metrics.jsonl
control_metrics.jsonl
locality_audit.jsonl
mechanism_ranking.json
summary.json
report.md
contract_snapshot.md
examples_sample.jsonl
job_progress/*.jsonl
```

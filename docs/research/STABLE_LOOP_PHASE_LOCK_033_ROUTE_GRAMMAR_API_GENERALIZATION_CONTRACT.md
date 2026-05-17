# STABLE_LOOP_PHASE_LOCK_033_ROUTE_GRAMMAR_API_GENERALIZATION Contract

## Summary

032 proved the integrated toy route-grammar loop:

```text
dense candidate grow
graph diagnostic labels
missing-successor repair
order-aware prune
receive-commit delivery eval
```

033 tests whether that loop can be expressed as a reusable route-grammar subsystem and hold across generalized toy route-task families.

No public `instnct-core` API changes.

## Required Arms

```text
HAND_PIPELINE_REFERENCE
LOOP_032_BASELINE
GENERALIZED_SINGLE_PATH
VARIABLE_WIDTH_PATH
LONG_ROUTE_STRESS
MULTI_TARGET_ROUTE_SET
BRANCHING_ROUTE_TREE
VARIABLE_GATE_RULE_FAMILY
PRODUCTION_LIKE_ROUTE_GRAMMAR_API
NO_GRAMMAR_API_CONTROL
RANDOM_ROUTE_TASK_CONTROL
RANDOM_PHASE_RULE_CONTROL
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
task_family_min_accuracy
generalization_success_rate
api_consistency_score
edge_count_initial
edge_count_final
prune_fraction
gate_shuffle_collapse
same_target_counterfactual_accuracy
random_control_accuracy
```

## Verdicts

```text
ROUTE_GRAMMAR_API_GENERALIZATION_POSITIVE
GENERALIZED_SINGLE_PATH_WORKS
VARIABLE_WIDTH_PATH_WORKS
LONG_ROUTE_STRESS_WORKS
MULTI_TARGET_ROUTE_SET_WORKS
BRANCHING_ROUTE_TREE_WORKS
VARIABLE_GATE_RULE_FAMILY_WORKS
PRODUCTION_LIKE_ROUTE_GRAMMAR_API_WORKS
NO_GRAMMAR_API_CONTROL_FAILS
RANDOM_ROUTE_TASK_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
PRODUCTION_API_NOT_READY
```

## Decision Gate

```text
ROUTE_GRAMMAR_API_GENERALIZATION_POSITIVE if a non-hand generalized arm reaches:

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
```

## Required Outputs

```text
queue.json
progress.jsonl
metrics.jsonl
generalization_metrics.jsonl
api_metrics.jsonl
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

## Claim Boundary

033 can support generalization of the route-grammar loop across toy route-task families and a runner-local reusable subsystem shape. It does not prove production API readiness, full VRAXION, language grounding, consciousness, Prismion uniqueness, biological equivalence, or physical quantum behavior.

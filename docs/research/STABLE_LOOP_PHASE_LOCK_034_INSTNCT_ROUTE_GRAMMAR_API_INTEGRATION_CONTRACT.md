# STABLE_LOOP_PHASE_LOCK_034_INSTNCT_ROUTE_GRAMMAR_API_INTEGRATION Contract

## Summary

033 showed that the route-grammar loop generalizes across toy route-task
families in a runner-local subsystem shape:

```text
dense candidate grow
graph diagnostic labels
missing-successor repair
order-aware prune
receive-commit delivery eval
```

034 moves that shape into an `instnct-core` owned experimental module and uses
the runner through that API. This is an experimental research surface, not
production API readiness.

No full VRAXION, language grounding, consciousness, Prismion uniqueness,
biological, FlyWire, or physical quantum claim.

## API Under Test

```text
instnct_core::experimental_route_grammar
```

The module exposes deterministic route-grammar primitives:

```text
RouteGrammarTask
RouteGrammarConfig
RouteGrammarLabelPolicy
RouteGrammarEdge
construct_route_grammar
```

The API receives dense candidate edges, weak seed successors, and autonomous
graph-diagnostic successor labels, then returns a single ordered successor path.

## Required Arms

```text
HAND_PIPELINE_REFERENCE
RUNNER_LOCAL_033_BASELINE
GENERALIZED_SINGLE_PATH
VARIABLE_WIDTH_PATH
LONG_ROUTE_STRESS
MULTI_TARGET_ROUTE_SET
BRANCHING_ROUTE_TREE
VARIABLE_GATE_RULE_FAMILY
INSTNCT_EXPERIMENTAL_ROUTE_GRAMMAR_API
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
api_consistency_score
edge_count_initial
edge_count_final
prune_fraction
gate_shuffle_collapse
same_target_counterfactual_accuracy
random_control_accuracy
forbidden_private_field_leak
nonlocal_edge_count
direct_output_leak_rate
```

## Verdicts

```text
INSTNCT_ROUTE_GRAMMAR_API_INTEGRATION_POSITIVE
INSTNCT_EXPERIMENTAL_ROUTE_GRAMMAR_API_WORKS
GENERALIZED_SINGLE_PATH_WORKS
VARIABLE_WIDTH_PATH_WORKS
LONG_ROUTE_STRESS_WORKS
MULTI_TARGET_ROUTE_SET_WORKS
BRANCHING_ROUTE_TREE_WORKS
VARIABLE_GATE_RULE_FAMILY_WORKS
NO_GRAMMAR_API_CONTROL_FAILS
RANDOM_ROUTE_TASK_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
PRODUCTION_API_NOT_READY
```

## Decision Gate

```text
INSTNCT_ROUTE_GRAMMAR_API_INTEGRATION_POSITIVE if a non-hand API-backed arm reaches:

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
wall/private/nonlocal/direct leaks = 0
```

## Required Outputs

```text
queue.json
progress.jsonl
metrics.jsonl
api_integration_metrics.jsonl
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

## Guardrails

```text
1. The API module must remain explicitly experimental.
2. Public production readiness is not claimed.
3. Random route-task and random phase-rule controls must fail.
4. No target/ output is committed.
5. Runs must emit heartbeat progress and partial metrics continuously.
```

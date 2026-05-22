# STABLE_LOOP_PHASE_LOCK_040_ROUTE_GRAMMAR_CANARY_TRAINING_ROLLOUT Contract

## Summary

039 showed that route grammar is useful and controlled behind a feature flag in
a bounded rollout matrix. 040 does not enable default training.

040 asks a canary rollout question:

```text
Can route grammar be exposed at 5%, 25%, and 50% in longer/mixed training
settings with rollback gates, non-route regression controls, overhead tracking,
and a regression corpus?
```

This is not a new mechanism probe and not a production promotion.

## Required Arms

```text
NO_ROUTE_GRAMMAR_BASELINE
CANARY_ROUTE_GRAMMAR_EXPOSURE_0
CANARY_ROUTE_GRAMMAR_EXPOSURE_5
CANARY_ROUTE_GRAMMAR_EXPOSURE_25
CANARY_ROUTE_GRAMMAR_EXPOSURE_50
CANARY_ROUTE_GRAMMAR_EXPOSURE_100_DIAGNOSTIC

CANARY_LONG_HORIZON_EXPOSURE_5
CANARY_LONG_HORIZON_EXPOSURE_25
CANARY_MIXED_TASK_EXPOSURE_5
CANARY_MIXED_TASK_EXPOSURE_25
CANARY_NON_ROUTE_EXPOSURE_5
CANARY_NON_ROUTE_EXPOSURE_25

CANARY_ROLLBACK_ON_REGRESSION
CANARY_ROLLBACK_ON_OVERHEAD
CANARY_SHADOW_MODE
CANARY_ROUTE_ONLY_GATE
CANARY_REGRESSION_CORPUS

NON_ROUTE_TASK_REGRESSION_CONTROL
CANARY_INTERFERENCE_CONTROL
RANDOM_ROUTE_GRAMMAR_CONTROL
RANDOM_PHASE_RULE_CONTROL
```

## Metrics

Canary exposure:

```text
canary_exposure_percent
accuracy_by_step
steps_to_80
steps_to_90
steps_to_95
final_accuracy
heldout_accuracy
OOD_accuracy
```

Route structure:

```text
successor_link_accuracy
route_order_accuracy
missing_successor_count
branch_count
cycle_count
source_to_target_reachability
route_continuity_score
family_min_accuracy
wrong_if_delivered_rate
```

Credit signal:

```text
candidate_delta_nonzero_fraction
positive_delta_fraction
mutation_accept_rate
operator_accept_rate
accepted_route_edges_per_step
rejected_bad_route_edges_per_step
```

Rollback and side effects:

```text
rollback_triggered
rollback_success
canary_stop_reason
non_route_task_accuracy_delta
baseline_behavior_drift
false_route_activation_rate
route_api_overuse_rate
task_type_precision
task_type_recall
compute_overhead_ratio
memory_overhead_ratio
```

## Required Outputs

```text
queue.json
progress.jsonl
metrics.jsonl
canary_metrics.jsonl
exposure_metrics.jsonl
rollback_metrics.jsonl
regression_corpus_metrics.jsonl
learning_curves.jsonl
credit_signal_metrics.jsonl
canary_gate_metrics.jsonl
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

## Verdicts

```text
CANARY_TRAINING_ROLLOUT_POSITIVE
CANARY_5_PERCENT_HAS_SIGNAL
CANARY_25_PERCENT_HAS_SIGNAL
CANARY_50_PERCENT_HAS_SIGNAL
CANARY_LONG_HORIZON_STABLE
CANARY_MIXED_TASKS_STABLE
CANARY_LEARNS_SUCCESSOR_STRUCTURE
CANARY_ROUTE_GATING_WORKS
CANARY_ROLLBACK_GATE_WORKS
CANARY_REGRESSION_CORPUS_PASSES
CANARY_OVERHEAD_ACCEPTABLE
CANARY_SHADOW_MODE_SAFE_NOT_SUFFICIENT
CANARY_NO_NON_ROUTE_REGRESSION
CANARY_0_PERCENT_CONTROL_FAILS
RANDOM_ROUTE_GRAMMAR_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
CANARY_CONTROL_CONTAMINATION
CANARY_CAUSES_NON_ROUTE_REGRESSION
CANARY_ROLLOUT_STILL_OPEN
PRODUCTION_API_NOT_READY
```

## Decision Gate

`CANARY_TRAINING_ROLLOUT_POSITIVE` requires:

```text
5%, 25%, and 50% route exposure arms pass route-structure gate
5% and 25% long-horizon arms pass
5% and 25% mixed-task arms pass
5% and 25% non-route arms do not regress
rollback-on-regression and rollback-on-overhead arms pass
route-only gate passes
regression corpus passes
0% and shadow-mode controls do not count as solved
random controls fail
compute_overhead_ratio acceptable
memory_overhead_ratio acceptable
default_training_enabled = false
```

Route-structure gate:

```text
sufficient_tick_final_accuracy >= 0.95
long_path_accuracy >= 0.95
family_min_accuracy >= 0.85
wrong_if_delivered_rate <= 0.10
route_order_accuracy >= 0.90
retained_successor_accuracy >= 0.90
missing_successor_count <= 0.05
same_target_counterfactual_accuracy >= 0.85
gate_shuffle_collapse >= 0.50
```

## Claim Boundary

040 can support:

```text
route grammar is canary-rollout ready in the tested bounded training matrix
```

040 cannot support:

```text
default training enablement
public beta promotion
production API readiness
full VRAXION
language grounding
consciousness
biological/FlyWire equivalence
physical quantum behavior
```

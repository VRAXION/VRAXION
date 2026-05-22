# STABLE_LOOP_PHASE_LOCK_039_FEATURE_FLAG_TRAINING_ROLLOUT_MATRIX Contract

## Summary

038 showed that the experimental route-grammar API is a useful bounded
training/search bias. 039 does not promote it and does not enable default
training.

039 asks a rollout question:

```text
If route grammar is available behind a feature flag, where does it help,
where is it neutral, and where can it interfere?
```

This is a training rollout matrix probe, not a new mechanism probe.

## Required Arms

```text
NO_ROUTE_GRAMMAR_BASELINE
ROUTE_GRAMMAR_FEATURE_FLAG_OFF
ROUTE_GRAMMAR_FEATURE_FLAG_ON_ROUTE_TASKS
ROUTE_GRAMMAR_FEATURE_FLAG_ON_MIXED_TASKS
ROUTE_GRAMMAR_FEATURE_FLAG_ON_OOD_TASKS
ROUTE_GRAMMAR_FEATURE_FLAG_ON_LONG_HORIZON
ROUTE_GRAMMAR_FEATURE_FLAG_ON_NON_ROUTE_TASKS
ROUTE_GRAMMAR_FEATURE_FLAG_ROUTE_ONLY_GATED
ROUTE_GRAMMAR_FEATURE_FLAG_AUTO_DETECT_ROUTE
ROUTE_GRAMMAR_FEATURE_FLAG_FALSE_POSITIVE_STRESS
ROUTE_GRAMMAR_FEATURE_FLAG_COST_CAPPED
ROUTE_GRAMMAR_FEATURE_FLAG_DIAGNOSTICS_SHADOW_ONLY

NON_ROUTE_TASK_REGRESSION_CONTROL
MIXED_TASK_INTERFERENCE_CONTROL
RANDOM_ROUTE_GRAMMAR_CONTROL
RANDOM_PHASE_RULE_CONTROL
```

## Metrics

Learning/search:

```text
accuracy_by_step
loss_or_score_by_step
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

Task rollout:

```text
route_task_accuracy_delta
mixed_task_accuracy_delta
ood_task_accuracy_delta
long_horizon_task_accuracy_delta
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
rollout_matrix_metrics.jsonl
feature_flag_metrics.jsonl
interference_metrics.jsonl
task_type_metrics.jsonl
learning_curves.jsonl
credit_signal_metrics.jsonl
feature_flag_gate_metrics.jsonl
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
FEATURE_FLAG_ROLLOUT_MATRIX_POSITIVE
FEATURE_FLAG_IMPROVES_ROUTE_TASKS
FEATURE_FLAG_IMPROVES_MIXED_TASKS
FEATURE_FLAG_GENERALIZES_OOD
FEATURE_FLAG_HANDLES_LONG_HORIZON
FEATURE_FLAG_LEARNS_SUCCESSOR_STRUCTURE
FEATURE_FLAG_ROUTE_GATING_WORKS
FEATURE_FLAG_COST_CAP_ACCEPTABLE
NO_NON_ROUTE_REGRESSION_OBSERVED
FALSE_ROUTE_ACTIVATION_CONTROL_FAILS
DIAGNOSTICS_SHADOW_ONLY_INSUFFICIENT
FEATURE_FLAG_OFF_CONTROL_FAILS
RANDOM_ROUTE_GRAMMAR_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
FEATURE_FLAG_CONTROL_CONTAMINATION
ROUTE_GRAMMAR_CAUSES_NON_ROUTE_REGRESSION
ROLLOUT_SIGNAL_STILL_WEAK
FEATURE_FLAG_ROLLOUT_STILL_OPEN
PRODUCTION_API_NOT_READY
```

## Decision Gate

`FEATURE_FLAG_ROLLOUT_MATRIX_POSITIVE` requires:

```text
route task arm passes route-structure gate
mixed task arm passes route-structure gate
OOD task arm passes route-structure gate
long-horizon arm passes route-structure gate
route-only gated or auto-detect arm passes
cost-capped arm passes with acceptable overhead
false-positive stress does not pass as a route task
diagnostics-shadow-only does not pass as a full rollout
non_route_task_accuracy_delta >= -0.02
false_route_activation_rate low
route_api_overuse_rate low
random controls fail
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

039 can support:

```text
route grammar is safe and useful as a feature-flagged training/search helper
for the tested rollout matrix
```

039 cannot support:

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

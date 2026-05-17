# STABLE_LOOP_PHASE_LOCK_042_FULL_MODEL_ROUTE_GRAMMAR_OPERATOR_BRIDGE Contract

## Summary

041 showed route grammar is limited-default-on-pilot ready in a bounded training
matrix. 042 is not final training. It bridges the subsystem into a
full-model-style mutation/search operator lane and asks whether it improves
checkpoint/search behavior, not just isolated route tasks.

Do not enable production default training, promote public beta, or claim full
VRAXION, language grounding, consciousness, biological equivalence, or physical
quantum behavior.

## Required Arms

```text
BASELINE_MUTATION_SEARCH
ROUTE_GRAMMAR_OPERATOR_SHADOW_ONLY
ROUTE_GRAMMAR_REPAIR_ONLY
ROUTE_GRAMMAR_PRUNE_ONLY
ROUTE_GRAMMAR_DIAGNOSTIC_LABELS_ONLY
ROUTE_GRAMMAR_FULL_LOOP
ROUTE_GRAMMAR_FULL_LOOP_FEATURE_FLAG
ROUTE_GRAMMAR_FULL_LOOP_COST_CAPPED
ROUTE_GRAMMAR_FULL_LOOP_ROLLBACK_GATED

NON_ROUTE_REGRESSION_CONTROL
MIXED_TASK_REGRESSION_CONTROL
RANDOM_ROUTE_GRAMMAR_CONTROL
RANDOM_PHASE_RULE_CONTROL
```

## Experimental Operators

```text
ADD_ROUTE_GRAMMAR_PATH
REPAIR_MISSING_SUCCESSOR
PRUNE_BRANCH_CYCLE_ROUTE
RECEIVE_COMMIT_DELIVERY
GRAPH_DIAGNOSTIC_LABELS
ORDER_AWARE_PRUNE
```

These are experimental, feature-flagged, rollbackable, and metrics-instrumented
operator-lane components, not production operators.

## Task Suite

```text
ROUTE_TASKS
MIXED_ROUTE_NON_ROUTE_TASKS
NON_ROUTE_TASKS
LONG_HORIZON_ROUTE_TASKS
OOD_ROUTE_TASKS
CONTEXT_CARRY_TASKS
CHECKPOINT_BEHAVIOR_REGRESSION_TASKS
ARTIFACT_SAFETY_CONTROLS
```

## Metrics

Model/search quality:

```text
best_checkpoint_score
heldout_score
ood_score
context_carry_score
artifact_safety_score
output_distribution_drift
behavior_drift_score
steps_to_best_checkpoint
```

Mutation/search dynamics:

```text
accepted_mutation_rate
positive_delta_fraction
candidate_delta_nonzero_fraction
operator_accept_rate
accepted_route_edges_per_step
rejected_bad_route_edges_per_step
```

Route structure:

```text
route_order_accuracy
retained_successor_accuracy
missing_successor_count
branch_count
cycle_count
source_to_target_reachability
wrong_if_delivered_rate
family_min_accuracy
```

Safety/regression:

```text
non_route_regression_delta
artifact_control_pass_rate
random_control_accuracy
baseline_behavior_drift
false_route_activation_rate
route_api_overuse_rate
rollback_success
compute_overhead_ratio
memory_overhead_ratio
default_training_enabled
```

## Required Outputs

```text
queue.json
progress.jsonl
metrics.jsonl
full_model_bridge_metrics.jsonl
checkpoint_metrics.jsonl
search_dynamics_metrics.jsonl
operator_acceptance_metrics.jsonl
route_structure_metrics.jsonl
context_carry_metrics.jsonl
artifact_safety_metrics.jsonl
regression_metrics.jsonl
control_metrics.jsonl
rollback_metrics.jsonl
summary.json
report.md
contract_snapshot.md
examples_sample.jsonl
job_progress/*.jsonl
```

## Verdicts

```text
FULL_MODEL_ROUTE_GRAMMAR_BRIDGE_POSITIVE
ROUTE_GRAMMAR_IMPROVES_CHECKPOINT_SEARCH
ROUTE_GRAMMAR_IMPROVES_CONTEXT_CARRY
ROUTE_GRAMMAR_IMPROVES_OOD
ROUTE_GRAMMAR_IMPROVES_CREDIT_SIGNAL
ROUTE_GRAMMAR_OPERATOR_ACCEPTED
ROUTE_GRAMMAR_CAUSES_NON_ROUTE_REGRESSION
ROUTE_GRAMMAR_CAUSES_ARTIFACT_REGRESSION
ROUTE_GRAMMAR_OVERHEAD_TOO_HIGH
ROUTE_GRAMMAR_SHADOW_ONLY_INSUFFICIENT
RANDOM_ROUTE_GRAMMAR_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
FINAL_TRAINING_NOT_READY
PRODUCTION_API_NOT_READY
```

## Positive Gate

`FULL_MODEL_ROUTE_GRAMMAR_BRIDGE_POSITIVE` requires the full-loop operator arm
to improve over `BASELINE_MUTATION_SEARCH` by at least one:

```text
best_checkpoint_score +0.05
heldout_score +0.05
ood_score +0.05
context_carry_score +0.05
steps_to_best_checkpoint improves >= 25%
```

and:

```text
non_route_regression_delta >= -0.02
artifact_safety_score does not regress
output_distribution_drift within allowed band
route_order_accuracy >= 0.90
missing_successor_count <= 0.05
family_min_accuracy >= 0.85
random controls fail
compute_overhead_ratio <= 1.15
memory_overhead_ratio <= 1.10
rollback_success = true
default_training_enabled = false
```

## Claim Boundary

042 can support:

```text
route grammar improves bounded full-model-style mutation/search checkpoint behavior
```

042 cannot support:

```text
final training readiness
production default training enablement
public beta promotion
production API readiness
full VRAXION
language grounding
consciousness
biological/FlyWire equivalence
physical quantum behavior
```

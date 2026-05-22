# STABLE_LOOP_PHASE_LOCK_041_LIMITED_DEFAULT_ON_TRAINING_PILOT Contract

## Summary

040 showed route grammar is canary-rollout ready in a bounded training matrix.
041 is the next gate: a limited default-on pilot, still behind strict kill-switch,
cost, drift, and regression gates.

This is not a new route mechanism and not a production promotion. It asks:

```text
Can route grammar run as a limited default-on pilot in longer and mixed
training settings without non-route drift, cost blow-up, or regression-corpus
failure?
```

## Required Arms

```text
HAND_PIPELINE_REFERENCE
NO_ROUTE_GRAMMAR_BASELINE
DEFAULT_OFF_CONTROL
CANARY_5_REFERENCE
CANARY_25_REFERENCE

LIMITED_DEFAULT_ON_ROUTE_TASKS
LIMITED_DEFAULT_ON_MIXED_TASKS
LIMITED_DEFAULT_ON_OOD_TASKS
LIMITED_DEFAULT_ON_LONG_HORIZON
LIMITED_DEFAULT_ON_NON_ROUTE_GATED
LIMITED_DEFAULT_ON_HARD_REGRESSION_CORPUS
LIMITED_DEFAULT_ON_KILL_SWITCH_REGRESSION
LIMITED_DEFAULT_ON_KILL_SWITCH_OVERHEAD
LIMITED_DEFAULT_ON_COST_ENVELOPE
LIMITED_DEFAULT_ON_DRIFT_MONITOR
LIMITED_DEFAULT_ON_SHADOW_AUDIT

NON_ROUTE_TASK_REGRESSION_CONTROL
DEFAULT_ON_INTERFERENCE_CONTROL
RANDOM_DEFAULT_ON_CONTROL
RANDOM_PHASE_RULE_CONTROL
```

## Metrics

Training and rollout:

```text
accuracy_by_step
steps_to_80
steps_to_90
steps_to_95
final_accuracy
heldout_accuracy
OOD_accuracy
default_on_exposure_percent
limited_default_on_pilot_enabled
production_default_training_enabled
```

Route structure:

```text
sufficient_tick_final_accuracy
long_path_accuracy
family_min_accuracy
wrong_if_delivered_rate
successor_link_accuracy
route_order_accuracy
retained_successor_accuracy
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

Safety gates:

```text
kill_switch_triggered
kill_switch_success
kill_switch_reason
non_route_drift_delta
baseline_behavior_drift
false_route_activation_rate
route_api_overuse_rate
task_type_precision
task_type_recall
compute_overhead_ratio
memory_overhead_ratio
cost_envelope_pass
hard_regression_pass_rate
```

## Required Outputs

```text
queue.json
progress.jsonl
metrics.jsonl
default_on_pilot_metrics.jsonl
default_exposure_metrics.jsonl
kill_switch_metrics.jsonl
hard_regression_metrics.jsonl
learning_curves.jsonl
credit_signal_metrics.jsonl
default_on_gate_metrics.jsonl
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
LIMITED_DEFAULT_ON_PILOT_POSITIVE
DEFAULT_ON_5_PERCENT_REFERENCE_HAS_SIGNAL
DEFAULT_ON_25_PERCENT_REFERENCE_HAS_SIGNAL
LIMITED_DEFAULT_ON_ROUTE_TASKS_STABLE
DEFAULT_ON_MIXED_TASKS_STABLE
DEFAULT_ON_OOD_TASKS_STABLE
DEFAULT_ON_LONG_HORIZON_STABLE
DEFAULT_ON_LEARNS_SUCCESSOR_STRUCTURE
KILL_SWITCH_GATE_WORKS
COST_ENVELOPE_ACCEPTABLE
DRIFT_MONITOR_GATE_WORKS
HARD_REGRESSION_CORPUS_PASSES
NO_NON_ROUTE_DRIFT_OBSERVED
SHADOW_AUDIT_SAFE_NOT_SUFFICIENT
DEFAULT_OFF_CONTROL_FAILS
RANDOM_DEFAULT_ON_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
DEFAULT_ON_CONTROL_CONTAMINATION
DEFAULT_ON_CAUSES_NON_ROUTE_DRIFT
DEFAULT_ON_PILOT_STILL_OPEN
PRODUCTION_API_NOT_READY
```

## Decision Gate

`LIMITED_DEFAULT_ON_PILOT_POSITIVE` requires:

```text
5% and 25% canary references pass
limited default-on route tasks pass
limited default-on mixed tasks pass
limited default-on OOD tasks pass
limited default-on long-horizon tasks pass
non-route gated/default controls do not regress
kill-switch regression and overhead gates pass
cost envelope and drift monitor gates pass
hard regression corpus passes
shadow audit does not count as solved
random controls fail
production_default_training_enabled = false
public_beta_promoted = false
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

041 can support:

```text
route grammar is limited-default-on-pilot ready in the tested bounded training matrix
```

041 cannot support:

```text
production default training enablement
public beta promotion
production API readiness
full VRAXION
language grounding
consciousness
biological/FlyWire equivalence
physical quantum behavior
```

# STABLE_LOOP_PHASE_LOCK_038_ROUTE_GRAMMAR_TRAINING_INTEGRATION_GATE Contract

## Summary

037 showed the experimental route-grammar API has enough contract evidence to
consider a future beta decision. 038 does not promote it.

038 asks whether the API is useful as a training/search bias:

```text
Does route grammar improve learning dynamics, credit assignment, and OOD route
generalization without harming non-route tasks?
```

## Required Arms

```text
NO_ROUTE_GRAMMAR_BASELINE
ROUTE_GRAMMAR_API_FROZEN_HELPER
ROUTE_GRAMMAR_API_TRAINING_FEATURE_FLAG
ROUTE_GRAMMAR_API_AUX_LABELS_ONLY
ROUTE_GRAMMAR_API_CONSTRUCTOR_ONLY
ROUTE_GRAMMAR_API_CONSTRUCTOR_PLUS_DIAGNOSTICS
ROUTE_GRAMMAR_API_NOISY_CANDIDATES
ROUTE_GRAMMAR_API_ABLATE_DIAGNOSTIC_LABELS
ROUTE_GRAMMAR_API_ABLATE_ORDER_PRUNE
ROUTE_GRAMMAR_API_ABLATE_RECEIVE_COMMIT_LEDGER
NON_ROUTE_TASK_REGRESSION_CONTROL
RANDOM_ROUTE_GRAMMAR_CONTROL
RANDOM_PHASE_RULE_CONTROL
```

## Metrics

```text
accuracy_by_step
steps_to_80
steps_to_90
steps_to_95
final_accuracy
heldout_accuracy
ood_accuracy
successor_link_accuracy
route_order_accuracy
missing_successor_count
branch_count
cycle_count
source_to_target_reachability
route_continuity_score
candidate_delta_nonzero_fraction
positive_delta_fraction
mutation_accept_rate
operator_accept_rate
accepted_route_edges_per_step
rejected_bad_route_edges_per_step
short_to_long_transfer
variable_width_transfer
multi_target_transfer
branching_route_transfer
variable_gate_policy_transfer
heldout_route_family_accuracy
non_route_task_accuracy_delta
baseline_behavior_drift
false_route_activation_rate
route_api_overuse_rate
compute_overhead_ratio
memory_overhead_ratio
```

## Verdicts

```text
ROUTE_GRAMMAR_TRAINING_INTEGRATION_POSITIVE
ROUTE_GRAMMAR_IMPROVES_SAMPLE_EFFICIENCY
ROUTE_GRAMMAR_IMPROVES_CREDIT_SIGNAL
ROUTE_GRAMMAR_LEARNS_SUCCESSOR_STRUCTURE
ROUTE_GRAMMAR_GENERALIZES_OOD
ROUTE_GRAMMAR_CAUSES_NON_ROUTE_REGRESSION
ROUTE_GRAMMAR_OVERHEAD_TOO_HIGH
DIAGNOSTIC_LABELS_REQUIRED
ORDER_PRUNE_REQUIRED
RECEIVE_COMMIT_LEDGER_REQUIRED
TRAINING_SIGNAL_STILL_WEAK
RANDOM_ROUTE_GRAMMAR_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
PRODUCTION_API_NOT_READY
```

## Decision Gate

```text
TRAINING_INTEGRATION_POSITIVE if route grammar ON beats baseline by:

steps_to_95 improved >= 25%
or final heldout accuracy +0.10
or OOD accuracy +0.10

and:

non_route_task_accuracy_delta >= -0.02
random controls fail
route_order_accuracy >= 0.90
missing_successor_count <= 0.05
family_min_accuracy >= 0.85
compute_overhead_ratio acceptable
```

## Required Outputs

```text
queue.json
progress.jsonl
metrics.jsonl
training_integration_metrics.jsonl
learning_curves.jsonl
credit_signal_metrics.jsonl
training_gate_metrics.jsonl
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

038 can support that route grammar is useful as an experimental training/search
bias in the tested suite. It cannot claim default training integration,
production readiness, full VRAXION, language grounding, consciousness,
biological equivalence, FlyWire wiring, or physical quantum behavior.

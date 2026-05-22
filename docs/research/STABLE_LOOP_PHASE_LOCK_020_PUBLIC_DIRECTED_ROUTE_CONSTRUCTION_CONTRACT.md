# STABLE_LOOP_PHASE_LOCK_020_PUBLIC_DIRECTED_ROUTE_CONSTRUCTION Contract

## Summary

019 established:

```text
correct directed route + receive-committed target ledger
  -> stable settled delivery when ticks are sufficient
```

020 tests the next blocker:

```text
Can public information construct a clean forward-only directed route without
true_path?
```

This is route construction only. It does not change the phase rule or public
`instnct-core` APIs.

## Fixed Mechanism

The completed phase rule remains:

```text
phase_i + gate_g -> phase_(i+g)
```

Readout uses the 019 receive-commit ledger. Public arms may use:

```text
wall mask
source
source phase
target
local free-neighbor relations
gates
```

Public arms must not use:

```text
true_path
oracle next cell
gate_sum
label
direct target oracle
```

## Required Arms

```text
BIDIRECTIONAL_GRID_BASELINE_019
TRUE_PATH_RECEIVE_COMMIT_LEDGER_DIAGNOSTIC
TRUE_PATH_PLUS_REVERSE_RECEIVE_COMMIT_LEDGER

PUBLIC_GRADIENT_RECEIVE_COMMIT_LEDGER
PUBLIC_MONOTONE_RECEIVE_COMMIT_LEDGER
PUBLIC_BFS_SHORTEST_ROUTE_RECEIVE_COMMIT_LEDGER
PUBLIC_DISTANCE_FIELD_DAG_RECEIVE_COMMIT_LEDGER
PUBLIC_DISTANCE_FIELD_SINGLE_SUCCESSOR_RECEIVE_COMMIT_LEDGER
PUBLIC_FRONTIER_PARENT_ROUTE_RECEIVE_COMMIT_LEDGER
PUBLIC_WALL_FOLLOW_RIGHT_RECEIVE_COMMIT_LEDGER
PUBLIC_WALL_FOLLOW_LEFT_RECEIVE_COMMIT_LEDGER
PUBLIC_ACYCLIC_NO_RECIPROCAL_RECEIVE_COMMIT_LEDGER

RANDOM_SAME_COUNT_RECEIVE_COMMIT_LEDGER
DIRECTION_SHUFFLE_RECEIVE_COMMIT_LEDGER
RANDOM_PHASE_RULE_RECEIVE_COMMIT_LEDGER
```

Diagnostic-only:

```text
TRUE_PATH_RECEIVE_COMMIT_LEDGER_DIAGNOSTIC
TRUE_PATH_PLUS_REVERSE_RECEIVE_COMMIT_LEDGER
```

## Metrics

Core metrics:

```text
phase_final_accuracy
sufficient_tick_final_accuracy
long_path_accuracy
family_min_accuracy
correct_target_lane_probability_mean
target_delivery_rate
wrong_if_delivered_rate
duplicate_delivery_rate
stale_delivery_rate
ledger_power_total
reverse_edge_sensitivity
reciprocal_edge_fraction
backflow_edge_fraction
directed_edge_count
gate_shuffle_collapse
same_target_counterfactual_accuracy
random_phase_rule_accuracy
direction_shuffle_accuracy
random_same_count_accuracy
forbidden_private_field_leak
nonlocal_edge_count
direct_output_leak_rate
```

Diagnostic route metrics may compare public constructed routes to `true_path`
after evaluation, but those metrics cannot be used by public arms at runtime.

## Verdicts

```text
PUBLIC_DIRECTED_ROUTE_CONSTRUCTION_POSITIVE
TRUE_PATH_UPPER_BOUND_CONFIRMED
PUBLIC_BFS_ROUTE_WORKS
PUBLIC_DISTANCE_FIELD_ROUTE_WORKS
PUBLIC_DISTANCE_FIELD_DAG_FANOUT_CONTAMINATION
PUBLIC_WALL_FOLLOW_FAILS
PUBLIC_GRADIENT_STILL_FAILS
PUBLIC_MONOTONE_STILL_FAILS
PUBLIC_ROUTE_CONSTRUCTION_FAILS
ROUTING_POLICY_STILL_BLOCKED
REVERSE_EDGES_BREAK_SETTLED_DELIVERY
RANDOM_DIRECTED_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
ROUTE_PRIOR_OVERPOWERS_CONTROL
DIRECT_SHORTCUT_CONTAMINATION
PRODUCTION_API_NOT_READY
```

## Positive Gate

`PUBLIC_DIRECTED_ROUTE_CONSTRUCTION_POSITIVE` requires a public non-diagnostic
arm to reach:

```text
sufficient_tick_final_accuracy >= 0.95
long_path_accuracy >= 0.95
family_min_accuracy >= 0.85
wrong_if_delivered_rate <= 0.10
same_target_counterfactual_accuracy >= 0.85
gate_shuffle_collapse >= 0.50
random controls fail
wall/private/nonlocal/direct leaks = 0
```

If only true-path works:

```text
ROUTING_POLICY_STILL_BLOCKED
```

## Required Outputs

```text
queue.json
progress.jsonl
metrics.jsonl
route_metrics.jsonl
routing_metrics.jsonl
delivery_metrics.jsonl
control_metrics.jsonl
family_metrics.jsonl
counterfactual_metrics.jsonl
locality_audit.jsonl
mechanism_ranking.json
summary.json
report.md
contract_snapshot.md
examples_sample.jsonl
job_progress/*.jsonl
```

No black-box rule:

```text
append progress at heartbeat
append metrics after every arm/family/path/tick block
refresh summary.json and report.md on heartbeat
do not commit target/ outputs
```

## Claim Boundary

020 can support a public route-prior result for the toy phase-lane substrate. It
cannot claim production architecture, FlyWire validation, full VRAXION, language
grounding, consciousness, Prismion uniqueness, biological equivalence, or
physical quantum behavior.

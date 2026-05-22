# STABLE_LOOP_PHASE_LOCK_021_ROUTE_IDENTITY_BREADCRUMB_FRONTIER Contract

## Summary

020 showed that simple public route priors do not recover the phase-label path:

```text
shortest path / BFS / distance field != route identity
```

021 tests whether an explicit public breadcrumb / route-token field can provide
the missing route identity for the 019 receive-commit delivery substrate.

This probe does not test breadcrumb learning. It tests whether the substrate can
use a public route identity once present.

## Fixed Mechanism

The completed phase rule remains:

```text
phase_i + gate_g -> phase_(i+g)
```

Readout uses the 019 receive-commit ledger.

## Public Route Identity

Each case exposes:

```text
route_breadcrumb[cell] : bool
route_order[cell]      : public route-frontier ordinal or unset
```

Public breadcrumb arms may use these fields. They may not use:

```text
private true_path
oracle next cell
label
gate_sum
direct target oracle
```

## Arms

```text
PUBLIC_DISTANCE_FIELD_SINGLE_SUCCESSOR_BASELINE
TRUE_PATH_RECEIVE_COMMIT_LEDGER_DIAGNOSTIC

PUBLIC_BREADCRUMB_MASK_BFS_RECEIVE_COMMIT_LEDGER
PUBLIC_BREADCRUMB_ORDERED_SUCCESSOR_RECEIVE_COMMIT_LEDGER
PUBLIC_BREADCRUMB_FRONTIER_PARENT_RECEIVE_COMMIT_LEDGER
PUBLIC_BREADCRUMB_MASK_PLUS_PUBLIC_NO_BACKFLOW_RECEIVE_COMMIT_LEDGER
PUBLIC_BREADCRUMB_MASK_WITH_SPURS_RECEIVE_COMMIT_LEDGER

RANDOM_BREADCRUMB_SAME_COUNT_RECEIVE_COMMIT_LEDGER
BREADCRUMB_ORDER_SHUFFLE_RECEIVE_COMMIT_LEDGER
RANDOM_SAME_COUNT_RECEIVE_COMMIT_LEDGER
DIRECTION_SHUFFLE_RECEIVE_COMMIT_LEDGER
RANDOM_PHASE_RULE_RECEIVE_COMMIT_LEDGER
```

Diagnostic-only:

```text
TRUE_PATH_RECEIVE_COMMIT_LEDGER_DIAGNOSTIC
```

## Metrics

```text
phase_final_accuracy
sufficient_tick_final_accuracy
long_path_accuracy
family_min_accuracy
wrong_if_delivered_rate
gate_shuffle_collapse
same_target_counterfactual_accuracy
target_delivery_rate
duplicate_delivery_rate
stale_delivery_rate
ledger_power_total
reciprocal_edge_fraction
backflow_edge_fraction
directed_edge_count
random_phase_rule_accuracy
direction_shuffle_accuracy
random_same_count_accuracy
forbidden_private_field_leak
nonlocal_edge_count
direct_output_leak_rate
```

## Verdicts

```text
TRUE_PATH_UPPER_BOUND_CONFIRMED
ROUTE_IDENTITY_BREADCRUMB_POSITIVE
ROUTE_IDENTITY_BREADCRUMB_FAILS
BREADCRUMB_MASK_BFS_WORKS
BREADCRUMB_ORDERED_SUCCESSOR_WORKS
BREADCRUMB_FRONTIER_WORKS
BREADCRUMB_SPURS_BREAK_ROUTE_IDENTITY
PUBLIC_DISTANCE_FIELD_BASELINE_STILL_FAILS
ROUTE_IDENTITY_REQUIRED
BREADCRUMB_RANDOM_CONTROL_FAILS
BREADCRUMB_ORDER_SHUFFLE_FAILS
BREADCRUMB_OVERPOWERS_CONTROL
RANDOM_DIRECTED_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
DIRECT_SHORTCUT_CONTAMINATION
PRODUCTION_API_NOT_READY
```

## Positive Gate

Public breadcrumb positive requires a non-diagnostic breadcrumb arm:

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

021 can support that explicit route identity / breadcrumb state is sufficient for
this toy phase-lane substrate. It cannot prove breadcrumb learning, production
routing, full VRAXION, language grounding, consciousness, Prismion uniqueness,
biological equivalence, or physical quantum behavior.

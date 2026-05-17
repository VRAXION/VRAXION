# STABLE_LOOP_PHASE_LOCK_019_DIRECTED_ROUTE_SETTLED_DELIVERY Contract

## Summary

018 showed that the correct forward-only directed route gives clean arrival:

```text
TRUE_PATH_DIRECTED_ROUTE_DIAGNOSTIC:
  sufficient_tick_best_accuracy = 1.000
  wrong_if_arrived_rate = 0.000
```

But final-tick readout misses the pulse:

```text
phase_final_accuracy = 0.188
```

019 tests whether receive-committed target delivery can preserve that clean
arrival as stable final output.

This is not routing search. It separates:

```text
directed route carries pulse
target delivery/readout preserves pulse
public routing remains open
```

No public `instnct-core` API changes.

## Required Files

```text
docs/research/STABLE_LOOP_PHASE_LOCK_019_DIRECTED_ROUTE_SETTLED_DELIVERY_CONTRACT.md
instnct-core/examples/phase_lane_directed_route_settled_delivery.rs
docs/research/STABLE_LOOP_PHASE_LOCK_019_DIRECTED_ROUTE_SETTLED_DELIVERY_RESULT.md
```

## Arms

```text
BIDIRECTIONAL_GRID_BASELINE_018
TRUE_PATH_DIRECTED_ROUTE_FINAL_TICK
TRUE_PATH_DIRECTED_ROUTE_BEST_TICK_DIAGNOSTIC

TRUE_PATH_DIRECTED_ROUTE_TARGET_LATCH_1TICK
TRUE_PATH_DIRECTED_ROUTE_TARGET_SETTLED_LEDGER_SUM
TRUE_PATH_DIRECTED_ROUTE_TARGET_SETTLED_LEDGER_MAX
TRUE_PATH_DIRECTED_ROUTE_RECEIVE_COMMIT_LEDGER_SUM
TRUE_PATH_DIRECTED_ROUTE_RECEIVE_COMMIT_LEDGER_MAX
TRUE_PATH_DIRECTED_ROUTE_CONSUME_ON_DELIVERY

TRUE_PATH_PLUS_REVERSE_RECEIVE_COMMIT_LEDGER
PUBLIC_GRADIENT_RECEIVE_COMMIT_LEDGER
PUBLIC_MONOTONE_RECEIVE_COMMIT_LEDGER

RANDOM_SAME_COUNT_RECEIVE_COMMIT_LEDGER
DIRECTION_SHUFFLE_RECEIVE_COMMIT_LEDGER
RANDOM_PHASE_RULE_RECEIVE_COMMIT_LEDGER
```

Diagnostic-only arms:

```text
TRUE_PATH_DIRECTED_ROUTE_*
TRUE_PATH_PLUS_REVERSE_RECEIVE_COMMIT_LEDGER
TRUE_PATH_DIRECTED_ROUTE_BEST_TICK_DIAGNOSTIC
```

Ledger rules:

```text
valid ledger entry requires:
  packet enters target through incoming directed edge
  target applies/accepts local gate semantics
  receive_commit = true

forbidden:
  arbitrary past snapshot scan
  best-tick as main claim
  direct target phase bucket oracle
```

Ledger modes:

```text
TARGET_LATCH_1TICK:
  target stores delivery one extra tick

TARGET_SETTLED_LEDGER_SUM:
  sum all receive-committed target phase deliveries

TARGET_SETTLED_LEDGER_MAX:
  keep max seen receive-committed target phase delivery

RECEIVE_COMMIT_LEDGER_SUM:
  sum only edge-delivered target receive commits

RECEIVE_COMMIT_LEDGER_MAX:
  max only edge-delivered target receive commits

CONSUME_ON_DELIVERY:
  consume the delivered target lane after receive commit
```

## Metrics And Verdicts

Core metrics:

```text
phase_final_accuracy
best_tick_accuracy
settled_final_accuracy
sufficient_tick_final_accuracy
long_path_accuracy
family_min_accuracy
correct_target_lane_probability_mean
target_delivery_rate
target_wrong_delivery_rate
wrong_if_delivered_rate
delivery_tick_histogram
first_delivery_tick_mean
final_minus_best_gap
ledger_power_total
duplicate_delivery_rate
stale_delivery_rate
reverse_edge_sensitivity
gate_shuffle_collapse
same_target_counterfactual_accuracy
random_phase_rule_accuracy
direction_shuffle_accuracy
random_same_count_accuracy
forbidden_private_field_leak
nonlocal_edge_count
direct_output_leak_rate
```

Required verdicts:

```text
DIRECTED_ROUTE_DELIVERY_SOLVES_DIAGNOSTIC
RECEIVE_COMMIT_LEDGER_STABILIZES_FINAL_READOUT
RECEIVE_COMMIT_LEDGER_SUM_WORKS
RECEIVE_COMMIT_LEDGER_MAX_WORKS
TARGET_LATCH_INSUFFICIENT
BEST_TICK_ONLY_NOT_STABLE
FINAL_READOUT_TIMING_LIMIT_CONFIRMED
REVERSE_EDGES_BREAK_SETTLED_DELIVERY
LEDGER_MASKS_ECHO_NOT_TRANSPORT
TRUE_PATH_DELIVERY_WORKS_PUBLIC_ROUTING_OPEN
PUBLIC_DELIVERY_ROUTE_HAS_SIGNAL
PUBLIC_ROUTE_DELIVERY_FAILS
ROUTING_POLICY_STILL_BLOCKED
TARGET_LEDGER_OVERPOWERS_CONTROL
DUPLICATE_DELIVERY_CONTAMINATION
STALE_DELIVERY_CONTAMINATION
RANDOM_DIRECTED_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
DIRECT_SHORTCUT_CONTAMINATION
PRODUCTION_API_NOT_READY
```

Strong diagnostic delivery gate:

```text
TRUE_PATH + RECEIVE_COMMIT_LEDGER passes if:
  phase_final_accuracy >= 0.95
  settled_final_accuracy >= 0.95
  long_path_accuracy >= 0.95
  family_min_accuracy >= 0.85
  wrong_if_delivered_rate <= 0.10
  random controls fail
  leaks = 0
```

## Outputs

```text
queue.json
progress.jsonl
metrics.jsonl
delivery_metrics.jsonl
readout_timing_metrics.jsonl
routing_metrics.jsonl
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

No-black-box rule:

```text
append progress at heartbeat
append metrics after every arm/family/path/tick block
refresh summary.json and report.md on heartbeat
do not commit target/ outputs
```

## Claim Boundary

019 can support directed-route delivery/readout stability only. It cannot claim
public routing solved unless public route arms pass controls. It cannot claim
production architecture, FlyWire validation, full VRAXION, language grounding,
consciousness, Prismion uniqueness, biological equivalence, or physical quantum
behavior.

# STABLE_LOOP_PHASE_LOCK_025_ROUTE_ORDER_AWARE_PRUNE Contract

## Summary

024 showed that dense-grow-prune has signal but does not pass:

```text
sufficient_tick_final_accuracy ~= 0.91
long_path_accuracy ~= 0.85
wrong_if_delivered_rate ~= 0.07
family_min_accuracy = 0.000
route_order_accuracy ~= 0.49
```

025 tests whether the prune objective can preserve ordered successor route structure, not merely delivery.

## Fixed Substrate

Keep:

```text
phase_i + gate_g -> phase_(i+g)
directed route transport
receive-commit target ledger
dense candidate route field input
```

No public `instnct-core` API changes.

## Required Arms

```text
TRUE_PATH_UPPER_BOUND
HAND_DENSE_THEN_PRUNE_REFERENCE
DENSE_GROW_4X_024_BASELINE
DELIVERY_ONLY_PRUNE
DELIVERY_PLUS_SUCCESSOR_CONSISTENCY
DELIVERY_PLUS_BRANCH_PENALTY
DELIVERY_PLUS_CYCLE_PENALTY
DELIVERY_PLUS_ROUTE_CONTINUITY
DELIVERY_PLUS_FAMILY_MIN_ADVERSARIAL
DELIVERY_PLUS_ALL_ORDER_REGULARIZERS
SOURCE_TARGET_ANCHOR_PLUS_ORDER_PRUNE
RANDOM_DENSE_CONTROL
RANDOM_PHASE_RULE_CONTROL
```

`TRUE_PATH_UPPER_BOUND` and `HAND_DENSE_THEN_PRUNE_REFERENCE` are references. The positive claim must come from non-control order-aware prune arms.

## Metrics

```text
sufficient_tick_final_accuracy
long_path_accuracy
family_min_accuracy
wrong_if_delivered_rate
retained_successor_accuracy
route_order_accuracy
branch_count
cycle_count
duplicate_successor_count
missing_successor_count
route_continuity_score
source_to_target_reachability
prune_fraction
final_edge_count
gate_shuffle_collapse
same_target_counterfactual_accuracy
random_control_accuracy
```

## Verdicts

```text
ROUTE_ORDER_AWARE_PRUNE_POSITIVE
SUCCESSOR_CONSISTENCY_REQUIRED
BRANCH_PENALTY_REQUIRED
CYCLE_PENALTY_REQUIRED
ROUTE_CONTINUITY_REQUIRED
FAMILY_MIN_ADVERSARIAL_REQUIRED
DELIVERY_ONLY_PRUNE_INSUFFICIENT
RANDOM_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
PRODUCTION_API_NOT_READY
```

## Decision Gate

Report `ROUTE_ORDER_AWARE_PRUNE_POSITIVE` only if a non-private order-aware prune arm reaches:

```text
sufficient_tick_final_accuracy >= 0.95
long_path_accuracy >= 0.95
family_min_accuracy >= 0.85
wrong_if_delivered_rate <= 0.10
route_order_accuracy >= 0.90
retained_successor_accuracy >= 0.90
branch_count <= 0.05
cycle_count <= 0.05
same_target_counterfactual_accuracy >= 0.85
gate_shuffle_collapse >= 0.50
random controls fail
wall/private/nonlocal/direct leaks = 0
```

If delivery-only prune remains below the gate and all-order regularizers pass, the blocker is specifically prune objective quality.

## Required Outputs

```text
queue.json
progress.jsonl
metrics.jsonl
order_prune_metrics.jsonl
regularizer_metrics.jsonl
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

No black-box runs: append heartbeat progress and refresh `summary.json` / `report.md` during long runs.

## Claim Boundary

025 can support route-order-aware prune behavior in toy phase-lane tasks only. It cannot claim production routing, full VRAXION, language grounding, consciousness, Prismion uniqueness, biological equivalence, or physical quantum behavior.

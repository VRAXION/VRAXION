# STABLE_LOOP_PHASE_LOCK_025_ROUTE_ORDER_AWARE_PRUNE Result

Status: complete.

## Verdicts

```text
ROUTE_ORDER_AWARE_PRUNE_POSITIVE
ALL_ORDER_REGULARIZERS_WORK
DELIVERY_ONLY_PRUNE_INSUFFICIENT
HAND_DENSE_REFERENCE_REPRODUCED
TRUE_PATH_UPPER_BOUND_CONFIRMED
RANDOM_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
PRODUCTION_API_NOT_READY
```

## Smoke Summary

3-seed smoke:

```text
seeds: 2026,2027,2028
widths: 8,12,16
path_lengths: 4,8,16,24,32
ticks: 8,16,24,32,48
rows: 20475
```

## Baseline Failure Reproduced

025 reproduces the 024 blocker:

```text
DELIVERY_ONLY_PRUNE:
  sufficient_tick_final_accuracy = 0.931
  long_path_accuracy = 0.898
  family_min_accuracy = 0.000
  wrong_if_delivered_rate = 0.046
  retained_successor_accuracy = 0.483
  route_order_accuracy = 0.474
  missing_successor_count = 12.1
  gate_shuffle_collapse = 0.530
  same_target_counterfactual_accuracy = 0.946
```

Interpretation:

```text
delivery-only prune carries useful signal,
but it still does not preserve ordered successor route structure.
```

It reaches decent aggregate delivery metrics but fails the robust claim because the ordered route is only about half recovered and some family bucket still collapses.

## Positive Arm

The all-order regularizer arm passes:

```text
DELIVERY_PLUS_ALL_ORDER_REGULARIZERS:
  sufficient_tick_final_accuracy = 1.000
  long_path_accuracy = 1.000
  family_min_accuracy = 1.000
  wrong_if_delivered_rate = 0.000
  retained_successor_accuracy = 1.000
  route_order_accuracy = 1.000
  missing_successor_count = 0.0
  branch_count = 0.0
  cycle_count = 0.0
  route_continuity_score = 1.000
  source_to_target_reachability = 1.000
  gate_shuffle_collapse = 0.719
  same_target_counterfactual_accuracy = 1.000
```

`SOURCE_TARGET_ANCHOR_PLUS_ORDER_PRUNE` also passes the same structural gate:

```text
SOURCE_TARGET_ANCHOR_PLUS_ORDER_PRUNE:
  sufficient_tick_final_accuracy = 1.000
  long_path_accuracy = 1.000
  family_min_accuracy = 1.000
  wrong_if_delivered_rate = 0.000
  retained_successor_accuracy = 1.000
  route_order_accuracy = 1.000
```

## Controls

Random controls fail:

```text
RANDOM_DENSE_CONTROL:
  sufficient_tick_final_accuracy = 0.641
  long_path_accuracy = 0.356
  wrong_if_delivered_rate = 0.389

RANDOM_PHASE_RULE_CONTROL:
  sufficient_tick_final_accuracy = 0.495
  long_path_accuracy = 0.490
```

## Interpretation

The 025 answer is:

```text
delivery is not enough.
route-order-aware regularization is required.
```

024 showed that dense growth finds useful delivery signal but misses ordered route recovery:

```text
route_order_accuracy ~= 0.49
family_min_accuracy = 0.000
```

025 shows that when prune is constrained by order regularizers, the dense field can be reduced to the full ordered successor route:

```text
retained_successor_accuracy = 1.000
route_order_accuracy = 1.000
family_min_accuracy = 1.000
wrong_if_delivered_rate = 0.000
```

## Current Blocker

The remaining blocker is not whether order-aware pruning can work. It can.

The next blocker is constructability of the order regularizer itself:

```text
Can the system learn or build the route-order regularizer / successor constraint
without being handed the ordered route?
```

This points to a next probe around learned route-order constraints, successor consistency objectives, or a curriculum that constructs the order regularizer from partial scaffolds.

## Claim Boundary

025 supports route-order-aware prune behavior in toy phase-lane tasks. It does not claim production routing, full VRAXION, language grounding, consciousness, Prismion uniqueness, biological equivalence, or physical quantum behavior.

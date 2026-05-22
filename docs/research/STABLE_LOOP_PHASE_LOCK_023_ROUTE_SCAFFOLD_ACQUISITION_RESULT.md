# STABLE_LOOP_PHASE_LOCK_023_ROUTE_SCAFFOLD_ACQUISITION Result

Status: complete.

## Verdicts

```text
SCAFFOLD_ACQUISITION_POSITIVE
DENSE_THEN_PRUNE_ROUTE_WORKS
SOURCE_TARGET_ANCHOR_SCAFFOLD_USEFUL
TRUE_PATH_UPPER_BOUND_CONFIRMED
RANDOM_FROM_SCRATCH_STILL_FAILS
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

The strongest scaffold acquisition result is not random growth. It is dense candidate route field followed by crystallize/prune back to the ordered successor route:

```text
BIDIRECTIONAL_CANDIDATE_FIELD_THEN_PRUNE:
  sufficient_tick_final_accuracy = 1.000
  long_path_accuracy = 1.000
  family_min_accuracy = 1.000
  wrong_if_delivered_rate = 0.000
  initial_successor_link_accuracy = 1.000
  initial_route_order_accuracy = 1.000
  scaffold_coverage = 1.000
  scaffold_noise_rate = 0.000
  gate_shuffle_collapse = 0.719
  same_target_counterfactual_accuracy = 1.000

DENSE_THEN_CRYSTALLIZE_PRUNE_ROUTE:
  sufficient_tick_final_accuracy = 1.000
  long_path_accuracy = 1.000
  family_min_accuracy = 1.000
  wrong_if_delivered_rate = 0.000
```

This reproduces the true-path upper bound after pruning:

```text
TRUE_PATH_UPPER_BOUND:
  sufficient_tick_final_accuracy = 1.000
  long_path_accuracy = 1.000
  family_min_accuracy = 1.000
  wrong_if_delivered_rate = 0.000
```

## Useful But Not Passing Scaffold

`SOURCE_TARGET_ANCHOR_SCAFFOLD` has measurable signal but does not pass the robust gate:

```text
phase_final_accuracy = 0.882
sufficient_tick_final_accuracy = 0.910
long_path_accuracy = 0.862
family_min_accuracy = 0.000
wrong_if_delivered_rate = 0.055
gate_shuffle_collapse = 0.557
same_target_counterfactual_accuracy = 0.910
```

Interpretation:

```text
source/target anchor scaffold can point the search in a useful direction,
but it is not robust across all gate families.
```

## Failed Scaffold Sources

Simple public route priors remain insufficient:

```text
DISTANCE_FIELD_SCAFFOLD:
  sufficient_tick_final_accuracy = 0.819
  long_path_accuracy = 0.670
  wrong_if_delivered_rate = 0.216

FRONTIER_PARENT_SCAFFOLD:
  sufficient_tick_final_accuracy = 0.627
  long_path_accuracy = 0.363

NOISY_BFS_SCAFFOLD:
  sufficient_tick_final_accuracy = 0.627
  long_path_accuracy = 0.363
```

Random from scratch still fails:

```text
RANDOM_FROM_SCRATCH_BASELINE:
  sufficient_tick_final_accuracy = 0.209
  long_path_accuracy = 0.168
  family_min_accuracy = 0.000

RANDOM_SCAFFOLD_CONTROL:
  sufficient_tick_final_accuracy = 0.209
  long_path_accuracy = 0.168

RANDOM_PHASE_RULE_CONTROL:
  sufficient_tick_final_accuracy = 0.495
  long_path_accuracy = 0.490
```

## Interpretation

022 showed that an ordered successor field can be repaired, completed, and pruned when scaffolded.

023 adds:

```text
dense candidate field -> crystallize/prune -> ordered successor route
```

is a viable scaffold acquisition strategy in this toy phase-lane setting.

The result does not solve random-from-scratch public route growth. It narrows the next blocker:

```text
How do we grow or propose the dense candidate route field cheaply enough,
then prune it to the ordered successor route?
```

## Claim Boundary

023 supports scaffold acquisition strategies for ordered-successor route-token fields in toy phase-lane tasks. It does not claim production routing, canonical mutation from scratch, full VRAXION, language grounding, consciousness, Prismion uniqueness, biological equivalence, or physical quantum behavior.

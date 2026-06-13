# E47 Edge Adapter Pocket Growth Probe Result

## Decision

```text
decision = e47_edge_adapter_pocket_positive
target_checker_failure_count = 0
sample_only_checker_failure_count = 0
run_id = d9ec0afd179451ec
```

E47 tested whether the edge/rail between two frozen nodes should be an explicit
adapter pocket instead of a passive wire bundle.

## Result Table

```text
| system | edge_type | physical_width | active_bits | growth_events | heldout_success | ood_success | old_intent_success | old_intent_regression | bit_accuracy | attempts_to_95 | accepted_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| raw_wire_direct_fixed16 | raw_wire | 16 | 8 | 0 | 0.062 | 0.062 | 0.073 | 0.927 | 0.688 | none | 0.000 |
| raw_wire_progressive_plus1 | raw_wire | 8 | 8 | 3 | 0.062 | 0.062 | 0.073 | 0.927 | 0.688 | none | 0.000 |
| edge_adapter_fixed16_to16 | adapter_pocket | 16 | 8 | 0 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 723 | 0.010 |
| edge_adapter_progressive_plus1_freeze_old | adapter_pocket | 8 | 8 | 3 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 4032 | 0.000 |
| edge_adapter_progressive_plus1_no_freeze | adapter_pocket | 8 | 8 | 3 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 4032 | 0.001 |
| edge_adapter_block_growth_plus4 | adapter_pocket | 8 | 8 | 3 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 2016 | 0.000 |
| identity_oracle_adapter_reference | oracle_adapter_reference | 8 | 8 | 3 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0 | 0.000 |
| random_adapter_control | random_adapter_control | 16 | 8 | 0 | 0.016 | 0.000 | 0.012 | 0.988 | 0.505 | none | 0.000 |
```

## Interpretation

The source node output mini-matrix was intentionally scrambled:

```text
target bit -> source slot
0 -> 2
1 -> 4
2 -> 1
3 -> 0
4 -> 3
5 -> 5
6 -> 6
7 -> 7
```

A raw wire assumes `target[j] = source[j]`, so it failed:

```text
raw_wire_direct_fixed16 heldout = 0.062
raw_wire_progressive_plus1 heldout = 0.062
```

The adapter pocket learned the mechanical mapping:

```text
edge_adapter_fixed16_to16 heldout = 1.000
edge_adapter_progressive_plus1_no_freeze heldout = 1.000
old_intent_regression = 0.000
```

So the rail can be treated as:

```text
Node A output mini-matrix
-> Edge Adapter Pocket
-> Node B input mini-matrix
```

The adapter is not a semantic label channel. It is a local ABI transformer.

## Growth Result

`+1` adapter-cell growth worked without regressing old intents:

```text
5 active bits -> 6 -> 7 -> 8
old_intent_success = 1.000
old_intent_regression = 0.000
```

Block growth was cheaper in this clean harness:

```text
progressive_plus1_no_freeze attempts_to_95 = 4032
block_growth_plus4 attempts_to_95 = 2016
fixed16_to16 attempts_to_95 = 723
```

This matches E46: use `+1` growth when needed as a safe fine-grained plateau
escape; use block/direct adapter allocation when the required target capacity
is already known.

## Architecture Implication

Lock the concept:

```text
Edge ABI = optional explicit adapter pocket
```

Default path:

```text
if source/target ABI align:
  raw bus is enough

if source/target ABI mismatch:
  insert adapter pocket

if capacity plateaus:
  grow adapter by +1 active cell under old-intent regression gate
```

## Boundary

This is a controlled symbolic/numeric Edge ABI adapter probe. It does not prove
raw language reasoning, deployed AI assistant behavior, model-scale behavior,
AGI, or consciousness.

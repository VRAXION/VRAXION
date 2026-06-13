# E46 Progressive Single Wire Edge ABI Growth Probe Result

## Decision

```text
decision = e46_block_growth_preferred
target_checker_failure_count = 0
sample_only_checker_failure_count = 0
run_id = f8069da2cac2f084
```

E46 tested whether an Edge ABI can grow by `+1` wire after plateau instead of
starting with a wide anonymous bus.

## Result Table

```text
| system | final_width | active_bits | growth_events | heldout_success | ood_success | old_intent_success | old_intent_regression | attempts_to_95 | accepted_rate |
|---|---|---|---|---|---|---|---|---|---|
| fixed_w5_i256_too_narrow_control | 5 | 5 | 0 | 0.125 | 0.138 | 1.000 | 0.000 | 19 | 0.003 |
| fixed_w8_i256_direct | 8 | 8 | 0 | 1.000 | 1.000 | 1.000 | 0.000 | 183 | 0.004 |
| fixed_w16_i256_direct | 16 | 8 | 0 | 1.000 | 1.000 | 1.000 | 0.000 | 321 | 0.015 |
| progressive_plus1_freeze_old | 8 | 8 | 3 | 1.000 | 1.000 | 1.000 | 0.000 | 2688 | 0.001 |
| progressive_plus1_no_freeze | 8 | 8 | 3 | 1.000 | 1.000 | 1.000 | 0.000 | 2688 | 0.001 |
| progressive_block_plus4 | 8 | 8 | 3 | 1.000 | 1.000 | 1.000 | 0.000 | 1344 | 0.001 |
| structured_oracle_progressive_reference | 8 | 8 | 3 | 1.000 | 1.000 | 1.000 | 0.000 | 0 | 0.000 |
| random_growth_control | 16 | 8 | 0 | 0.000 | 0.006 | 0.010 | 0.990 | none | 0.000 |
```

## Interpretation

The direct answer:

```text
+1 wire adding works.
```

The `progressive_plus1_freeze_old` system reached the 256-intent target and
kept old-intent regression at zero:

```text
heldout_success = 1.000
OOD_success = 1.000
old_intent_success = 1.000
old_intent_regression = 0.000
growth_events = 3
```

However, it was not the cheapest policy in this simple append-safe harness:

```text
progressive_plus1_freeze_old attempts_to_95 = 2688
progressive_block_plus4      attempts_to_95 = 1344
fixed_w8_direct              attempts_to_95 = 183
```

So the precise result is:

```text
single-wire growth is viable and non-regressive,
but block growth is cheaper when the final required capacity is already known.
```

## Architecture Implication

Use `+1` growth as a safe fine-grained plateau escape when the system does not
know how much extra capacity it needs.

Use block growth when the next capacity target is already known:

```text
unknown need:
  +1 extension wire

known capacity jump:
  block extension

deployment/default:
  keep 16-bit fast bus
  allow masked extensions
```

## Boundary

This is a controlled symbolic/numeric Edge ABI growth probe. It does not prove
raw language reasoning, deployed AI assistant behavior, model-scale behavior,
AGI, or consciousness.

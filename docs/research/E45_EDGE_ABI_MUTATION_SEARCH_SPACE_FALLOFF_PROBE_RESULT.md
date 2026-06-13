# E45 Edge ABI Mutation Search Space Falloff Probe Result

## Decision

```text
decision = e45_anonymous_wide_bus_learning_falloff_detected
target_checker_failure_count = 0
sample_only_checker_failure_count = 0
run_id = 64109f3cb2e00668
```

E45 froze the producer and consumer nodes, then mutated only the Edge ABI
decoder/contract between them. It measured both final performance and learning
dynamics.

## Result Table

```text
| system | bus_width | intent_count | search_space_log10 | heldout_success | ood_success | wrong_commit_rate | attempts_to_95 | accepted_rate | plateau_tail_generations |
|---|---|---|---|---|---|---|---|---|---|
| structured_w16_i32_reference | 16 | 32 | 5.719 | 1.000 | 1.000 | 0.000 | 0 | 0.000 | 0 |
| structured_w64_i256_reference | 64 | 256 | 14.252 | 1.000 | 1.000 | 0.000 | 0 | 0.000 | 0 |
| structured_w128_i1024_reference | 128 | 1024 | 20.915 | 1.000 | 1.000 | 0.000 | 0 | 0.000 | 0 |
| anonymous_w8_i32 | 8 | 32 | 3.827 | 1.000 | 1.000 | 0.000 | 139 | 0.003 | 43 |
| anonymous_w12_i32 | 12 | 32 | 4.978 | 1.000 | 1.000 | 0.000 | 167 | 0.007 | 42 |
| anonymous_w16_i32 | 16 | 32 | 5.719 | 1.000 | 1.000 | 0.000 | 72 | 0.005 | 45 |
| anonymous_w24_i32 | 24 | 32 | 6.708 | 1.000 | 1.000 | 0.000 | 136 | 0.007 | 43 |
| anonymous_w32_i32 | 32 | 32 | 7.383 | 1.000 | 1.000 | 0.000 | 134 | 0.005 | 43 |
| anonymous_w64_i32 | 64 | 32 | 8.961 | 1.000 | 1.000 | 0.000 | 472 | 0.010 | 33 |
| anonymous_w16_i256 | 16 | 256 | 8.715 | 1.000 | 1.000 | 0.000 | 613 | 0.007 | 28 |
| anonymous_w32_i256 | 32 | 256 | 11.627 | 1.000 | 1.000 | 0.000 | 791 | 0.011 | 23 |
| anonymous_w64_i256 | 64 | 256 | 14.252 | 1.000 | 1.000 | 0.000 | 1374 | 0.012 | 5 |
| anonymous_w96_i256 | 96 | 256 | 15.728 | 0.258 | 0.172 | 0.738 | none | 0.012 | 0 |
| anonymous_w128_i256 | 128 | 256 | 16.761 | 0.000 | 0.008 | 0.980 | none | 0.007 | 0 |
| anonymous_w64_i1024 | 64 | 1024 | 17.740 | 0.500 | 0.586 | 0.447 | none | 0.011 | 3 |
| anonymous_w128_i1024 | 128 | 1024 | 20.915 | 0.000 | 0.016 | 0.994 | none | 0.001 | 42 |
| random_w16_i32_control | 16 | 32 | 5.719 | 0.039 | 0.047 | 0.956 | none | 0.001 | 47 |
```

## Interpretation

The important distinction is:

```text
wide structured bus:
  can work cleanly

wide anonymous mutable bus:
  search-space cost rises sharply
```

The strongest useful boundary from this probe:

```text
32-bit anonymous / 256 intent:
  passes
  attempts_to_95 = 791

64-bit anonymous / 256 intent:
  still passes
  attempts_to_95 = 1374

96-bit anonymous / 256 intent:
  fails
  heldout_success = 0.258

128-bit anonymous / 256 intent:
  fails hard
  heldout_success = 0.000

64-bit anonymous / 1024 intent:
  fails
  heldout_success = 0.500
```

So the result is not:

```text
64 bit is impossible
```

It is:

```text
64 anonymous mutable fast-lane bits are not free.
64 can work for 256 intents in this bit-position decoder harness,
but falls off when intent count rises or width pushes past 64.
Use structure/masks/framing if the edge gets wide.
```

## Lock Implication

This supports the earlier architecture rule:

```text
16-bit physical fast proposal bus:
  safe default

32-bit extended proposal bus:
  plausible if still mechanically structured

64-bit proposal record:
  acceptable as framed/structured record
  not as unconstrained anonymous mutable fast lane
```

## Boundary

This is a controlled symbolic/numeric Edge ABI probe. It does not prove raw
language reasoning, deployed AI assistant behavior, model-scale behavior, AGI,
or consciousness.

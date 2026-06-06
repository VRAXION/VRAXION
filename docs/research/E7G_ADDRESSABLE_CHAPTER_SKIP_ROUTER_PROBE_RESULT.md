# E7G Addressable Chapter Skip Router Probe Result

## Decision

```text
decision = e7g_addressable_chapter_skip_confirmed
best_non_oracle_system = addressable_chapter_router_mutation
deterministic_replay_passed = true
checker_failure_count = 0
```

Artifact root:

```text
target/pilot_wave/e7g_addressable_chapter_skip_router_probe/
```

## Systems Run

```text
sequential_pipe_scan
fixed_short_pipe_router
fused_long_pipe_path_model
addressable_chapter_router_mutation
addressable_router_sparse_call_prior
dense_graph_soft_router_gradient
random_segment_walk_control
oracle_chapter_skip_reference
```

## Mean Evidence Metrics

```text
addressable_chapter_router_mutation    heldout=0.905802 ood=0.905857 route=1.000000 answer=1.000000 steps=3.011 irrelevant=0.000000
addressable_router_sparse_call_prior   heldout=0.905802 ood=0.905857 route=1.000000 answer=1.000000 steps=3.011 irrelevant=0.000000
dense_graph_soft_router_gradient       heldout=0.893222 ood=0.791459 route=0.977652 answer=0.992803 steps=3.105 irrelevant=0.019084
fixed_short_pipe_router                heldout=0.525644 ood=0.538322 route=0.220455 answer=0.662879 steps=3.011 irrelevant=0.128680
fused_long_pipe_path_model             heldout=0.665621 ood=0.526987 route=0.493561 answer=0.781818 steps=3.011 irrelevant=0.084269
oracle_chapter_skip_reference          heldout=0.905802 ood=0.905857 route=1.000000 answer=1.000000 steps=3.011 irrelevant=0.000000
random_segment_walk_control            heldout=0.398911 ood=0.486542 route=0.000379 answer=0.573106 steps=2.992 irrelevant=0.450343
sequential_pipe_scan                   heldout=0.475447 ood=0.457311 route=0.295833 answer=0.744318 steps=10.000 irrelevant=0.000000
```

## Skip Comparison

```text
best_addressable_system = addressable_chapter_router_mutation
addressable_minus_sequential_heldout = +0.430355
addressable_minus_fused_ood = +0.378870
addressable_minus_dense_heldout = +0.012580
sequential_mean_steps = 10.0
addressable_mean_steps = 3.010985
dense_irrelevant_branch_rate = 0.019084
```

## Interpretation

E7G supports the direct-address chapter-skip mechanism on this controlled proxy:

```text
Router -> chapter_id -> Router -> chapter_id -> halt
```

beat:

```text
scan every chapter
fixed local short-pipe routing
fused path memorization
random walk
```

The dense soft router nearly matched heldout usefulness but had a larger OOD drop. That is the important boundary signal: the soft graph-like control can learn much of the visible route pattern, but the hard addressable router remained cleaner and more stable on heldout/OOD/counterfactual/adversarial aggregates.

## What This Does Not Prove

E7G does not prove pocket genesis. Chapter boundaries and chapter IDs were already available as task addresses.

The result says:

```text
if chapters already exist,
direct address + router return + halt is useful and efficient.
```

It does not say:

```text
mutation can discover the chapters themselves.
```

## Next Recommended Probe

```text
E7H_POCKET_GRANULARITY_DISCOVERY_PROBE
```

This should test whether mutation/rollback can discover reusable chapter boundaries from smaller segments using merge/split/freeze/repair mutations, without falling back into dense micro-node routing.

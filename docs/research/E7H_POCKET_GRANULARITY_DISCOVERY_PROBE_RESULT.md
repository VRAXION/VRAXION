# E7H Pocket Granularity Discovery Probe Result

## Decision

```text
decision = e7h_mutation_discovers_reusable_pocket_granularity
best_non_oracle_system = mutation_discovered_pockets
deterministic_replay_passed = true
checker_failure_count = 0
```

Artifact root:

```text
target/pilot_wave/e7h_pocket_granularity_discovery_probe/
```

## Systems Run

```text
atomic_microsegment_router
fixed_human_pockets
fused_long_pipe
mutation_discovered_pockets
discovered_pockets_plus_router
discovered_pockets_plus_limited_repair
dense_graph_control
random_boundary_control
oracle_granularity_reference
```

## Mean Evidence Metrics

```text
atomic_microsegment_router                 heldout=0.680000 ood=0.680000 cf=0.680000 adv=0.680000 route=1.000000 answer=1.000000 steps=5.995 pockets=0.000 avg_size=0.000 reuse=0.000
dense_graph_control                        heldout=0.679417 ood=0.340918 cf=0.535943 adv=0.679545 route=0.998864 answer=1.000000 steps=6.006 pockets=0.000 avg_size=0.000 reuse=0.000
discovered_pockets_plus_limited_repair     heldout=0.812555 ood=0.812648 cf=0.812476 adv=0.812549 route=1.000000 answer=1.000000 steps=2.997 pockets=6.000 avg_size=2.000 reuse=0.500
discovered_pockets_plus_router             heldout=0.812555 ood=0.812648 cf=0.812476 adv=0.812549 route=1.000000 answer=1.000000 steps=2.997 pockets=6.000 avg_size=2.000 reuse=0.500
fixed_human_pockets                        heldout=0.812555 ood=0.812648 cf=0.812476 adv=0.812549 route=1.000000 answer=1.000000 steps=2.997 pockets=6.000 avg_size=2.000 reuse=0.500
fused_long_pipe                            heldout=0.871394 ood=0.493116 cf=0.694952 adv=0.871722 route=0.981061 answer=0.992803 steps=1.000 pockets=0.000 avg_size=0.000 reuse=0.000
mutation_discovered_pockets                heldout=0.812555 ood=0.812648 cf=0.812476 adv=0.812549 route=1.000000 answer=1.000000 steps=2.997 pockets=6.000 avg_size=2.000 reuse=0.500
oracle_granularity_reference               heldout=0.812555 ood=0.812648 cf=0.812476 adv=0.812549 route=1.000000 answer=1.000000 steps=2.997 pockets=6.000 avg_size=2.000 reuse=0.500
random_boundary_control                    heldout=0.236759 ood=0.315557 cf=0.232194 adv=0.234521 route=0.012121 answer=0.577652 steps=5.995 pockets=0.000 avg_size=0.000 reuse=0.000
```

## Granularity Comparison

```text
best_discovered_system = mutation_discovered_pockets
discovered_minus_atomic_heldout = +0.132555
discovered_minus_fixed_heldout = 0.000000
discovered_minus_fused_ood = +0.319532
discovered_minus_dense_ood = +0.471730
discovered_pocket_count = 6.000
average_discovered_pocket_size = 2.000
reuse_count_per_pocket = 0.499558
freeze_survival_score = 1.000000
repair_gain_over_router_heldout = 0.000000
local_repair_use_rate = 0.441288
```

## Interpretation

E7H supports the pocket granularity discovery hypothesis on this controlled proxy.

The task exposed only microsegment paths. The natural pocket IDs were hidden from model input and used only for evaluation/audit. Mutation discovered six reusable pockets with average size 2.0, matching the fixed human pocket scaffold and oracle granularity reference.

The key result is not that discovered pockets beat the fixed human scaffold. They matched it. The important result is:

```text
mutation-discovered pockets > atomic microsegment routing
mutation-discovered pockets > fused long pipe on OOD
mutation-discovered pockets > dense graph control on OOD
mutation-discovered pockets ~= fixed human pocket scaffold
```

The fused long pipe had the highest heldout usefulness, but it collapsed on OOD relative to discovered pockets:

```text
fused_long_pipe OOD = 0.493116
mutation_discovered_pockets OOD = 0.812648
```

That is why the robust decision favors reusable discovered pocket granularity rather than fused path memorization.

## What This Does Not Prove

E7H does not prove raw-world pocket genesis or broad reasoning. Microsegments were already supplied. The result says mutation/rollback can discover reusable intermediate grouping over provided microsegments on this proxy.

It does not prove:

```text
raw input automatically becomes microsegments
large-scale learning will behave the same way
dense neural methods are never useful
```

## Next Recommended Probe

```text
E7I_NOISY_MICROSEGMENT_AND_BOUNDARY_AMBIGUITY_PROBE
```

Stress the result by adding noisy microsegments, ambiguous boundaries, partially overlapping reusable pockets, and decoy pair frequencies. The next falsification should test whether mutation still discovers useful intermediate pockets when the natural boundaries are not clean adjacent pairs.

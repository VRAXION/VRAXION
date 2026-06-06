# E7B Pocket Routing Composition Probe Result

## Decision

```text
decision = e7b_mutation_router_composition_viable
best_non_oracle_system = router_plus_limited_pocket_repair
deterministic_replay_passed = true
checker_failure_count = 0
```

Artifact root:

```text
target/pilot_wave/e7b_pocket_routing_composition_probe/
```

Evidence scale:

```text
seeds = 12
systems = 8
system result rows = 96
GPU gradient device = cuda
CPU mutation workers = 23
```

## Mean Metrics

| system | usefulness | answer | route | composition | adversarial | shortcut | params |
|---|---:|---:|---:|---:|---:|---:|---:|
| monolithic_backprop_model | 0.733490 | 0.638125 | 0.930104 | 0.592604 | 0.702812 | 0.023542 | 6152 |
| monolithic_mutation_model | 0.447240 | 0.503229 | 0.459688 | 0.231563 | 0.429771 | 0.197187 | 876 |
| frozen_pockets_gradient_router | 0.866276 | 0.861458 | 0.905937 | 0.789792 | 0.847333 | 0.038021 | 7048 |
| frozen_pockets_mutation_router | 0.989635 | 0.996771 | 0.982500 | 0.982500 | 0.987917 | 0.003229 | 48 |
| frozen_pockets_binary_router | 0.953073 | 0.982812 | 0.923333 | 0.923333 | 0.951250 | 0.017188 | 43 |
| router_plus_limited_pocket_repair | 0.991042 | 1.000000 | 0.982083 | 0.982083 | 0.991042 | 0.000000 | 48 |
| random_router_control | 0.427865 | 0.695000 | 0.160729 | 0.160729 | 0.420833 | 0.305000 | 0 |
| oracle_pocket_router_reference | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0 |

## Mutation Counts

| system | attempts | accepted | rejected | rollback |
|---|---:|---:|---:|---:|
| monolithic_mutation_model | 72000 | 8266 | 63734 | 63734 |
| frozen_pockets_mutation_router | 72000 | 51020 | 20980 | 20980 |
| frozen_pockets_binary_router | 72000 | 30066 | 41934 | 41934 |
| router_plus_limited_pocket_repair | 72000 | 50714 | 21286 | 21286 |

## Interpretation

The result supports the pocket-routing hypothesis on this controlled proxy:

```text
mutation/rollback did not learn the full monolithic task well
but did learn a compact router over frozen pocket outputs
```

The strongest non-oracle system was `router_plus_limited_pocket_repair`, but the plain `frozen_pockets_mutation_router` was already near ceiling with only 48 parameters.

The binary router also remained viable:

```text
frozen_pockets_binary_router usefulness = 0.953073
route_accuracy = 0.923333
adversarial_usefulness = 0.951250
```

## Leakage And Controls

The random router control did not pass:

```text
random_control_usefulness = 0.427865
random_control_passed = true
```

The adversarial split used misleading route hints. The mutation router stayed high:

```text
mutation_router_adversarial_usefulness = 0.987917
gradient_router_adversarial_usefulness = 0.847333
```

## Boundary

This probe used frozen deterministic symbolic pockets to isolate the routing question. It does not prove that the pockets themselves were learned by mutation. It shows that, once reusable pockets exist, mutation/rollback can learn the switchboard/composition layer far better than monolithic mutation on this proxy.

## Next Recommendation

Run a follow-up that removes the deterministic-pocket shortcut:

```text
E7C_LEARNED_POCKET_ROUTING_BRIDGE
```

Goal:

```text
pretrain pockets separately
freeze or binarize them
then test whether mutation/rollback can still learn the router
and whether limited mutation can repair pocket drift
```

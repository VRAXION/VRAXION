# E7C Learned Pocket Routing Bridge Result

## Decision

```text
decision = e7c_learned_pocket_mutation_router_viable
best_mutation_router_system = router_plus_limited_pocket_repair
deterministic_replay_passed = true
checker_failure_count = 0
```

Artifact root:

```text
target/pilot_wave/e7c_learned_pocket_routing_bridge/
```

Evidence scale:

```text
seeds = 12
systems = 9
system result rows = 108
GPU pocket training = cuda
GPU gradient lane = cuda
CPU mutation workers = 23
```

## Learned Pocket Quality

```text
mean_candidate_answer_accuracy = 0.980149
mean_branch_accuracy = 1.000000
mean_oracle_learned_route_answer_ceiling = 0.977014
learned_pocket_ceiling_usefulness = 0.985917
```

The learned pockets were strong enough that the run was not a pocket-quality bottleneck.

## Mean Metrics

| system | usefulness | answer | route | composition | adversarial | shortcut | params |
|---|---:|---:|---:|---:|---:|---:|---:|
| monolithic_backprop_model | 0.729698 | 0.628125 | 0.935208 | 0.588750 | 0.698958 | 0.041146 | 6152 |
| monolithic_mutation_model | 0.445578 | 0.510625 | 0.443750 | 0.233021 | 0.437458 | 0.208125 | 876 |
| learned_pockets_gradient_router | 0.862120 | 0.854167 | 0.907292 | 0.783229 | 0.831354 | 0.055208 | 7048 |
| learned_pockets_mutation_router | 0.975500 | 0.975000 | 0.982500 | 0.960833 | 0.974958 | 0.025000 | 48 |
| learned_binary_pockets_mutation_router | 0.947104 | 0.963333 | 0.936875 | 0.916875 | 0.943896 | 0.036667 | 43 |
| router_plus_limited_pocket_repair | 0.980708 | 0.976042 | 0.991875 | 0.970208 | 0.975792 | 0.023958 | 48 |
| random_router_control | 0.426839 | 0.697083 | 0.157500 | 0.154479 | 0.414104 | 0.302917 | 0 |
| oracle_learned_pocket_router_reference | 0.985917 | 0.978333 | 1.000000 | 0.978333 | 0.982667 | 0.021667 | 0 |
| oracle_symbolic_reference | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.021667 | 0 |

## Mutation Counts

| system | attempts | accepted | rejected | rollback |
|---|---:|---:|---:|---:|
| monolithic_mutation_model | 72000 | 9207 | 62793 | 62793 |
| learned_pockets_mutation_router | 72000 | 49847 | 22153 | 22153 |
| learned_binary_pockets_mutation_router | 72000 | 32025 | 39975 | 39975 |
| router_plus_limited_pocket_repair | 72000 | 51118 | 20882 | 20882 |

## Interpretation

E7C confirms the E7B pocket-routing result under a stronger condition:

```text
the pocket outputs were learned by separate frozen pocket models,
not hand-coded symbolic pocket outputs.
```

The learned-pocket mutation router nearly matched the learned-pocket oracle ceiling:

```text
oracle learned-pocket ceiling = 0.985917
router_plus_limited_pocket_repair = 0.980708
learned_pockets_mutation_router = 0.975500
```

Monolithic mutation remained weak:

```text
monolithic_mutation_model = 0.445578
```

This supports the hypothesis that mutation/rollback is much more effective as a compact router/switchboard learner over reusable pockets than as a monolithic full-task learner.

## Binary Branch

The learned binary-pocket mutation router also remained viable:

```text
learned_binary_pockets_mutation_router = 0.947104
route_accuracy = 0.936875
adversarial_usefulness = 0.943896
```

It did not beat the float learned-pocket router, but it stayed strong enough to keep the binary routing branch alive.

## Leakage And Controls

```text
random_control_usefulness = 0.426839
random_control_passed = true
hidden_correct_route_index_used_as_input = false
route_name_or_index_leakage_claim = false
deterministic_replay_passed = true
```

## Boundary

This remains a controlled symbolic/numeric proxy. It does not prove open-ended reasoning. It does show that the pocket-routing result survives the move from hand-coded pocket outputs to separately learned frozen pocket outputs.

## Next Recommendation

Run:

```text
E7D_LEARNED_POCKET_DRIFT_AND_REPAIR_PROBE
```

Goal:

```text
corrupt or quantize learned pocket outputs/models,
then test whether router mutation alone can adapt,
or whether limited pocket repair is necessary.
```

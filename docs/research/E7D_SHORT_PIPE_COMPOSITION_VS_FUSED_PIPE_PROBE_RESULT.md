# E7D Short-Pipe Composition Vs Fused-Pipe Probe Result

## Decision

```text
decision = e7d_short_pipe_router_flow_preferred
best_non_oracle_system = short_pipe_router_composition
deterministic_replay_passed = true
checker_failure_count = 0
```

Artifact root:

```text
target/pilot_wave/e7d_short_pipe_composition_vs_fused_pipe_probe/
```

Evidence scale:

```text
seeds = 12
systems = 9
system result rows = 108
CPU mutation jobs = 60
GPU gradient lane = cuda
deterministic replay = hash match
```

## Mean Metrics

| system | usefulness | step-penalized | OOD | adversarial | route | shortcut | params | steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| short_pipe_router_composition | 0.964069 | 0.952069 | 0.954759 | 0.969148 | 0.948958 | 0.019560 | 125 | 3.0 |
| router_plus_limited_pocket_repair | 0.934426 | 0.922426 | 0.939426 | 0.925667 | 0.906481 | 0.035301 | 125 | 3.0 |
| short_pipe_no_router_between | 0.818069 | 0.812069 | 0.825611 | 0.802556 | 0.737847 | 0.095023 | 125 | 2.0 |
| fused_long_pipe_gradient_router | 0.612336 | 0.612336 | 0.366000 | 0.646250 | 0.573843 | 0.136111 | 22107 | 1.0 |
| fused_long_pipe_mutation_router | 0.343417 | 0.343417 | 0.331074 | 0.329426 | 0.063194 | 0.353009 | 12525 | 1.0 |
| monolithic_matrix_core_gradient | 0.531094 | 0.531094 | 0.279333 | 0.611787 | 0.595833 | 0.133333 | 17307 | 1.0 |
| monolithic_mutation_model | 0.285409 | 0.285409 | 0.260838 | 0.297069 | 0.080787 | 0.333565 | 3015 | 1.0 |
| random_router_control | 0.325495 | 0.325495 | 0.344148 | 0.285370 | 0.038773 | 0.363889 | 0 | 1.0 |
| oracle_short_pipe_reference | 1.000000 | 0.988000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0 | 3.0 |

## Topology Result

```text
short_best_system = short_pipe_router_composition
fused_best_system = fused_long_pipe_gradient_router
short_minus_fused_usefulness = +0.351734
short_minus_fused_ood_usefulness = +0.588759
short_minus_fused_params = -21982
router_between_gain_over_no_router = +0.146000
```

The short-pipe router generalized to held-out AB compositions:

```text
short_pipe_router_composition OOD usefulness = 0.954759
fused_long_pipe_gradient_router OOD usefulness = 0.366000
fused_long_pipe_gradient_router OOD route accuracy = 0.000000
```

This is the decisive signal: the fused long-pipe systems learned seen composition structure but did not transfer to unseen pair compositions, while the short-pipe router reused primitive pipes.

## Mutation Counts

| system | attempts | accepted | rejected | rollback |
|---|---:|---:|---:|---:|
| fused_long_pipe_mutation_router | 31200 | 2115 | 29085 | 29085 |
| monolithic_mutation_model | 31200 | 3742 | 27458 | 27458 |
| router_plus_limited_pocket_repair | 31200 | 2861 | 28339 | 28339 |
| short_pipe_no_router_between | 31200 | 1679 | 29521 | 29521 |
| short_pipe_router_composition | 31200 | 2788 | 28412 | 28412 |

## Interpretation

E7D supports the short-pipe flow-router topology on this proxy:

```text
Router -> primitive pipe A -> Router -> primitive pipe B
```

beat:

```text
fused AB pipe library
monolithic gradient model
monolithic mutation model
no-router-between control
```

The no-router control still worked partially, but adding router return after the first pipe produced a large gain. That means the intermediate flow-state/branch feedback mattered; the result was not just a static two-token lookup.

## Boundary

This remains a controlled symbolic/numeric topology probe. It does not prove broad reasoning, language reasoning, consciousness, or model-scale behavior.

## Next Recommendation

Run a drift/repair follow-up:

```text
E7E_FLOW_PIPE_DRIFT_AND_ROUTER_REPAIR_PROBE
```

Goal:

```text
corrupt or quantize individual short pipes,
then test whether router mutation alone can route around damage,
or whether limited pocket/pipe repair is needed.
```

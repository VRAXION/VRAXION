# E7E Flow-Pipe Drift And Router Repair Probe Result

## Decision

```text
decision = e7e_router_plus_limited_repair_preferred
best_non_oracle_system = router_plus_limited_pipe_repair
deterministic_replay_passed = true
checker_failure_count = 0
```

Artifact root:

```text
target/pilot_wave/e7e_flow_pipe_drift_and_router_repair_probe/
```

Evidence scale:

```text
seeds = 12
systems = 9
system result rows = 108
CPU mutation jobs = 36
GPU gradient lane = cuda
deterministic replay = hash match
```

## Mean Metrics

| system | usefulness | answer | semantic route | OOD | adversarial | damage hit | routearound | repair use | params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| router_plus_limited_pipe_repair | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 0.408333 | 80 |
| oracle_repair_reference | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 0 |
| oracle_routearound_reference | 0.854271 | 0.834838 | 0.928125 | 0.871852 | 0.832824 | 0.356250 | 1.000000 | 0.000000 | 0 |
| router_routearound_mutation_only | 0.829671 | 0.826042 | 0.907986 | 0.845269 | 0.807333 | 0.396181 | 0.935888 | 0.000000 | 70 |
| monolithic_gradient_drift_adapter | 0.666597 | 0.578588 | 0.648611 | 0.538791 | 0.695678 | 0.346412 | 0.973611 | 0.000000 | 31798 |
| damaged_primary_no_adaptation | 0.550940 | 0.636227 | 1.000000 | 0.565625 | 0.526883 | 0.961921 | 0.059506 | 0.000000 | 0 |
| fused_long_pipe_repair_mutation | 0.475338 | 0.997106 | 0.056481 | 0.474230 | 0.476306 | 0.955324 | 0.038789 | 0.000000 | 12525 |
| fused_long_pipe_gradient_adapter | 0.441307 | 0.587847 | 0.630440 | 0.293167 | 0.472861 | 0.947454 | 0.075226 | 0.000000 | 23323 |
| random_route_control | 0.370531 | 0.630093 | 0.041667 | 0.407386 | 0.348198 | 0.745718 | 0.256204 | 0.000000 | 0 |

## Repair Comparison

```text
routearound_gain_over_damaged = +0.278732
repair_gain_over_routearound = +0.170329
repair_gain_over_damaged = +0.449060
limited_repair_to_oracle_gap = 0.000000
```

## Interpretation

E7E supports a stronger version of the E7D topology:

```text
short-pipe router is useful under drift,
but router + limited local pipe repair is preferred.
```

The route-around-only router did real work:

```text
damaged_primary_no_adaptation = 0.550940
router_routearound_mutation_only = 0.829671
```

So the router can avoid many damaged pipes when a clean backup exists. But route-around alone was bounded by cases where both physical variants of a semantic primitive were corrupted:

```text
oracle_routearound_reference = 0.854271
router_routearound_mutation_only = 0.829671
```

Limited pipe repair closed that remaining gap:

```text
router_plus_limited_pipe_repair = 1.000000
oracle_repair_reference = 1.000000
```

The fused long-pipe repair branch did not become robust:

```text
fused_long_pipe_repair_mutation = 0.475338
fused_long_pipe_gradient_adapter = 0.441307
```

This says the repairable short-pipe topology was much more effective than trying to repair or adapt fused AB circuits on this proxy.

## Important Boundary

This does not prove that local repair is free or easy in a broader system. The final E7E runner uses an explicit repair-toggle mutation so the experiment can actually test the value of limited repair. The result shows that when local pipe repair is available, it is highly valuable under this drift profile.

## Current Architecture Implication

The best current flow picture is:

```text
Router -> short pipe -> Router -> short pipe
with:
  route-around for damaged but redundant pipes
  limited local repair when no clean route-around path exists
```

## Next Recommendation

Run:

```text
E7F_REPAIR_BUDGET_AND_NOISY_HEALTH_SIGNAL_PROBE
```

Goal:

```text
make repair non-free:
  limited repair budget
  noisy/incomplete health signals
  delayed validation feedback
  moving drift profile

Then test whether router+repair still wins,
or whether route-around-only becomes preferable under real repair cost.
```

# E39B Router Discovers Location And Scale Probe Result

Status: complete.

Decision:

```text
e39b_router_discovers_location_and_scale_confirmed
```

Run root:

```text
target/pilot_wave/e39b_router_discovers_location_and_scale_probe
```

Artifact sample:

```text
docs/research/artifact_samples/e39b_router_discovers_location_and_scale_probe
```

Checker:

```text
target_checker_failure_count = 0
sample_only_checker_failure_count = 0
deterministic_replay_passed = true
```

## Result

Official evidence run:

| System | Exact | Cell accuracy | Read spread | Write spread | Scan cells |
|---|---:|---:|---:|---:|---:|
| `oracle_location_scale_reference` | 1.000000 | 1.000000 | 0.078147 | 0.078147 | 0.0 |
| `origin_bound_router` | 0.008333 | 0.948085 | 0.062500 | 0.062500 | 0.0 |
| `mutated_location_router` | 0.322222 | 0.968332 | 0.479867 | 0.062500 | 111.6 |
| `mutated_location_plus_scale_router` | 0.997222 | 0.999859 | 0.457937 | 0.078147 | 99.7 |
| `scan_all_windows_control` | 0.997222 | 0.999859 | 1.000000 | 0.078147 | 264.0 |
| `full_flow_painter_control` | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 256.0 |
| `random_location_scale_control` | 0.008333 | 0.948085 | 0.072678 | 0.072678 | 0.0 |

Multi-seed confirm:

| Run | Decision | Primary exact | Primary write spread | Primary read spread |
|---|---|---:|---:|---:|
| primary `39021` | `e39b_router_discovers_location_and_scale_confirmed` | 0.997222 | 0.078147 | 0.457937 |
| confirm `39022` | `e39b_router_discovers_location_and_scale_confirmed` | 0.998611 | 0.075543 | 0.471604 |
| confirm `39023` | `e39b_router_discovers_location_and_scale_confirmed` | 0.997222 | 0.072743 | 0.440652 |
| confirm `39024` | `e39b_router_discovers_location_and_scale_confirmed` | 0.998611 | 0.072982 | 0.446658 |

## Learned Router State

The primary learned mutation router started from:

```text
location_policy = marker_scan
scale_policy = marker_map
marker_to_scale = {7: 4, 8: 4, 9: 4}
op = copy
require_guard = false
```

and ended at:

```text
location_policy = marker_scan
scale_policy = marker_map
marker_to_scale = {7: 2, 8: 4, 9: 6}
op = invert
require_guard = true
```

Mutation stats:

```text
accepted = 199
rejected = 505
rollback = 505
```

## Interpretation

E39B separates four cases:

```text
origin-bound router:
  writes locally, but at the wrong place.

location-only router:
  finds where to write, but fixed scale fails variable-size patches.

location+scale router:
  infers visible marker/guard protocol, local scale, and local call target.

scan-all/full-flow controls:
  can solve by excessive reading or writing, but have worse footprint economics.
```

This confirms the E39A call shape, but removes the oracle coordinate scaffold:

```text
CALL(pocket_id, location, scale)
```

can be supplied by a learned/mutated router from visible Flow Field evidence,
not only by the row generator.

Boundary: E39B is a controlled spatial Flow-grid proxy. It does not prove raw
language reasoning, AGI, consciousness, deployed-model behavior, or model-scale
behavior.

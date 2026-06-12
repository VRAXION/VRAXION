# E39B Router Discovers Location And Scale Probe Contract

Milestone:

```text
E39B_ROUTER_DISCOVERS_LOCATION_AND_SCALE_PROBE
```

## Purpose

E39A showed that a local pocket can solve spatial Flow-grid transforms when the
call includes explicit:

```text
CALL(pocket_id, location, scale)
```

E39B removes that scaffold. The router must infer `location` and `scale` from
visible Flow-grid evidence before calling the local pocket.

This is not a new language claim, model-scale claim, AGI claim, or deployment
claim. It is a controlled spatial Flow-grid proxy.

## Systems

```text
oracle_location_scale_reference
origin_bound_router
mutated_location_router
mutated_location_plus_scale_router
scan_all_windows_control
full_flow_painter_control
random_location_scale_control
```

The oracle reference is ineligible for scientific comparison. Valid primary
systems must not read hidden row coordinates.

## Task

Each row contains a 2D Flow grid with:

```text
background values
visible marker cell
two guard cells
local target patch
optional decoy markers
```

The marker/guard pattern is visible in the Flow grid. The hidden row target is
used only for evaluation. The local operation is an invert transform on the
target patch.

The router must infer:

```text
patch location
patch scale
operation
```

from the visible grid protocol.

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
router_discovery_report.json
system_results.json
footprint_report.json
mutation_report.json
row_level_results.jsonl
footprint_frames.jsonl
aggregate_metrics.json
deterministic_replay.json
decision.json
summary.json
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
report.md
```

Sample pack:

```text
docs/research/artifact_samples/e39b_router_discovers_location_and_scale_probe/
```

## Metrics

```text
exact_rate
cell_accuracy
read_spread_ratio
write_spread_ratio
changed_spread_ratio
scan_cell_count_mean
illegal_write_count_mean
missed_target_write_count_mean
accepted/rejected/rollback mutation counts
parameter diff/hash
deterministic replay hash match
```

## Decision Labels

```text
e39b_router_discovers_location_and_scale_confirmed
e39b_location_only_sufficient
e39b_scan_all_required
e39b_full_flow_required
e39b_invalid_footprint_artifact_detected
```

Confirmed requires:

```text
mutated_location_plus_scale_router exact >= 0.95
mutated_location_plus_scale_router write_spread <= 0.12
mutated_location_router exact < 0.95
origin/random controls stay low
scan_all_windows_control solves but with high scan cost
full_flow_painter_control solves but writes diffusely
checker failure_count = 0
sample-only checker passes
deterministic replay passes
```

## Hard Requirements

No gradient descent, optimizer, backprop, direct solver, or hidden oracle access
inside valid primary systems. Oracle/direct target access is allowed only in the
named ineligible reference/control systems.

Long-ish runs must write progress and partial artifacts during the run, not only
at the end.

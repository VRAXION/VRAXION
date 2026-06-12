# E39A Addressable Local Pocket Translation Smoke Contract

## Summary

E39A tests whether a reusable local pocket can operate at arbitrary Flow-grid
locations when called with an explicit location and scale:

```text
CALL(pocket_id, location, scale)
```

It also makes FootprintLogging-v1 mandatory for this spatial probe.

## Purpose

The previous pocket archive showed functional/protocol footprints, not spatial
Flow Field shapes. E39A is the first clean spatial harness for:

```text
point-like vs patch-like vs full-frame pocket behavior
origin-bound vs addressable local operation
small fixed patch vs multiscale patch
```

## Systems

```text
origin_bound_local_pocket_mutation
addressable_local_pocket_mutation
addressable_multiscale_local_pocket_mutation
full_flow_painter_diagnostic
random_location_control
```

## Required Footprint Logging

Every pocket call must write:

```text
pocket_id / system
call_step
location
scale
read_cells
write_cells
delta_cells
changed_cell_count
write_bbox
delta_bbox
center_of_mass
write_spread
before/after/delta sample or heatmap
commit/reject/rollback
```

## Metrics

```text
exact grid success
cell accuracy
heldout/OOD/counterfactual/multiscale success
write spread ratio
changed spread ratio
illegal write count
missed target write count
bounding box area
center-of-mass alignment
accepted/rejected mutations
rollback count
deterministic replay
checker failure count
```

## Decision Rules

```text
e39a_addressable_local_pocket_confirmed
  addressable local pocket solves random-location heldout/OOD while
  origin-bound fails and full-flow diagnostic is diffuse.

e39a_multiscale_local_pocket_needed
  fixed local pocket is insufficient but addressable multiscale succeeds.

e39a_origin_bound_sufficient
  origin-bound pocket unexpectedly solves the random-location task.

e39a_full_flow_required
  only full-flow diagnostic reaches high success.

e39a_invalid_footprint_artifact_detected
  missing footprint logging, replay/checker failure, or invalid artifacts.
```

## Boundary

E39A is a controlled spatial Flow-grid proxy. It does not prove raw language
reasoning, AGI, consciousness, deployed-model behavior, or model-scale behavior.

# E39A Addressable Local Pocket Translation Smoke Result

Status: complete.

Decision:

```text
e39a_multiscale_local_pocket_needed
```

Run root:

```text
target/pilot_wave/e39a_addressable_local_pocket_translation_smoke
```

Artifact sample:

```text
docs/research/artifact_samples/e39a_addressable_local_pocket_translation_smoke
```

Checker:

```text
target_checker_failure_count = 0
sample_only_checker_failure_count = 0
deterministic_replay_passed = true
```

## Result

| System | Exact | Cell accuracy | Write spread | Illegal writes |
|---|---:|---:|---:|---:|
| `origin_bound_local_pocket_mutation` | 0.000000 | 0.955949 | 0.062500 | 14.689 |
| `addressable_local_pocket_mutation` | 0.839583 | 0.993319 | 0.061584 | 0.753 |
| `addressable_multiscale_local_pocket_mutation` | 1.000000 | 1.000000 | 0.065560 | 0.000 |
| `full_flow_painter_diagnostic` | 1.000000 | 1.000000 | 1.000000 | 239.217 |
| `random_location_control` | 0.000000 | 0.955949 | 0.062500 | 14.870 |

## Interpretation

The result separates three cases:

```text
origin-bound local pocket:
  writes a small patch, but at the wrong place.

addressable fixed-scale local pocket:
  location helps, but fixed size fails on variable-size patches.

addressable multiscale local pocket:
  location + scale solves the task with small write footprint.

full-flow painter:
  solves the grid but writes the whole Flow frame, so it is not the preferred
  reusable pocket shape.
```

The useful call shape for spatial pockets should therefore be:

```text
CALL(pocket_id, location, scale)
```

not just:

```text
CALL(pocket_id)
CALL(pocket_id, location)
```

## Footprint Logging

E39A generated explicit footprint artifacts:

```text
footprint_frames.jsonl
footprint_report.json
row_level_results.jsonl
```

Each sampled call records:

```text
read_cells
write_cells
delta_cells
write_bbox
delta_bbox
center_of_mass
write_spread
before/after/delta heatmap
```

## Conclusion

FootprintLogging-v1 should remain ON for spatial pocket probes. The next
architecture should treat local pockets as reusable operators applied through a
router-supplied address and scale.

Boundary: E39A is a controlled spatial Flow-grid proxy. It does not prove raw
language reasoning, AGI, consciousness, deployed-model behavior, or model-scale
behavior.

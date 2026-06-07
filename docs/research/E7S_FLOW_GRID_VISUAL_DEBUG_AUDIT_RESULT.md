# E7S FlowGrid Visual Debug Audit Result

Run root:

```text
target/pilot_wave/e7s_flow_grid_visual_debug_audit
```

## Status

```text
status = complete
decision = e7s_flow_grid_visual_debug_ready
checker_failure_count = 0
deterministic_replay_passed = true
```

## Source

```text
source = target/pilot_wave/e7r_numeric_pocket_masked_flow_io_contract_probe
source_type = e7r_artifact_plus_visualization_sample
```

E7R did not contain complete frame-level Flow traces, so E7S used the E7R
artifact contracts and aggregate metrics plus a deterministic visualization
sample. This does not change the E7R scientific result.

## Visual Output

```text
html = target/pilot_wave/e7s_flow_grid_visual_debug_audit/flow_grid_visualizer.html
json = target/pilot_wave/e7s_flow_grid_visual_debug_audit/flow_grid_frames.json
jsonl = target/pilot_wave/e7s_flow_grid_visual_debug_audit/flow_grid_frames.jsonl
grid = 5x8
flow_dim = 40
example_count = 16
systems = current_untyped_flow_baseline, anonymous_fixed_mask_contract,
          anonymous_shuffled_mask_contract, learned_mask_contract,
          oracle_mask_reference
```

## Mean Visual Pattern

```text
system                                  source_useful  write_spread  delta_mag  preserve_corrupt
current_untyped_flow_baseline           0.650139       0.964063      0.112458   0.000000
anonymous_fixed_mask_contract           0.770972       0.063281      0.018868   0.000000
anonymous_shuffled_mask_contract        0.768194       0.063281      0.019538   0.000000
learned_mask_contract                   0.812083       0.037500      0.016734   0.000000
oracle_mask_reference                   0.992917       0.013281      0.013281   0.000000
```

## Interpretation

The visual audit is ready. It shows the E7R improvement in a way that matches
the hypothesis:

```text
untyped Flow = broad RAM smear
anonymous masks = constrained IO boundary
learned sparse mask = most compact non-oracle boundary
```

No masked preserve/write corruption pattern appeared in the deterministic
visualization sample. The point of the artifact is inspection, not a new model
score.

Recommended next step:

```text
Use the FlowGrid audit to inspect failed E7Q/E7R-style examples before designing
E7S2 or E7T. The likely next scientific question is whether router+pocket
co-training can keep the same mechanical IO hygiene without private protocols.
```

## Boundary

E7S only visualizes numeric pocket Flow[D] IO behavior in a controlled
pocket-router proxy. It does not prove raw-language learning, AGI,
consciousness, or model-scale behavior.

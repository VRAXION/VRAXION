# E7S FlowGrid Visual Debug Audit Contract

## Purpose

`E7S_FLOW_GRID_VISUAL_DEBUG_AUDIT` follows E7R.

Core question:

```text
Can we visually inspect numeric pocket Flow[D] as shared RAM and see what each
pocket call reads, writes, preserves, changes, or corrupts?
```

This is a visualization/debug tool, not a new architecture or capability
probe.

## Source

Prefer E7R artifacts:

```text
target/pilot_wave/e7r_numeric_pocket_masked_flow_io_contract_probe/
```

E7R did not save complete frame-level Flow traces, so E7S may create a
deterministic visualization sample from the E7R task generator, E7R mask
contracts, and E7R aggregate metrics. That sample must be marked as
visualization replay/sample and must not alter E7R result claims.

## Visual Model

```text
Flow[D] = shared RAM / working state
Pocket = operation that reads/writes Flow[D]
Router = chooses next pocket call
Mask = mechanical memory protection
Visualizer = microscope
```

For E7R:

```text
D = 40
grid = 5x8
```

## Systems

```text
current_untyped_flow_baseline
anonymous_fixed_mask_contract
anonymous_shuffled_mask_contract
learned_mask_contract
oracle_mask_reference
```

## Required Visual Layers

```text
cell value heatmap
read cells outline
write cells outline
preserve cells outline
changed cells highlighted
illegal write highlighted
preserve corruption highlighted
pocket call sequence timeline
final answer correctness
split/family tag
```

The UI may use mechanical labels:

```text
read
write
preserve
changed
delta
```

It must not use semantic lane concepts such as:

```text
truth slot
memory slot
confidence slot
result slot
```

## Required Artifacts

```text
backend_manifest.json
flow_grid_frames.json
flow_grid_frames.jsonl
flow_grid_visualizer.html
aggregate_metrics.json
decision.json
summary.json
report.md
deterministic_replay.json
progress.jsonl
```

The HTML report must be a single self-contained file with no external CDN or
network dependency.

## Metrics

```text
write spread
preserve corruption
write mask violation
delta magnitude
flow drift per step
next-pocket input compatibility proxy
route steps
answer correctness
composition success/failure
mask sparsity
lane shuffle robustness if E7R artifact is available
```

## Decision Labels

```text
e7s_flow_grid_visual_debug_ready
e7s_flow_grid_visual_debug_sample_only
e7s_flow_grid_detected_io_corruption_pattern
e7s_flow_grid_visual_debug_blocked
```

## Guardrails

```text
no training
no model changes
no semantic lane labels as model input
no hidden answer/route leakage claims
no AGI/consciousness/raw-language/model-scale claims
deterministic replay required
checker failure_count must be 0
```

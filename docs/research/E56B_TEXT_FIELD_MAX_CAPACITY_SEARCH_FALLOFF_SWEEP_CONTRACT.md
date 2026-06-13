# E56B Text Field Max Capacity Search Falloff Sweep Contract

## Purpose

E56B follows E56A. It does not ask whether Text Field works. It asks how large
the Text Field should be before final runtime lock.

Core question:

```text
What is the largest Text Field capacity that remains worth it under a 1x-3x
training/search slowdown budget?
```

## Boundary

This is a deterministic capacity/search-cost sweep. It does not claim raw
language reasoning, AGI, consciousness, deployment quality, or model-scale
behavior.

## Configurations

```text
fast_default_4x128_o32
normal_4x256_o64
gate_edge_5x256_o64
max_v1_8x256_o64
wide_4x512_o128
wide_8x512_o128
oversize_8x1024_o256
```

## Metrics

```text
success
trace_exact
false_commit_rate
boundary_failure_rate
work_bytes
unique_coverage
slowdown_vs_fast_default
attempts_to_95
hardware_bottleneck_predicted
```

## Slowdown Gate

```text
accepted_slowdown_range = 1x-3x
```

The selected max must:

```text
success >= 0.95
slowdown_vs_fast_default <= 3.0
false_commit_rate <= 0.03
```

## Decisions

```text
e56b_text_field_max_v1_selected
e56b_fast_default_sufficient
e56b_extended_capacity_useful_within_3x
e56b_no_clean_capacity_within_3x_gate
e56b_search_space_falloff_after_max_v1
e56b_hardware_bottleneck_before_search_falloff
e56b_invalid_artifact_detected
```

## Required Artifacts

```text
backend_manifest.json
capacity_sweep_manifest.json
row_level_results.jsonl
capacity_results.json
stage_metrics.json
system_results.json
search_falloff_report.json
hardware_cost_report.json
recommendation_report.json
aggregate_metrics.json
decision.json
summary.json
deterministic_replay.json
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
report.md
```

## Hard Requirements

```text
row-level eval
deterministic replay
no gradient descent
no optimizer/backprop
target checker failure_count = 0
sample-only checker passes
```

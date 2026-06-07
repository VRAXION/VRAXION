# E7Y Natural Output Bundle Width Audit Contract

## Purpose

`E7Y_NATURAL_OUTPUT_BUNDLE_WIDTH_AUDIT` follows E7X.

Core question:

```text
What anonymous output width does a numeric pocket naturally need when composed
through the shared Flow/RAM router?
```

E7X showed that simple one-value write calibration did not close the oracle
gap. E7Y tests whether the issue is that a pocket needs a small multi-channel
write bundle instead of one output cell.

## Systems

```text
single_value_write_baseline
output_bundle_N2
output_bundle_N3
output_bundle_N4
output_bundle_N5
output_bundle_N6
output_bundle_N8
output_bundle_N12
oracle_write_reference
dense_graph_danger_control
```

The bundle channels are anonymous. They are deterministic RAM cells with no
semantic lane labels and no runtime-random placement.

## Method

For each output width `N`:

```text
train pocket output head with N anonymous channels
map N output channels to N deterministic RAM cells
compose through the existing router/route rows
evaluate heldout/OOD/counterfactual/adversarial row-level metrics
```

The first channel is the existing result cell. Additional channels are
anonymous support channels trained against deterministic canonical local-state
projections. The model never receives semantic names such as confidence,
memory, truth, or answer lanes.

## Metrics

```text
composition usefulness
answer accuracy
route accuracy
OOD/counterfactual/adversarial usefulness
output bundle width
RAM cells used
write spread
output channel correlation
output channel redundancy
oracle bundle similarity
bundle MAE to oracle
next-pocket input compatibility
plateau point
gap to oracle
deterministic replay
checker failure_count
```

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
pocket_training_report.json
bundle_contract_report.json
output_width_curve_report.json
channel_morphology_report.json
oracle_bundle_similarity_report.json
ram_bundle_frame_report.json
dense_graph_control_report.json
system_results.json
aggregate_metrics.json
decision.json
summary.json
report.md
deterministic_replay.json
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
```

## Decision Labels

```text
e7y_natural_output_bundle_width_detected
e7y_single_output_cell_sufficient
e7y_large_output_bundle_required
e7y_output_bundle_width_not_sufficient
e7y_graph_soup_regression_detected
```

## Guardrails

```text
no semantic labels
no new router
no runtime-random output placement
oracle only as reference
dense graph only as danger/control
real row-level eval required
deterministic replay required
checker failure_count must be 0
```

## Boundary

E7Y is a controlled numeric Flow/RAM output-width diagnostic. It does not prove
raw-language learning, AGI, consciousness, or model-scale behavior.

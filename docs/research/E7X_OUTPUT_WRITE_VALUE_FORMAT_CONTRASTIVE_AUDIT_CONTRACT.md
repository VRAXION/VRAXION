# E7X Output Write Value-Format Contrastive Audit Contract

## Purpose

`E7X_OUTPUT_WRITE_VALUE_FORMAT_CONTRASTIVE_AUDIT` follows E7W.

Core question:

```text
What is different about the RAM write value when numeric pocket composition
succeeds versus fails?
```

E7X is a diagnostic value-format audit. It does not change router architecture,
does not add semantic lanes, and does not move to image or language tasks.

## Systems

```text
baseline_real_write
oracle_write_reference
affine_calibrated_write
monotonic_calibrated_write
zscore_normalized_write
codebook_write
sign_or_quantized_write
residual_delta_write
router_integrated_write
```

`oracle_write_reference` is diagnostic only. Other transforms are fit from
training split pocket-write pairs and evaluated row-level on heldout/OOD/
counterfactual/adversarial splits.

## Metrics

```text
composition usefulness
answer accuracy
OOD/counterfactual/adversarial usefulness
oracle write similarity
cell-wise correlation with oracle write
cosine similarity
mean absolute error to oracle write
scale ratio
bias offset
value range
saturation rate
sign mismatch rate
entropy / effective value levels
noise floor
delta magnitude
next-pocket input compatibility
per-pocket write morphology
deterministic replay
checker failure_count
```

## Visual/Debug Artifacts

```text
write_histogram_report.json
oracle_real_scatter_report.json
ram_grid_frame_report.json
write_morphology_report.json
top_failing_rows_report.json
```

These are data artifacts for visualization/debugging, not new evidence claims.

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
pocket_training_report.json
read_map_report.json
write_transform_report.json
write_morphology_report.json
write_histogram_report.json
oracle_real_scatter_report.json
ram_grid_frame_report.json
top_failing_rows_report.json
system_results.json
deterministic_replay.json
aggregate_metrics.json
decision.json
summary.json
report.md
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
```

## Decision Labels

```text
e7x_output_scale_bias_calibration_bottleneck
e7x_output_nonlinear_calibration_bottleneck
e7x_canonical_value_code_required
e7x_delta_write_format_required
e7x_flow_integrator_required
e7x_output_value_format_not_sufficient
```

## Guardrails

```text
no semantic labels
no new router
oracle only as diagnostic/reference
no hardcoded oracle leakage in learned transforms
real row-level eval required
deterministic replay required
checker failure_count must be 0
```

## Boundary

E7X only audits output write value format in a controlled numeric pocket-router
proxy. It does not prove raw-language learning, AGI, consciousness, or
model-scale behavior.

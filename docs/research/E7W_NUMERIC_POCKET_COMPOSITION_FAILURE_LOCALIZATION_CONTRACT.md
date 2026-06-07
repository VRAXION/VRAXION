# E7W Numeric Pocket Composition Failure Localization Contract

## Purpose

`E7W_NUMERIC_POCKET_COMPOSITION_FAILURE_LOCALIZATION` follows E7V.

Core question:

```text
Where exactly does numeric pocket composition fail?
```

E7W is a diagnostic ladder. It does not introduce a new architecture and does
not claim a new capability. It localizes whether the remaining gap to oracle is
caused by intermediate Flow/RAM drift, a specific pocket, read context, write
contract, output calibration, or flow integration.

## Systems

```text
baseline_best_current
oracle_route_only
oracle_intermediate_state_after_each_pocket
one_real_pocket_at_a_time
oracle_read_map_real_write
real_read_map_oracle_write
output_calibration_bridge
residual_delta_integration
broad_read_tiny_write_reference
pruned_read_tiny_write_reference
```

## Metrics

```text
answer accuracy
composition usefulness
route accuracy
per-step RAM error
intermediate state drift
output calibration error
next-pocket input compatibility
read-context error
write-placement error
one-pocket failure attribution
heldout/OOD/counterfactual/adversarial usefulness
deterministic replay
checker failure_count
```

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
pocket_training_report.json
localization_report.json
one_pocket_attribution_report.json
calibration_report.json
step_drift_report.json
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
e7w_intermediate_state_drift_bottleneck
e7w_specific_pocket_bottleneck_detected
e7w_read_context_bottleneck
e7w_output_write_contract_bottleneck
e7w_output_calibration_bottleneck
e7w_flow_integration_bottleneck
e7w_composition_failure_unlocalized
```

## Guardrails

```text
no new architecture
no semantic lane labels
no hardcoded improvement flags
real row-level eval required
deterministic replay required
checker failure_count must be 0
```

## Boundary

E7W only localizes failures in a controlled numeric pocket-router proxy. It
does not prove raw-language learning, AGI, consciousness, or model-scale
behavior.

# E40 Mutable ALU Rule Cell Router Probe Contract

Milestone:

```text
E40_MUTABLE_ALU_RULE_CELL_ROUTER_PROBE
```

## Purpose

E39B showed that a router can infer `location+scale` from visible Flow-grid
markers. E40 tests a sharper question:

```text
Is a mutable ALU/rule-cell router more useful than a flat marker table when the
call decision depends on multiple Flow cells?
```

This is a controlled spatial Flow-grid proxy. It is not a raw language,
deployed-model, AGI, consciousness, or model-scale claim.

## Task

Every row contains a visible local header:

```text
marker cell
condition cell A
condition cell B
condition cell C
guard cell
target patch
decoy headers
```

The marker alone is intentionally insufficient. The correct local call is:

```text
location = marker + offset
scale = boolean rule over A and B
op = boolean rule over C
```

The target patch is transformed by the inferred operation.

## Systems

```text
oracle_alu_rule_reference
flat_marker_table_router
location_only_fixed_call_router
boolean_alu_without_op_router
mutable_alu_rule_cell_router
scan_all_rule_control
full_flow_painter_control
random_rule_control
```

The oracle reference is ineligible. Valid primary systems must use visible Flow
cells only.

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
alu_rule_program_report.json
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
docs/research/artifact_samples/e40_mutable_alu_rule_cell_router_probe/
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
e40_mutable_alu_rule_cell_router_positive
e40_flat_marker_table_sufficient
e40_boolean_scale_only_sufficient
e40_scan_all_required
e40_full_flow_required
e40_invalid_artifact_detected
```

Positive requires:

```text
mutable_alu_rule_cell_router exact >= 0.95
mutable_alu_rule_cell_router write_spread <= 0.12
flat_marker_table_router exact < 0.75
boolean_alu_without_op_router exact < 0.90
random_rule_control exact < 0.40
scan_all_rule_control solves but reads diffusely
full_flow_painter_control solves but writes diffusely
checker failure_count = 0
sample-only checker passes
deterministic replay passes
```

## Hard Requirements

No gradient descent, optimizer, backprop, direct solver, or hidden oracle access
inside valid primary systems. The run must write progress/heartbeat/mutation
partial artifacts during execution.

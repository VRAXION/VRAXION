# E7T Progressive RAM Port Allocation Probe Contract

## Purpose

`E7T_PROGRESSIVE_RAM_PORT_ALLOCATION_PROBE` follows E7R/E7S.

Core question:

```text
Do numeric pockets need a fixed RAM-wrapper/port map, and how many anonymous
RAM cells are minimally sufficient for composable pocket outputs?
```

This is a Flow/RAM IO allocation test. It is not a new architecture, not a
semantic lane-label test, and not a dense graph test.

## Model

```text
Flow[D] = shared RAM / working state
Pocket = callable numeric program
Router = scheduler / dispatcher
IO-wrapper = mechanical read/write access control
```

RAM cells are anonymous. No cell is named as memory, truth, confidence, or
result in the model input.

## Systems

```text
untyped_flow_baseline
output_write_map_only
input_read_map_only
input_plus_output_port_map
learned_sparse_mask_reference
progressive_write_slot_allocation
progressive_read_write_slot_allocation
learned_port_map_then_freeze
shared_write_control
integrator_shared_write_control
oracle_port_map_reference
dense_graph_danger_control
```

`shared_write_control`, `integrator_shared_write_control`, and
`dense_graph_danger_control` are diagnostic controls.

## Allocation Rules

```text
pockets may only write assigned RAM cells at inference when write maps are active
pockets may only read assigned RAM cells when read maps are active
non-written cells must be preserved
progressive allocation starts small and adds RAM cells
the selected budget is the smallest budget within plateau tolerance
```

Plateau rule:

```text
add slot while validation / OOD / counterfactual usefulness improves
choose smallest slot count within 0.002 of best
rollback if generalization drops
```

## Required Metrics

```text
composition usefulness
heldout usefulness
OOD usefulness
counterfactual usefulness
adversarial usefulness
answer accuracy
route accuracy
write spread
changed cell count
read cell count
write cell count
preserve corruption
write violation
RAM collision rate
next-pocket input compatibility
output calibration error
slot plateau curve per system
smallest stable write budget
smallest stable read budget
cost / bit budget
dense graph control comparison
deterministic replay
checker failure_count
```

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
pocket_training_report.json
port_map_report.json
slot_plateau_report.json
shared_write_report.json
flow_grid_frames.json
system_results.json
mutation_history.json
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
e7t_output_port_map_positive
e7t_input_output_port_map_positive
e7t_progressive_write_slot_allocation_positive
e7t_progressive_read_write_slot_allocation_positive
e7t_learned_frozen_port_map_positive
e7t_direct_shared_write_collision_detected
e7t_integrator_shared_write_positive
e7t_sparse_mask_contract_still_preferred
e7t_graph_soup_regression_detected
e7t_ram_port_allocation_no_advantage
```

## Guardrails

```text
no semantic lane labels
no anywhere-write primary path
no hidden answer or route input
no hardcoded improvement flags
no dense graph primary success
real row-level eval required
deterministic replay required
checker failure_count must be 0
```

## Boundary

E7T is a controlled numeric Flow/RAM IO allocation probe. It does not prove
raw-language learning, AGI, consciousness, or model-scale behavior.

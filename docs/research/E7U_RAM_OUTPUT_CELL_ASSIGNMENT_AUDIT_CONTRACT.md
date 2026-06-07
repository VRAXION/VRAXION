# E7U RAM Output Cell Assignment Audit Contract

## Purpose

`E7U_RAM_OUTPUT_CELL_ASSIGNMENT_AUDIT` follows E7T.

Core question:

```text
If a numeric pocket only needs one anonymous output RAM cell, how should that
cell be assigned?
```

This is only a RAM output-cell assignment audit. It does not change the
architecture and does not introduce semantic cell names.

## Runtime Rule

```text
once an output cell is assigned and validated, it is frozen
pockets may write only assigned output cells
no runtime-random output writes
```

The evaluator may use assignment metadata to interpret which anonymous RAM cell
holds which pocket output. That metadata is a mechanical write map, not a
semantic lane label.

## Systems

```text
next_free_output_cell_allocator
random_initial_then_freeze
mutation_selected_output_cell
progressive_output_cell_budget
learned_sparse_mask_reference
shared_write_collision_control
integrator_shared_write_control
oracle_output_cell_reference
dense_graph_danger_control
```

## Mutation Operators

```text
move_write_cell
swap_write_cell
add_output_cell
remove_output_cell
freeze_write_map
rollback_on_validation_drop
```

## Metrics

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
output cell count
RAM collision rate
preserve corruption
write violation
next-pocket compatibility
output calibration error
slot assignment stability
deterministic replay
checker failure_count
```

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
pocket_training_report.json
write_cell_assignment_report.json
collision_report.json
progressive_budget_report.json
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
e7u_next_free_output_allocator_sufficient
e7u_mutation_selected_output_cell_positive
e7u_output_cell_location_not_important
e7u_multi_output_cell_needed
e7u_direct_shared_write_collision_detected
e7u_integrator_shared_write_positive
e7u_sparse_mask_still_preferred
e7u_graph_soup_regression_detected
e7u_output_cell_assignment_no_advantage
```

## Guardrails

```text
no semantic labels
no runtime-random output writes
no dense graph primary success
no hardcoded improvement flags
real row-level eval required
deterministic replay required
checker failure_count must be 0
```

## Boundary

E7U only tests anonymous RAM output-cell assignment in a controlled numeric
pocket-router proxy. It does not prove raw-language learning, AGI,
consciousness, or model-scale behavior.

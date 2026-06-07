# E7V RAM Read Context Selection Audit Contract

## Purpose

`E7V_RAM_READ_CONTEXT_SELECTION_AUDIT` follows E7T/E7U.

Core question:

```text
Can mutation/rollback discover compact RAM read maps per pocket while keeping
the write side fixed and simple?
```

E7V is only a read-context selection audit. It does not change the architecture,
does not add semantic RAM labels, and does not move to image or language tasks.

## Fixed Write Policy

```text
deterministic next-free output cell allocator
1 output cell per pocket
frozen write map
no direct shared-write
```

The read map is per-pocket and anonymous. The evaluator may know which output
cell a pocket owns, but cells are not named as memory, truth, confidence, or
answer in model input.

## Systems

```text
broad_read_next_free_write_baseline
fixed_small_read_control
random_read_map_control
progressive_add_read_cells
prune_from_broad_read
swap_mutation_read_map
grid_neighborhood_read_map
sensitivity_guided_read_map_mutation
learned_sparse_mask_reference
oracle_read_map_reference
dense_graph_danger_control
```

`oracle_read_map_reference` and `dense_graph_danger_control` are references, not
candidate lifecycle policies.

## Mutation Operators

```text
add_read_cell
remove_read_cell
swap_read_cell
move_read_cell_nearby
expand_read_window
shrink_read_window
clone_read_map_from_similar_pocket
rollback_on_validation_or_generalization_drop
```

The write map remains fixed during these mutations.

## Plateau Rule

```text
add/prune/swap while validation or generalization usefulness improves by >= 0.002
choose the smallest read set within 0.002 of the best score
rollback if OOD/counterfactual/adversarial quality drops
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
read cell count
write cell count
smallest stable read count
read budget curve
next-pocket input compatibility
output calibration error
preserve corruption
write violation
write spread
read-map sparsity
grid locality score
mutation accepted/rejected/rollback counts
deterministic replay
checker failure_count
```

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
pocket_training_report.json
read_map_report.json
read_budget_curve_report.json
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
e7v_compact_read_map_positive
e7v_read_context_pruning_positive
e7v_progressive_read_growth_positive
e7v_read_map_swap_mutation_positive
e7v_ram_grid_topology_positive
e7v_broad_context_still_required
e7v_sparse_mask_still_preferred
e7v_graph_soup_regression_detected
e7v_read_context_selection_no_advantage
```

## Guardrails

```text
no semantic labels
no dense graph primary path
no runtime-random write map
no shared direct write
no hardcoded improvement flags
real row-level eval required
deterministic replay required
checker failure_count must be 0
```

## Boundary

E7V only tests anonymous RAM read-context selection in a controlled numeric
pocket-router proxy. It does not prove raw-language learning, AGI,
consciousness, or model-scale behavior.

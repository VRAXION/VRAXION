# E7U RAM Output Cell Assignment Audit Result

Run root:

```text
target/pilot_wave/e7u_ram_output_cell_assignment_audit
```

## Status

```text
status = complete
decision = e7u_direct_shared_write_collision_detected
best_non_reference_system = learned_sparse_mask_reference
checker_failure_count = 0
deterministic_replay_passed = true
```

## Evidence Configuration

```text
seeds = 99901,99902,99903
flow_dim = 40
output_bank = anonymous RAM cells 24..35
runtime_random_output_writes = false
artifact_contract = pass
deterministic_replay_hash_match = pass
```

The run wrote progress, hardware heartbeat, partial aggregate snapshots,
row-level eval samples, mutation history, output-cell assignment maps,
collision diagnostics, progressive budget curves, and deterministic replay
artifacts.

## Mean Scores

```text
system                                usefulness  answer_acc  write_spread  output_cells  collision
next_free_output_cell_allocator        0.626563    0.726562    0.014934      1.000         0.000000
random_initial_then_freeze             0.535937    0.635938    0.016228      1.000         0.000000
mutation_selected_output_cell          0.610938    0.710938    0.015225      1.000         0.000000
progressive_output_cell_budget         0.625521    0.725521    0.014944      1.000         0.000000
learned_sparse_mask_reference          0.634896    0.734896    0.029324      1.500         0.000000
shared_write_collision_control         0.540104    0.640104    0.011618      1.000         1.000000
integrator_shared_write_control        0.543750    0.643750    0.011608      1.000         1.000000
oracle_output_cell_reference           0.992953    1.000000    0.000000      1.000         0.000000
dense_graph_danger_control             0.489167    0.629167    0.000000      0.000         0.000000
```

## Assignment Findings

The next-free deterministic allocator was close to the best non-oracle system:

```text
learned_sparse_mask_reference = 0.634896
next_free_output_cell_allocator = 0.626563
gap = 0.008333
```

Mutation-selected output cells did not beat next-free:

```text
mutation_selected_output_cell = 0.610938
next_free_output_cell_allocator = 0.626563
```

Random initial freeze was worse:

```text
random_initial_then_freeze = 0.535937
```

This says cell location/order still matters in this proxy. A simple stable
next-free block was better than arbitrary random frozen cells.

## Progressive Output Budget

Progressive budget selected one output cell on every seed:

```text
seed 99901: chosen_output_budget = 1
seed 99902: chosen_output_budget = 1
seed 99903: chosen_output_budget = 1
mean output cells = 1.0
```

Adding more output cells did not help; it usually hurt OOD/adversarial
usefulness.

## Shared Write Control

Direct shared-write and integrator shared-write both collided on every seed:

```text
shared_collision_rate = 1.0
integrator_collision_rate = 1.0
shared_usefulness = 0.540104
integrator_usefulness = 0.543750
```

The integrator did not rescue shared output writes enough to be useful.

## Interpretation

E7U confirms the E7T write-side picture:

```text
1 output cell per pocket is enough on this proxy
next-free output assignment is nearly sufficient
mutation-selected output cell placement does not help
random frozen placement is worse
shared output writes collide and should not be the main path
```

The learned sparse mask remains a small-margin best system, but it uses a
larger average output footprint. The practical engineering result is that the
write side can probably stay simple:

```text
deterministic next-free output allocator + frozen one-cell write map
```

Recommended next step:

```text
E7V_RAM_READ_CONTEXT_SELECTION_AUDIT
```

## Boundary

E7U only tests anonymous RAM output-cell assignment in a controlled numeric
pocket-router proxy. It does not prove raw-language learning, AGI,
consciousness, or model-scale behavior.

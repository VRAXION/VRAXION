# E7T Progressive RAM Port Allocation Probe Result

Run root:

```text
target/pilot_wave/e7t_progressive_ram_port_allocation_probe
```

## Status

```text
status = complete
decision = e7t_sparse_mask_contract_still_preferred
best_non_reference_system = learned_sparse_mask_reference
checker_failure_count = 0
deterministic_replay_passed = true
```

## Evidence Configuration

```text
seeds = 99801,99802,99803
flow_dim = 40
input_bank = anonymous Flow cells 0..23 plus result cells 24..29
write_bank = anonymous result/scratch port cells
replay_hash_match = pass
artifact_contract = pass
```

The run wrote progress, hardware heartbeat, partial aggregate snapshots,
row-level eval samples, mutation history, slot plateau curves, shared-write
diagnostics, and deterministic replay artifacts.

## Mean Scores

```text
system                                      usefulness  answer_acc  write_spread  read_cells  write_cells
untyped_flow_baseline                      0.525000    0.625000    0.995413      40.000      40.000
output_write_map_only                      0.595833    0.695833    0.014736      40.000      1.000
input_read_map_only                        0.479167    0.579167    0.996291      12.000      40.000
input_plus_output_port_map                 0.580208    0.680208    0.015036      30.000      1.000
learned_sparse_mask_reference              0.604167    0.704167    0.023822      39.100      1.392
progressive_write_slot_allocation          0.597396    0.697396    0.014856      40.000      1.000
progressive_read_write_slot_allocation     0.581250    0.681250    0.015073      30.000      1.000
learned_port_map_then_freeze               0.586979    0.686979    0.024547      39.100      1.392
shared_write_control                       0.596875    0.696875    0.039814      40.000      2.000
integrator_shared_write_control            0.595313    0.695312    0.039962      40.000      2.000
oracle_port_map_reference                  0.992953    1.000000    0.012297      30.000      1.000
dense_graph_danger_control                 0.440208    0.580208    0.000000      0.000       0.000
```

## Plateau Findings

Progressive write allocation plateaued immediately:

```text
seed 99801: chosen_write_budget = 1, best_score = 0.637500
seed 99802: chosen_write_budget = 1, best_score = 0.609375
seed 99803: chosen_write_budget = 1, best_score = 0.545313
mean smallest stable write budget = 1
```

Progressive read/write allocation did not find a small read footprint:

```text
seed 99801: chosen_read_budget = 30
seed 99802: chosen_read_budget = 30
seed 99803: chosen_read_budget = 30
mean smallest stable read budget = 30
```

## Shared Write Control

Direct and integrator shared-write controls had collision rate `1.0` by design.
They did not beat the learned sparse mask reference:

```text
shared_write_control            usefulness = 0.596875
integrator_shared_write_control usefulness = 0.595313
learned_sparse_mask_reference   usefulness = 0.604167
```

So shared write was not the best path in this run.

## Interpretation

The cleanest result is:

```text
write-port discipline matters a lot
read-port restriction is expensive on this task
progressive write allocation says 1 output cell per pocket is enough
learned sparse mask remains the best non-oracle wrapper
```

E7T did not show that progressive +1 write allocation beats the learned sparse
E7R-style mask. It did show that most of the E7S RAM-smear problem is fixed by
a very small output/write port:

```text
untyped write_spread = 0.995413
output port write_spread = 0.014736
```

The remaining gap to oracle is therefore probably not caused by needing more
write RAM. It is more likely read/context selection or pocket trainability.

Recommended next step:

```text
E7U_READ_CONTEXT_SELECTION_AND_ROUTER_VISIBLE_STATE_AUDIT
```

## Boundary

E7T only tests numeric pocket Flow/RAM IO allocation in a controlled
pocket-router proxy. It does not prove raw-language learning, AGI,
consciousness, or model-scale behavior.

# E7I Pocket Size Optimum Sweep Contract

## Purpose

E7H showed mutation can discover reusable pocket boundaries from clean microsegments.
E7H may still contain generator imprint because its hidden natural pocket size was effectively 2.

E7I measures the pocket-size curve:

```text
size 1 atomic
fixed size 2
fixed size 3
fixed size 4
mixed size 2-3
mixed size 2-4
variable mutation-discovered size
fused long pipe
dense graph control
```

## Runner And Checker

- Runner: `scripts/probes/run_e7i_pocket_size_optimum_sweep.py`
- Checker: `scripts/probes/run_e7i_pocket_size_optimum_sweep_check.py`
- Default artifact root: `target/pilot_wave/e7i_pocket_size_optimum_sweep/`

## Task Families

```text
family_A_natural_size_2
family_B_natural_size_3
family_C_natural_size_4
family_D_mixed_size_2_4
family_E_no_stable_pocket_size
family_F_decoy_pair_frequency
family_G_reuse_sparse_family
```

## Systems

```text
atomic_microsegment_router
fixed_size_2_pockets
fixed_size_3_pockets
fixed_size_4_pockets
mixed_size_2_3_pockets
mixed_size_2_4_pockets
mutation_discovered_variable_size_pockets
fixed_human_pocket_scaffold
fused_long_pipe
dense_graph_control
random_boundary_control
oracle_family_granularity_reference
```

## Metrics

Required metrics:

```text
heldout usefulness
OOD usefulness
counterfactual usefulness
adversarial usefulness
eval mean usefulness
route accuracy
answer accuracy
mean route steps
pocket count
average pocket size
pocket size distribution
reuse count per pocket
freeze survival
overfit/generalization gap
irrelevant branch rate
loop rate
router complexity
parameter cost
accepted/rejected mutations
rollback count
deterministic replay hash match
hardware heartbeat
```

## Decisions

Allowed decisions:

```text
e7i_stable_pocket_size_optimum_detected
e7i_variable_pocket_granularity_preferred
e7i_size2_was_generator_imprint
e7i_pocket_size_needs_prior_scaffold
e7i_atomic_microsegment_routing_preferred
e7i_fused_pipe_overfit_detected
e7i_pocket_granularity_collapses_to_graph_soup
e7i_no_clear_size_frontier
e7i_leak_or_artifact_detected
```

## Hardware And Progress

Evidence runs use:

```text
GPU lane:
  dense_graph_control

CPU lanes:
  mutation_discovered_variable_size_pockets
```

The runner must write `progress.jsonl`, `hardware_heartbeat.jsonl`, mutation/training snapshots, `partial_aggregate_snapshot.json`, final artifacts, deterministic replay artifacts, and checker output.

## Boundary

E7I is a controlled symbolic/numeric pocket-size sweep. It does not make broad reasoning or large-scale behavior claims.

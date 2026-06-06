# E7J Dynamic Thought Capacity Allocation Probe Contract

## Purpose

E7I showed that there is no universal pocket size. Pocket granularity was
task-family dependent, and the variable-size discovery system improved over
generic fixed sizes but still trailed a family-aware scaffold.

E7J keeps the external interface fixed:

```text
CALL(pocket_id, Flow[D]) -> Flow[D]
```

and tests whether the system can allocate internal pocket capacity:

```text
input_adapter:  D -> K
core_matrix:    K -> K
output_adapter: K -> D
```

Allowed `K` values:

```text
1, 2, 4, 8
```

## Runner And Checker

- Runner: `scripts/probes/run_e7j_dynamic_thought_capacity_allocation_probe.py`
- Checker: `scripts/probes/run_e7j_dynamic_thought_capacity_allocation_probe_check.py`
- Default artifact root: `target/pilot_wave/e7j_dynamic_thought_capacity_allocation_probe/`

## Systems

```text
fixed_K1_pockets
fixed_K2_pockets
fixed_K4_pockets
fixed_K8_pockets
variable_K_allocator
variable_K_grow_shrink_mutation
variable_K_split_merge_mutation
family_aware_capacity_scaffold
fused_long_pipe
dense_graph_danger_control
random_capacity_control
```

## Task Families

E7J reuses the E7I multi-family symbolic/numeric proxy:

```text
family_A_natural_size_2
family_B_natural_size_3
family_C_natural_size_4
family_D_mixed_size_2_4
family_E_no_stable_pocket_size
family_F_decoy_pair_frequency
family_G_reuse_sparse_family
```

## Metrics

Required metrics:

```text
heldout usefulness
OOD usefulness
counterfactual usefulness
adversarial usefulness
capacity value score
route accuracy
answer accuracy
mean route steps
pocket count
average K
K distribution
capacity fit rate
capacity under/over rate
reuse count per pocket
freeze survival
local repair gain
parameter cost
compute cost
router complexity
overfit/generalization gap
irrelevant branch rate
loop rate
accepted/rejected mutations
rollback count
deterministic replay hash match
hardware heartbeat
```

Capacity value is the main score. It rewards usefulness, OOD/counterfactual
robustness, reuse, and repairability while penalizing parameter cost, compute
cost, routing complexity, under-capacity, over-capacity, and overfit.

## Decisions

Allowed decisions:

```text
e7j_dynamic_capacity_allocation_positive
e7j_dynamic_capacity_partially_positive
e7j_fixed_capacity_sufficient
e7j_capacity_needs_prior_scaffold
e7j_variable_capacity_overfits_or_cost_ignored
e7j_fused_pipe_capacity_preferred
e7j_dense_graph_collapse_detected
e7j_leak_or_artifact_detected
```

## Hard Requirements

```text
real row-level eval
cost accounting
deterministic replay
checker failure_count = 0
no hardcoded improvements
progress artifacts
mutation history snapshots
training history snapshots
CPU + GPU lane execution
no broad AGI/consciousness/model-scale claims
```

## Boundary

E7J is a controlled symbolic/numeric capacity-allocation proxy. It does not
claim raw language reasoning, broad intelligence, consciousness, or model-scale
behavior.

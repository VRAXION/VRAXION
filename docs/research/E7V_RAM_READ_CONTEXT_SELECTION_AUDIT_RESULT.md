# E7V RAM Read Context Selection Audit Result

Status: complete.

```text
decision = e7v_compact_read_map_positive
best_system = prune_from_broad_read
broad_read_still_needed = no
best_read_map_strategy = prune_from_broad_read
smallest_stable_read_count = 30.25 mean cells
grid_topology_helped = no
sensitivity_guided_mutation_helped = weak/no
best_vs_learned_sparse_mask = 0.612109 vs 0.596289
best_vs_oracle = 0.612109 vs 0.992953
deterministic_replay_passed = true
checker_failure_count = 0
```

## Mean Scores

```text
broad_read_next_free_write_baseline        useful=0.599023 acc=0.699023 read=40.000 ood=0.531250
progressive_add_read_cells                 useful=0.611133 acc=0.711133 read=31.000 ood=0.553125
prune_from_broad_read                      useful=0.612109 acc=0.712109 read=30.250 ood=0.550000
swap_mutation_read_map                     useful=0.599805 acc=0.699805 read=37.688 ood=0.517188
grid_neighborhood_read_map                 useful=0.557422 acc=0.657422 read=12.000 ood=0.427344
sensitivity_guided_read_map_mutation       useful=0.589844 acc=0.689844 read=12.000 ood=0.503125
learned_sparse_mask_reference              useful=0.596289 acc=0.696289 read=37.667 ood=0.530469
oracle_read_map_reference                  useful=0.992953 acc=1.000000 read=0.000 ood=0.992953
dense_graph_danger_control                 useful=0.450039 acc=0.590039 read=0.000 ood=0.428750
```

## Budget Curve Summary

```text
progressive_add_read_cells chosen budgets:
  8: 1 seed
  24: 1 seed
  30: 2 seeds
  40: 4 seeds

prune_from_broad_read chosen budgets:
  8: 1 seed
  24: 2 seeds
  30: 1 seed
  40: 4 seeds
```

## Interpretation

E7V supports compact read-context selection, but not extreme read sparsity. The
best policy pruned broad RAM reads from 40 cells to about 30 cells on average
while slightly improving usefulness. Very small fixed/grid/sensitivity maps at
12 cells lost too much context, especially on OOD rows.

The write side stayed fixed throughout:

```text
deterministic next-free output cell
1 output cell per pocket
frozen write map
no direct shared write
```

Dense graph control did not win. Learned sparse mask did not beat the simpler
prune/progressive read-budget path in this run.

## Recommended Next Step

```text
E7W_RAM_READ_CONTEXT_BLOCK_STRUCTURE_AUDIT
```

Reason: E7V says the read side can shrink, but the stable budget is still large
at roughly 24-40 cells depending on seed. The next question is whether those
cells form reusable anonymous blocks/context bands, or whether the proxy still
needs broad diffuse read context.

Boundary: this is a controlled anonymous RAM read-context selection probe, not a
raw-language, AGI, consciousness, or model-scale claim.

# E7J Dynamic Thought Capacity Allocation Probe Result

## Decision

```text
decision = e7j_dynamic_capacity_allocation_positive
best_system = family_aware_capacity_scaffold
best_variable_K_system = variable_K_split_merge_mutation
deterministic_replay_passed = true
checker_failure_count = 0
```

E7J tested whether a fixed external `Flow[D]` interface can support variable
internal capacity allocation per callable thought-pocket:

```text
CALL(pocket_id, Flow[D]) -> Flow[D]
internal pocket capacity K in {1, 2, 4, 8}
```

The result is positive on this controlled proxy. The best variable-K system
almost closed the scaffold gap while beating every generic fixed-K baseline by a
large margin.

## Mean Capacity Value

```text
family_aware_capacity_scaffold        0.770195
variable_K_split_merge_mutation       0.769250
variable_K_grow_shrink_mutation       0.768918
variable_K_allocator                  0.748964
fixed_K8_pockets                      0.645795
fixed_K4_pockets                      0.572079
fused_long_pipe                       0.462502
fixed_K2_pockets                      0.458849
dense_graph_danger_control            0.323579
fixed_K1_pockets                      0.304800
random_capacity_control               0.264376
```

## Frontier

```text
best_fixed_K_system = fixed_K8_pockets
best_variable_K_system = variable_K_split_merge_mutation
variable_minus_best_fixed_capacity_value = +0.123456
scaffold_minus_variable_capacity_value = +0.000945
variable_minus_fused_ood_capacity_value = +0.338985
dense_capacity_value_mean = 0.323579
```

## Split Signals

```text
family_aware_capacity_scaffold:
  capacity = 0.770195
  usefulness = 0.716222
  OOD capacity = 0.779396
  counterfactual capacity = 0.746325
  adversarial capacity = 0.777372
  average K = 4.186
  mean compute cost = 480.6

variable_K_split_merge_mutation:
  capacity = 0.769250
  usefulness = 0.714778
  OOD capacity = 0.778454
  counterfactual capacity = 0.745384
  adversarial capacity = 0.776416
  average K = 4.082
  mean compute cost = 480.9

fixed_K8_pockets:
  capacity = 0.645795
  usefulness = 0.644159
  OOD capacity = 0.649235
  average K = 8.000
  mean compute cost = 1124.4
```

## Family Winners

```text
family_A_natural_size_2            fixed_K2_pockets
family_B_natural_size_3            family_aware_capacity_scaffold
family_C_natural_size_4            family_aware_capacity_scaffold
family_D_mixed_size_2_4            family_aware_capacity_scaffold
family_E_no_stable_pocket_size     fixed_K1_pockets
family_F_decoy_pair_frequency      family_aware_capacity_scaffold
family_G_reuse_sparse_family       variable_K_split_merge_mutation
```

## Interpretation

Variable internal capacity allocation worked. It did not simply set every pocket
to `K=8`: the winning variable system settled near average `K=4.082`, while the
best fixed baseline was `fixed_K8_pockets` with much higher compute cost and much
lower capacity value.

The result also separates two ideas:

```text
fixed K everywhere:
  simple, but either under-capacity or over-expensive

variable K per thought-pocket:
  close to family-aware scaffold while keeping cost controlled
```

Dense graph control did not win, so this run did not collapse into graph soup.
Fused long pipe was far behind on capacity value and OOD capacity.

## Next Step

Run a harder capacity-transfer falsification:

```text
E7K_DYNAMIC_CAPACITY_TRANSFER_AND_DRIFT_PROBE
```

The next question is whether the learned K allocation transfers after pocket
health drift, new family mixtures, and partial pocket repair budgets.

## Boundary

This is a controlled symbolic/numeric capacity-allocation proxy over fixed
callable thought-pockets. It does not claim raw language reasoning, broad
intelligence, consciousness, or model-scale behavior.

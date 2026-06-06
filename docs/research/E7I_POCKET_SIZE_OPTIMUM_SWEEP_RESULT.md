# E7I Pocket Size Optimum Sweep Result

## Decision

```text
decision = e7i_pocket_size_needs_prior_scaffold
best_non_oracle_system = fixed_human_pocket_scaffold
deterministic_replay_passed = true
checker_failure_count = 0
```

E7I measured the pocket-size curve across seven task families to test whether the
size-2 pocket discovered in E7H was a stable optimum or partly a generator imprint.

The result is conservative: variable-size mutation discovery beat every generic
fixed-size baseline, but it did not beat the family-aware human scaffold. That
means pocket size appears task-family dependent in this proxy, and the current
variable discovery mechanism still benefits from a prior scaffold.

## Mean Eval Usefulness

```text
fixed_human_pocket_scaffold                 0.777642
oracle_family_granularity_reference         0.777642
mutation_discovered_variable_size_pockets   0.762958
fixed_size_3_pockets                        0.730905
fixed_size_2_pockets                        0.729435
mixed_size_2_3_pockets                      0.701616
mixed_size_2_4_pockets                      0.699521
fixed_size_4_pockets                        0.686096
atomic_microsegment_router                  0.639417
fused_long_pipe                             0.525154
dense_graph_control                         0.350339
random_boundary_control                     0.225539
```

## Split Signals

```text
fixed_human_pocket_scaffold:
  heldout = 0.789886
  OOD = 0.790456
  counterfactual = 0.740322
  adversarial = 0.789904
  mean steps = 3.582
  average pocket size = 2.607

mutation_discovered_variable_size_pockets:
  heldout = 0.773348
  OOD = 0.774003
  counterfactual = 0.730552
  adversarial = 0.773928
  mean steps = 4.140
  average pocket size = 2.900
```

## Family Winners

```text
family_A_natural_size_2            fixed_size_2_pockets
family_B_natural_size_3            fixed_size_3_pockets
family_C_natural_size_4            fixed_size_4_pockets
family_D_mixed_size_2_4            fixed_human_pocket_scaffold
family_E_no_stable_pocket_size     fixed_size_2_pockets
family_F_decoy_pair_frequency      fixed_human_pocket_scaffold
family_G_reuse_sparse_family       fixed_human_pocket_scaffold
```

## Frontier

```text
best_generic_fixed_size_system = fixed_size_3_pockets
variable_minus_best_generic_fixed_eval = +0.032052
variable_minus_size2_eval = +0.033523
fixed_human_minus_variable_eval = +0.014684
variable_minus_fused_ood = +0.276520
fused_heldout_minus_ood = +0.137138
```

## Interpretation

E7H size 2 was not a universal optimum. It won on the natural-size-2 family, but
size 3 won the natural-size-3 family and size 4 won the natural-size-4 family.
Mixed and decoy/reuse families favored a family-aware scaffold.

Variable-size discovery is real signal, not noise: it beat the best generic fixed
size by more than 0.03 eval usefulness and stayed far above fused/dense controls.
But it is not yet strong enough to claim scaffold-free pocket sizing because the
family-aware scaffold remained best overall.

Atomic routing was too step-expensive. Fused long pipe did not win and showed
OOD weakness. Dense graph control failed rather than replacing pocket routing.

## Next Step

Run a focused scaffold-reduction probe: keep the E7I multi-family setup, but give
the mutation-discovered variable-size system better split/merge/relabel operators
and test whether it can close the +0.014684 gap to the family-aware scaffold
without exposing hidden natural pocket sizes as inputs.

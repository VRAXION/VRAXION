# E7K Dynamic Pocket Spawn And Promotion Probe Result

## Decision

```text
decision = e7k_dynamic_pocket_spawn_positive
best_system = oracle_spawn_scaffold
best_spawn_system = control_spawn_plus_limited_repair
deterministic_replay_passed = true
checker_failure_count = 0
```

## Mean Scores

```text
fixed_library_no_spawn                 spawn=0.463121 useful=0.463121 ood=0.469219 promoted=0.00
fixed_library_router_plus_repair       spawn=0.531902 useful=0.531902 ood=0.530675 promoted=0.00
oracle_spawn_scaffold                  spawn=0.799863 useful=0.700260 ood=0.800579 promoted=4.00
random_spawn_control                   spawn=0.376738 useful=0.466939 ood=0.383109 promoted=5.00
control_spawn_blank_pocket             spawn=0.543723 useful=0.549926 ood=0.543769 promoted=2.10
control_spawn_from_split               spawn=0.648431 useful=0.647006 ood=0.645390 promoted=7.10
control_spawn_from_composed_route      spawn=0.739912 useful=0.659659 ood=0.736339 promoted=5.60
control_spawn_plus_limited_repair      spawn=0.792914 useful=0.701229 ood=0.787248 promoted=4.00
dense_graph_danger_control             spawn=0.356332 useful=0.356332 ood=0.349625 promoted=0.00
```

## Frontier

```text
best_fixed_system = fixed_library_router_plus_repair
best_spawn_system = control_spawn_plus_limited_repair
spawn_minus_best_fixed = 0.261012560858
oracle_minus_best_spawn = 0.006948218151
random_spawn_value_mean = 0.376738269380
dense_spawn_value_mean = 0.356331899084
best_spawn_promoted_pocket_count = 4.0
best_spawn_average_K = 3.5
best_spawn_average_depth = 1.0
best_spawn_unnecessary_spawn_rate = 0.45
```

## Phase Winners

```text
phase_1_existing_library_sufficient          fixed_library_no_spawn
phase_2_missing_reusable_transform           control_spawn_plus_limited_repair
phase_3_reuse_multiple_contexts              control_spawn_plus_limited_repair
phase_4_ood_counterfactual_generalization    oracle_spawn_scaffold
phase_5_damage_drift_repair                  control_spawn_plus_limited_repair
```

## Interpretation

E7K gives positive proxy evidence that a typed control layer can promote new callable pockets when the fixed pocket library is insufficient. The winning learned spawn variant was `control_spawn_plus_limited_repair`: it beat fixed library repair by `+0.261013` mean spawn value and landed within `0.006948` of the oracle scaffold.

The result does not say blank spawning is enough. Blank spawn was weak, split spawn overproduced, and the strongest learned path was route-cache/composed spawn plus limited local repair. Random spawn and dense graph danger controls did not match the learned spawn systems.

## Artifact Root

```text
target/pilot_wave/e7k_dynamic_pocket_spawn_and_promotion_probe/
```

# E7L Spawn Repair Cost And Noisy Health Falsification Result

## Decision

```text
decision = e7l_cost_aware_spawn_repair_survives
best_non_oracle_system = cost_aware_spawn_plus_repair
best_system_including_oracle = cost_aware_spawn_plus_repair
deterministic_replay_passed = true
checker_failure_count = 0
```

## Mean Scores

```text
no_adaptation                              net=0.461329 raw=0.486397 ood=0.467329 spawnP=0.000 spawnR=0.000 junk=0.000
route_around_only                          net=0.530490 raw=0.556854 ood=0.524975 spawnP=0.000 spawnR=0.000 junk=0.000
repair_only                                net=0.500122 raw=0.565190 ood=0.493138 spawnP=0.000 spawnR=0.000 junk=0.000
spawn_only                                 net=0.500482 raw=0.598670 ood=0.501794 spawnP=0.800 spawnR=0.633 junk=0.000
spawn_plus_limited_repair_clean            net=0.489349 raw=0.686180 ood=0.479549 spawnP=0.579 spawnR=0.733 junk=0.194
cost_aware_spawn_plus_repair               net=0.542774 raw=0.677634 ood=0.532163 spawnP=0.800 spawnR=0.633 junk=0.000
noisy_health_spawn_plus_repair             net=0.501780 raw=0.677634 ood=0.489755 spawnP=0.800 spawnR=0.633 junk=0.000
delayed_feedback_spawn_plus_repair         net=0.511894 raw=0.687362 ood=0.501907 spawnP=0.713 spawnR=0.733 junk=0.125
oracle_health_spawn_repair_reference       net=0.482682 raw=0.701148 ood=0.482763 spawnP=0.713 spawnR=0.733 junk=0.094
random_spawn_repair_control                net=0.219458 raw=0.488889 ood=0.220524 spawnP=0.000 spawnR=0.000 junk=0.065
dense_graph_danger_control                 net=0.367868 raw=0.404215 ood=0.355238 spawnP=0.000 spawnR=0.000 junk=0.000
```

## Falsification Frontier

```text
cost_aware_net = 0.542774394474
route_around_net = 0.530490333333
repair_only_net = 0.500122300596
spawn_only_net = 0.500482231775
clean_spawn_repair_net = 0.489349045888
noisy_spawn_repair_net = 0.501779602807
delayed_spawn_repair_net = 0.511894040470
dense_graph_net = 0.367867569084
random_control_net = 0.219458258627
cost_aware_unnecessary_spawn_rate = 0.400000
clean_unnecessary_spawn_rate = 0.456250
cost_aware_junk_pocket_rate = 0.000000
clean_junk_pocket_rate = 0.193750
delayed_feedback_regret = 0.086278823264
```

## Interpretation

E7K survived the cost/noise falsification in a narrower form: the clean spawn+repair policy did not remain best once spawn, repair, maintenance, route-step, overproduction, health, and delayed-regret costs were charged. The winner was `cost_aware_spawn_plus_repair`, which kept the typed spawn/repair mechanism but pruned junk better and beat route-around-only by `+0.012284` mean net utility.

The result is not "spawn freely." Clean spawn+repair had higher raw usefulness but lower net utility because it overproduced and retained junk-like pockets. Route-around-only was a strong cheap baseline and nearly matched the winner, so future work should treat route-around as a serious default action, not just a fallback.

## Phase Winners

```text
phase_1_existing_library_sufficient          no_adaptation
phase_2_missing_reusable_transform           spawn_only
phase_3_reuse_multiple_contexts              spawn_only
phase_4_ood_counterfactual_generalization    spawn_only
phase_5_damage_drift_repair                  delayed_feedback_spawn_plus_repair
```

## Artifact Root

```text
target/pilot_wave/e7l_spawn_repair_cost_and_noisy_health_falsification/
```

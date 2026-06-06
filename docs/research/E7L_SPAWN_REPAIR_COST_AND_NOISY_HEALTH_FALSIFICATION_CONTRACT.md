# E7L Spawn Repair Cost And Noisy Health Falsification Contract

## Purpose

E7K showed that dynamic pocket spawn and promotion works in a clean typed pocket-flow proxy. E7L is a falsification run: it tests whether that mechanism still works when spawn, repair, maintenance, route steps, noisy/incomplete health, delayed validation, moving drift, and junk-spawn pressure are included.

Core question:

```text
Can the control/router layer choose correctly between route-around, repair,
spawn, and no-op under cost/noise constraints?
```

## Systems

```text
no_adaptation
route_around_only
repair_only
spawn_only
spawn_plus_limited_repair_clean
cost_aware_spawn_plus_repair
noisy_health_spawn_plus_repair
delayed_feedback_spawn_plus_repair
oracle_health_spawn_repair_reference
random_spawn_repair_control
dense_graph_danger_control
```

Mutation systems must use mutation/rollback only. The dense graph control is the only gradient-trained danger baseline.

## Stress Conditions

```text
repair_cost
spawn_cost
pocket_maintenance_cost
noisy_health_signal
incomplete_health_signal
delayed_validation_feedback
moving_drift_profile
false_positive_reusable_route
false_negative_reusable_route
junk_spawn_pressure
```

The E7K symbolic row generator is reused, but the evaluation objective changes from clean spawn value to net utility under stress.

## Net Utility

```text
net_utility =
  raw_usefulness
  - spawn_cost
  - repair_cost
  - maintenance_cost
  - route_step_cost
  - overproduction_penalty
  - delayed_regret_penalty
  - health_signal_penalty
```

Net utility, not raw usefulness, decides the non-oracle winner.

## Required Metrics

```text
raw usefulness
net utility after cost
heldout/OOD/counterfactual/adversarial net utility
route accuracy
answer accuracy
spawn precision
spawn recall
unnecessary spawn rate
promoted pocket count
promoted pocket reuse count
repair precision
repair recall
repair gain per cost
route-around success
health false positive rate
health false negative rate
delayed feedback regret
junk pocket rate
library size
maintenance cost
dense graph control comparison
accepted/rejected/rollback mutation counts
deterministic replay
```

## Decision Rules

```text
e7l_cost_aware_spawn_repair_survives
  Cost-aware spawn+repair is the best non-oracle action policy under cost/noise.

e7l_routearound_preferred_under_cost
  Route-around beats spawn/repair because action cost is too high.

e7l_repair_preferred_spawn_too_expensive
  Repair-only beats spawn policies after cost.

e7l_spawn_preferred_repair_not_needed
  Spawn-only wins after cost and repair is not needed on this proxy.

e7l_spawn_repair_requires_clean_health_signal
  Clean spawn+repair works but noisy/incomplete health collapses.

e7l_spawn_overproduction_failure
  Spawn policies produce too many unused or junk pockets under cost.

e7l_delayed_feedback_instability
  Delayed validation feedback causes regret and large net-utility collapse.

e7l_graph_soup_regression_detected
  Dense graph danger control wins cleanly, suggesting typed pocket routing is not needed here.

e7l_leak_or_artifact_detected
  Random/control/leakage/checker gates fail.
```

## Artifact Contract

Required artifacts:

```text
backend_manifest.json
task_generation_report.json
cost_noise_stress_report.json
action_policy_report.json
spawn_repair_policy_report.json
health_signal_report.json
delayed_feedback_report.json
system_results.json
mutation_history.json
training_history.json
leakage_report.json
deterministic_replay.json
aggregate_metrics.json
decision.json
summary.json
report.md
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
checker_report.json
```

The checker must fail on missing systems, missing row-level samples, missing accepted/rejected mutations, rollback mismatch, missing parameter diff/hash, deterministic replay mismatch, missing stress metrics, mutation backprop/optimizer calls, dense graph leakage in mutation systems, broad claims, or `failure_count != 0`.

## Boundary

E7L is a controlled symbolic/numeric pocket-flow falsification. It does not test raw language, deployed large-model behavior, consciousness, or AGI.

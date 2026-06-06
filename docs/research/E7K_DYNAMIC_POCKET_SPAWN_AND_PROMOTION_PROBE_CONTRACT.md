# E7K Dynamic Pocket Spawn And Promotion Probe Contract

## Purpose

E7K tests whether a typed pocket-flow control layer can create a new callable pocket when the existing library is insufficient.

The fixed external interface is:

```text
CALL(pocket_id, Flow[D]) -> Flow[D]
```

A spawned pocket has an ID, segment contract, internal K capacity, depth, promotion state, freeze state, and optional local repair permission.

## Systems

```text
fixed_library_no_spawn
fixed_library_router_plus_repair
oracle_spawn_scaffold
random_spawn_control
control_spawn_blank_pocket
control_spawn_from_split
control_spawn_from_composed_route
control_spawn_plus_limited_repair
dense_graph_danger_control
```

Only the four `control_spawn_*` systems use mutation/rollback. The dense graph system is a danger control, not an accepted architecture path.

## Task Phases

```text
phase_1_existing_library_sufficient
phase_2_missing_reusable_transform
phase_3_reuse_multiple_contexts
phase_4_ood_counterfactual_generalization
phase_5_damage_drift_repair
```

Rows expose the microsegment path and phase token. Missing motif IDs are hidden and are used only for evaluation of spawn precision/recall.

## Required Metrics

```text
heldout usefulness
OOD usefulness
counterfactual usefulness
adversarial usefulness
answer accuracy
route accuracy
spawn precision
spawn recall
unnecessary spawn rate
failed spawn rollback count
promoted pocket count
promoted pocket reuse count
spawned pocket average K
spawned pocket average depth
route step reduction
route cost reduction
freeze survival
local repair gain
overfit gap
dense graph control comparison
accepted/rejected/rollback mutation counts
deterministic replay
```

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
spawn_mechanism_report.json
spawn_promotion_report.json
phase_spawn_winner_report.json
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
```

## Decisions

```text
e7k_dynamic_pocket_spawn_positive
e7k_composed_route_pocket_spawn_positive
e7k_split_spawn_positive
e7k_spawn_needs_prior_scaffold
e7k_spawn_artifact_or_task_too_easy
e7k_spawn_overproduction_failure
e7k_no_spawn_needed_existing_library_sufficient
e7k_pocket_spawn_collapses_to_graph_soup
e7k_leak_or_artifact_detected
```

## Checker Gates

The checker must fail on missing artifacts, missing systems, missing row-level samples, missing accepted/rejected mutations, rollback mismatch, missing parameter diff/hash, deterministic replay mismatch, mutation systems using backprop/optimizer calls, random spawn artifact without artifact decision, dense graph leakage into mutation systems, or broad claims in `report.md`.

## Boundary

E7K is a controlled symbolic/numeric pocket-flow proxy. It only tests dynamic spawn and promotion under the artifact contract above.

# E7H Pocket Granularity Discovery Probe Contract

## Purpose

E7H follows E7G.

E7G showed:

```text
if chapter IDs already exist,
direct chapter calls and skip-to addressing work.
```

E7H asks the harder question:

```text
Can mutation/rollback discover what should become a pocket/chapter
from smaller microsegments?
```

The target abstraction is:

```text
microsegment < discovered pocket/chapter < fused long pipe
```

## Runner And Checker

- Runner: `scripts/probes/run_e7h_pocket_granularity_discovery_probe.py`
- Checker: `scripts/probes/run_e7h_pocket_granularity_discovery_probe_check.py`
- Default artifact root: `target/pilot_wave/e7h_pocket_granularity_discovery_probe/`

## Systems

```text
atomic_microsegment_router
fixed_human_pockets
fused_long_pipe
mutation_discovered_pockets
discovered_pockets_plus_router
discovered_pockets_plus_limited_repair
dense_graph_control
random_boundary_control
oracle_granularity_reference
```

## Allowed Mutations

```text
merge_segments
split_pocket
move_boundary
assign_chapter_id
freeze_unfreeze
local_repair_permission
router_prior
call_skip_preference
```

## Forbidden Mechanisms

```text
dense all-to-all soft routing in mutation systems
anonymous micro-node soup
continuous activation mixing between all microsegments
route labels leaked into pocket IDs
direct oracle chapter grouping as model input
```

The dense graph system is allowed only as a danger control.

## Metrics

Required metrics:

```text
usefulness
OOD usefulness
counterfactual usefulness
adversarial usefulness
route accuracy
answer accuracy
mean route steps
discovered pocket count
average pocket size
reuse count per pocket
freeze survival score
local repair gain
overfit/generalization gap
irrelevant branch rate
loop rate
dense graph control comparison
accepted/rejected mutations
rollback count
deterministic replay hash match
hardware heartbeat
```

## Hardware Rule

Evidence runs use concurrent lanes:

```text
GPU lane:
  dense_graph_control

CPU lanes:
  mutation_discovered_pockets
  discovered_pockets_plus_router
  discovered_pockets_plus_limited_repair
```

The runner must write:

```text
progress.jsonl
hardware_heartbeat.jsonl
mutation_history_snapshots/*.json
training_history_snapshots/*.json
partial_aggregate_snapshot.json
```

## Decisions

Allowed decisions:

```text
e7h_mutation_discovers_reusable_pocket_granularity
e7h_pocket_boundaries_need_prior_scaffold
e7h_no_stable_pocket_granularity_detected
e7h_long_pipe_needed_for_this_family
e7h_pocket_discovery_collapses_to_graph_soup
e7h_discovered_pockets_need_limited_repair
e7h_leak_or_artifact_detected
e7h_no_clear_granularity_winner
```

## Checker Gates

The checker fails on missing artifacts, missing systems, missing row-level samples, missing accepted/rejected mutations, rollback mismatch, missing parameter diff/hash, mutation-only backprop or optimizer use, missing CUDA gradient history, leaked pocket IDs, missing freeze/reuse/repair report, failed deterministic replay, missing partial writeouts, missing hardware heartbeat, or broad claims in the artifact report.

## Boundary

E7H is a controlled symbolic/numeric pocket-boundary proxy. It does not test raw world learning or broad reasoning.

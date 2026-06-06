# E7B Pocket Routing Composition Probe Contract

## Purpose

E7B tests whether mutation/rollback can learn a compact router over frozen pocket outputs. This is different from raw mutation learning a full model from scratch.

The core question:

```text
Given reusable pockets, can mutation/rollback learn the switchboard that composes them into new task solutions?
```

## Runner And Checker

- Runner: `scripts/probes/run_e7b_pocket_routing_composition_probe.py`
- Checker: `scripts/probes/run_e7b_pocket_routing_composition_probe_check.py`
- Default artifact root: `target/pilot_wave/e7b_pocket_routing_composition_probe/`

## Systems

```text
monolithic_backprop_model
monolithic_mutation_model
frozen_pockets_gradient_router
frozen_pockets_mutation_router
frozen_pockets_binary_router
router_plus_limited_pocket_repair
random_router_control
oracle_pocket_router_reference
```

## Pocket Library

The probe uses frozen deterministic symbolic pockets to isolate the routing question:

```text
add
xor
memory
compare
branch
```

This probe does not claim that the pockets themselves were learned. It tests routing over frozen pocket semantics.

## Task Families

```text
add_then_compare
xor_then_compare
memory_add_then_compare
branch_apply_then_compare
memory_xor_then_compare
```

Each row has:

```text
target answer
target route
candidate pocket answers
misleading route distractor
family/context features
OOD/counterfactual/adversarial split controls
```

## Metrics

Required metrics:

```text
answer_accuracy
route_accuracy
composition_accuracy
shortcut_rate
usefulness_score
heldout_usefulness
ood_usefulness
counterfactual_usefulness
adversarial_usefulness
generalization_gap
parameter_count
mutation accepted/rejected/rollback
deterministic replay hash match
hardware heartbeat
```

## Hardware Rule

Evidence runs should use concurrent lanes:

```text
GPU lane:
  monolithic_backprop_model
  frozen_pockets_gradient_router

CPU lanes:
  monolithic_mutation_model
  frozen_pockets_mutation_router
  frozen_pockets_binary_router
  router_plus_limited_pocket_repair
```

The runner must write:

```text
progress.jsonl
hardware_heartbeat.jsonl
partial_status/*.json
mutation_history_snapshots/*.json
partial_aggregate_snapshot.json
```

## Decisions

Allowed decisions:

```text
e7b_mutation_router_composition_viable
e7b_gradient_only_composition_viable
e7b_monolithic_mutation_sufficient_or_task_too_easy
e7b_pocket_router_no_advantage_detected
e7b_router_leak_or_artifact_detected
```

## Checker Gates

The checker fails on missing artifacts, missing systems, missing seed rows, missing GPU gradient history, missing hardware heartbeat, mutation-only backprop/optimizer use, absent accepted/rejected mutation counts, rollback mismatch, missing row-level samples, failed leakage control without leak decision, deterministic replay mismatch, or broad claims in the report.

## Boundary

This is a controlled symbolic pocket-routing proxy. It does not prove learned pockets, broad reasoning, or model-scale behavior.

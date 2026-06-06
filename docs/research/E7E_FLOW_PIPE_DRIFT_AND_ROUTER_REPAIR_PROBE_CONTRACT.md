# E7E Flow-Pipe Drift And Router Repair Probe Contract

## Purpose

E7E is the falsification follow-up to E7D.

E7D supported:

```text
Router -> short pipe A -> Router -> short pipe B
```

E7E asks what happens when some short pipes drift, quantize, or become wrong:

```text
Can the router route around damaged pipes,
or is limited local pipe repair required?
```

## Runner And Checker

- Runner: `scripts/probes/run_e7e_flow_pipe_drift_and_router_repair_probe.py`
- Checker: `scripts/probes/run_e7e_flow_pipe_drift_and_router_repair_probe_check.py`
- Default artifact root: `target/pilot_wave/e7e_flow_pipe_drift_and_router_repair_probe/`

## Setup

The task reuses the E7D symbolic/numeric composition proxy. Each semantic primitive has two physical pipes:

```text
primary
backup
```

Per seed, the runner applies deterministic drift:

```text
route-around case:
  primary corrupted, backup clean

repair-required case:
  primary and backup both corrupted

light stress case:
  one pipe quantized
```

Rows still target the clean semantic answer. This tests whether systems preserve useful behavior after pipe damage.

## Systems

```text
damaged_primary_no_adaptation
router_routearound_mutation_only
router_plus_limited_pipe_repair
fused_long_pipe_repair_mutation
monolithic_gradient_drift_adapter
fused_long_pipe_gradient_adapter
random_route_control
oracle_routearound_reference
oracle_repair_reference
```

## Metrics

Required metrics:

```text
answer_accuracy
semantic_route_accuracy
composition_accuracy
damaged_pipe_hit_rate
routearound_rate
repair_use_rate
usefulness_score
heldout_usefulness
ood_usefulness
counterfactual_usefulness
adversarial_usefulness
generalization_gap
parameter_count
accepted/rejected mutations
rollback count
deterministic replay hash match
hardware heartbeat
```

## Hardware Rule

Evidence runs use concurrent lanes:

```text
GPU lane:
  monolithic_gradient_drift_adapter
  fused_long_pipe_gradient_adapter

CPU lanes:
  router_routearound_mutation_only
  router_plus_limited_pipe_repair
  fused_long_pipe_repair_mutation
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
e7e_router_routearound_sufficient
e7e_router_plus_limited_repair_preferred
e7e_fused_pipe_repair_more_robust
e7e_pipe_redundancy_insufficient
e7e_gradient_adapter_only_viable
e7e_leak_or_artifact_detected
e7e_no_clear_repair_strategy
```

## Checker Gates

The checker fails on missing artifacts, missing systems, missing drift profiles, missing row-level samples, missing GPU gradient history, no accepted/rejected mutations, rollback mismatch, mutation-only backprop/optimizer use, failed deterministic replay, failed leakage control without leak decision, missing partial writeouts, missing hardware heartbeat, or broad claims in the report.

## Boundary

E7E is a controlled symbolic/numeric flow-pipe drift proxy. It does not prove broad reasoning, language reasoning, consciousness, or model-scale behavior.

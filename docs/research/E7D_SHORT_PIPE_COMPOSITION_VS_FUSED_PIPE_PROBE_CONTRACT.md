# E7D Short-Pipe Composition Vs Fused-Pipe Probe Contract

## Purpose

E7D tests a topology question raised after E7B/E7C:

```text
Does the system need long fused AB pipes,
or can a recurrent router compose short reusable pocket pipes:

Router -> A -> Router -> B -> stop
```

This is a controlled symbolic/numeric proxy. It does not test open-ended reasoning.

## Runner And Checker

- Runner: `scripts/probes/run_e7d_short_pipe_composition_vs_fused_pipe_probe.py`
- Checker: `scripts/probes/run_e7d_short_pipe_composition_vs_fused_pipe_probe_check.py`
- Default artifact root: `target/pilot_wave/e7d_short_pipe_composition_vs_fused_pipe_probe/`

## Systems

```text
monolithic_matrix_core_gradient
monolithic_mutation_model
fused_long_pipe_gradient_router
fused_long_pipe_mutation_router
short_pipe_no_router_between
short_pipe_router_composition
router_plus_limited_pocket_repair
random_router_control
oracle_short_pipe_reference
```

## Task

Rows contain primitive operation tokens and raw 4-bit values. The target is a two-step composition:

```text
state0 = a
state1 = primitive_A(state0)
state2 = primitive_B(state1)
answer = state2 > threshold
route = A_then_B
```

Some rows use `branch_after_first`: the second primitive is chosen after seeing the first-pipe flow state. This separates:

```text
short_pipe_router_composition:
  recomputes branch after the first pipe

short_pipe_no_router_between:
  ignores first-pipe feedback
```

OOD rows hold out complete AB compositions while keeping each primitive operation seen. This tests whether short pipes compose better than pair-specific fused pipes.

## Metrics

Required metrics:

```text
answer_accuracy
route_accuracy
composition_accuracy
shortcut_rate
usefulness_score
step_penalized_usefulness
latency_steps
heldout_usefulness
ood_usefulness
counterfactual_usefulness
adversarial_usefulness
ood_route_accuracy
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
  monolithic_matrix_core_gradient
  fused_long_pipe_gradient_router

CPU lanes:
  monolithic_mutation_model
  fused_long_pipe_mutation_router
  short_pipe_no_router_between
  short_pipe_router_composition
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
e7d_short_pipe_router_flow_preferred
e7d_fused_long_pipe_required
e7d_hybrid_common_fused_plus_short_preferred
e7d_monolithic_sufficient_or_task_too_easy
e7d_router_overhead_failure
e7d_leak_or_artifact_detected
e7d_no_clear_topology_winner
```

## Checker Gates

The checker fails on missing artifacts, missing variants, missing OOD unseen-pair split, missing row-level samples, missing GPU gradient history, no accepted/rejected mutations, rollback mismatch, mutation-only backprop/optimizer use, failed deterministic replay, failed leakage control without leak decision, missing partial writeouts, missing hardware heartbeat, or broad claims in the report.

## Boundary

E7D tests a flow-router topology on a controlled symbolic/numeric proxy. It does not prove broad reasoning, language reasoning, consciousness, or model-scale behavior.

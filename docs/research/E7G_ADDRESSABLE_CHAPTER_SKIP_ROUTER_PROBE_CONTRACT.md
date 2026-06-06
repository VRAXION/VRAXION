# E7G Addressable Chapter Skip Router Probe Contract

## Purpose

E7G isolates the direct chapter-skip question after E7D/E7E.

E7D supported:

```text
Router -> short pipe -> Router -> short pipe
```

E7E supported route-around plus limited local repair under pipe drift.

E7G asks a narrower question:

```text
If useful chapters already exist,
can the router directly address only the needed chapters,
skip irrelevant chapters,
return after each call,
and halt without scanning a full pipe?
```

This is not pocket genesis. Chapter boundaries are already given.

## Runner And Checker

- Runner: `scripts/probes/run_e7g_addressable_chapter_skip_router_probe.py`
- Checker: `scripts/probes/run_e7g_addressable_chapter_skip_router_probe_check.py`
- Default artifact root: `target/pilot_wave/e7g_addressable_chapter_skip_router_probe/`

## Systems

```text
sequential_pipe_scan
fixed_short_pipe_router
fused_long_pipe_path_model
addressable_chapter_router_mutation
addressable_router_sparse_call_prior
dense_graph_soft_router_gradient
random_segment_walk_control
oracle_chapter_skip_reference
```

## Task

Rows contain:

```text
chapter library
requested chapter path
row context values
distractor chapters
target answer after executing the requested path
```

The critical comparison is:

```text
sequential scan:
  chapter 0 -> 1 -> 2 -> ... -> N

direct addressing:
  Router -> chapter 7 -> Router -> chapter 2 -> Router -> halt
```

OOD rows hold out transition families and include large backward jumps.
Counterfactual rows flip a requested chapter where possible.
Adversarial rows add misleading distractor chapters.

## Metrics

Required metrics:

```text
answer_accuracy
route_accuracy
skip_efficiency
irrelevant_branch_rate
overrun_rate
underrun_rate
loop_rate
mean_steps
path_validity
usefulness_score
heldout/OOD/counterfactual/adversarial usefulness
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
  dense_graph_soft_router_gradient

CPU lanes:
  addressable_chapter_router_mutation
  addressable_router_sparse_call_prior
```

The runner must write:

```text
progress.jsonl
hardware_heartbeat.jsonl
partial_status or equivalent snapshots
mutation_history_snapshots/*.json
training_history_snapshots/*.json
partial_aggregate_snapshot.json
```

## Decisions

Allowed decisions:

```text
e7g_addressable_chapter_skip_confirmed
e7g_sequential_scan_sufficient
e7g_fused_path_model_sufficient
e7g_dense_graph_soft_router_preferred
e7g_overbranching_or_loop_failure
e7g_leak_or_artifact_detected
e7g_no_clear_chapter_skip_winner
```

## Checker Gates

The checker fails on missing artifacts, missing systems, missing row-level samples, missing accepted/rejected mutations, rollback mismatch, missing parameter diff/hash, mutation-only backprop or optimizer use, missing CUDA gradient history, failed deterministic replay, failed leakage control without leak decision, missing partial writeouts, missing hardware heartbeat, or broad claims in the artifact report.

## Boundary

E7G is a controlled symbolic/numeric chapter-skip proxy. It tests direct addressing over already-existing chapters only.

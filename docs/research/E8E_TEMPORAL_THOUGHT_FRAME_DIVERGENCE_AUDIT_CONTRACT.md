# E8E Temporal Thought-Frame Divergence Audit Contract

## Purpose

`E8E_TEMPORAL_THOUGHT_FRAME_DIVERGENCE_AUDIT` is a diagnostic-only continuation after E8C/E8D. It keeps the same controlled symbolic/numeric Flow/RAM proxy and asks where learned pocket traces diverge from oracle thought-frame traces.

The core question:

```text
Does the learned system fail because it leaves the oracle trajectory early,
accumulates drift across route steps, corrupts consumer-sensitive cells,
or enters a different attractor-like RAM path?
```

## Non-Goals

E8E does not add a new architecture, router, semantic lane labels, bridge, image task, language task, or deployed-model claim. Oracle writes are allowed only in reference systems and explicit intervention arms.

## Systems

```text
oracle_trace_reference
current_best_learned_trace
consumer_distill_trace_reference
substrate_first_trace
mutation_only_trace
dense_graph_danger_trace
```

## Trace Model

For each row:

```text
Oracle:
  Flow_0 -> oracle_step_1 -> Flow_1 -> oracle_step_2 -> Flow_2 ...

Learned:
  Flow_0 -> learned_step_1 -> Flow_1' -> learned_step_2 -> Flow_2' ...
```

Every route step records frame similarity, delta similarity, consumer-read-mask error, result-cell error, support-cell sign mismatch, transition validity, first divergence, drift slope, local editability, and wrong-attractor/collapse indicators.

## Interventions

```text
oracle_reset_after_step_1
oracle_reset_after_each_step
learned_step_1_oracle_rest
oracle_step_1_learned_rest
one_learned_pocket_at_a_time
consumer_sensitive_cell_replacement_only
```

These arms are diagnostic and explicitly marked as oracle-assisted where applicable.

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
trace_divergence_report.json
intervention_report.json
attractor_report.json
local_editability_report.json
mutation_history_report.json
producer_dynamics_report.json
system_results.json
row_level_samples.json
aggregate_metrics.json
decision.json
summary.json
report.md
deterministic_replay.json
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
```

## Decision Labels

```text
e8e_first_step_write_divergence
e8e_temporal_drift_accumulation
e8e_consumer_sensitive_state_mismatch
e8e_recoverable_state_drift
e8e_wrong_attractor_trace
e8e_answer_shortcut_trace_invalid
```

## Checker Requirements

The checker fails on missing artifacts, missing systems, missing row-level trace data, missing intervention rows, missing mutation accept/reject/rollback history, deterministic replay mismatch, semantic lane label leakage, new-router flags, oracle writes in learned inference, or checker `failure_count != 0`.

## Boundary

E8E is a controlled symbolic/numeric pocket/RAM trace audit. It does not test raw language, images, AGI, consciousness, deployed-model behavior, or model scale.

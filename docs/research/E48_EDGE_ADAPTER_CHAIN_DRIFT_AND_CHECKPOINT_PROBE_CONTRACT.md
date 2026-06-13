# E48 Edge Adapter Chain Drift And Checkpoint Probe Contract

## Purpose

E48 follows E47. E47 showed that a single edge can be an explicit adapter
pocket. E48 asks whether several edge adapters can be chained without semantic
or mechanical drift.

Core question:

```text
Can a chain of Edge Adapter Pockets preserve the same state across multiple
node-local ABI transforms, or does the system need Agency checkpoints,
canonical resets, and trace validation between edges?
```

## Boundary

This is a controlled symbolic/numeric Edge ABI chain probe. It does not test
raw language reasoning, deployed assistant behavior, AGI, consciousness, or
model-scale behavior.

## Systems

```text
raw_wire_chain
single_edge_adapter_only
adapter_chain_no_checkpoint
adapter_chain_with_agency_checkpoint
adapter_chain_with_canonical_bus_reset
adapter_chain_plus_trace_validation
oracle_adapter_chain_reference
random_adapter_chain_control
```

## Task

Each node has its own local bit layout. The chain carries the same logical
intent through depths:

```text
depth = 1, 2, 3, 5
```

An edge adapter transforms:

```text
Node[i] local mini-matrix -> Node[i+1] local mini-matrix
```

The clean task requires final decoded intent to match the original intent.

Adversarial rows include:

```text
bit_flip    -> state drift after an edge
stale_trace -> stale/replayed edge state that can still look bit-valid
```

Correct adversarial behavior is `DEFER`, not a confident wrong commit.

## Metrics

Required metrics:

```text
heldout_success
OOD_success
counterfactual_success
adversarial_success
depth_1_success
depth_2_success
depth_3_success
depth_5_success
drift_rate
wrong_commit_rate
false_defer_rate
trace_mismatch_commit_rate
per_edge_adapter_success
old_intent_regression
accepted/rejected/rollback counts
parameter diff/hash
deterministic replay hash match
```

## Decisions

Allowed decisions:

```text
e48_adapter_chain_stable
e48_adapter_chain_requires_agency_checkpoint
e48_adapter_chain_requires_checkpoint_and_trace_validation
e48_canonical_bus_reset_required
e48_adapter_chain_drift_detected
e48_invalid_artifact_detected
```

Positive chain evidence requires:

```text
row-level eval
heldout/OOD/counterfactual/adversarial >= 0.95 for the winning valid system
wrong_commit_rate = 0
old_intent_regression <= 0.01
deterministic replay passes
checker failure_count = 0
```

## Required Artifacts

```text
backend_manifest.json
chain_drift_report.json
system_results.json
row_level_results.jsonl
adapter_chain_curve.jsonl
aggregate_metrics.json
deterministic_replay.json
decision.json
summary.json
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
results_table.md
report.md
```

## Sample Pack

The sample pack must live under:

```text
docs/research/artifact_samples/e48_edge_adapter_chain_drift_and_checkpoint_probe/
```

## Hard Requirements

```text
no gradient descent
no optimizer/backprop
no Python eval/SymPy/direct solver in learned systems
accepted/rejected mutation evidence for mutable systems
rollback count equals rejected count
target checker passes with failure_count = 0
sample-only checker passes
```

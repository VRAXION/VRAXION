# E48 Edge Adapter Chain Drift And Checkpoint Probe Result

## Decision

```text
decision = e48_adapter_chain_requires_checkpoint_and_trace_validation
target_checker_failure_count = 0
sample_only_checker_failure_count = 0
run_id = 7e6a8d1892753040
```

E48 tested whether E47-style Edge Adapter Pockets remain stable across a chain
of multiple node-local ABI transforms.

## Result Table

```text
| system | edge_type | heldout_success | ood_success | adversarial_success | depth_1_success | depth_2_success | depth_3_success | depth_5_success | drift_rate | wrong_commit_rate | trace_mismatch_commit_rate | per_edge_adapter_success |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw_wire_chain | raw_wire_chain | 0.028 | 0.049 | 0.000 | 0.069 | 0.028 | 0.014 | 0.056 | 0.972 | 0.967 | 0.100 | 0.200 |
| single_edge_adapter_only | single_adapter_then_raw | 0.278 | 0.264 | 0.000 | 1.000 | 0.028 | 0.014 | 0.062 | 0.786 | 0.779 | 0.100 | 0.325 |
| adapter_chain_no_checkpoint | adapter_chain_no_checkpoint | 0.257 | 0.521 | 0.000 | 0.035 | 0.028 | 1.000 | 0.750 | 0.989 | 0.637 | 0.100 | 0.150 |
| adapter_chain_with_agency_checkpoint | adapter_chain_with_checkpoint | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 0.100 | 0.100 | 0.100 | 1.000 |
| adapter_chain_with_canonical_bus_reset | adapter_chain_with_canonical_reset | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 0.100 | 0.100 | 0.100 | 1.000 |
| adapter_chain_plus_trace_validation | adapter_chain_with_checkpoint_and_trace | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.100 | 0.000 | 0.000 | 1.000 |
| oracle_adapter_chain_reference | oracle_adapter_chain_reference | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.100 | 0.000 | 0.000 | 1.000 |
| random_adapter_chain_control | random_adapter_chain_control | 0.000 | 0.035 | 0.000 | 0.035 | 0.000 | 0.007 | 0.042 | 0.989 | 0.983 | 0.100 | 0.100 |
```

## Interpretation

E48 split two different problems:

```text
1. ABI/bit drift through multiple adapters
2. stale or replayed trace that still looks bit-valid
```

Raw wiring and a single first-edge adapter were not enough:

```text
raw_wire_chain heldout = 0.028
single_edge_adapter_only heldout = 0.278
```

Training the whole chain only from end-to-end outcome was unstable:

```text
adapter_chain_no_checkpoint heldout = 0.257
wrong_commit_rate = 0.637
per_edge_adapter_success = 0.150
```

Local Agency checkpointing fixed the clean chain:

```text
adapter_chain_with_agency_checkpoint
heldout = 1.000
OOD = 1.000
depth_5 = 1.000
per_edge_adapter_success = 1.000
```

But checkpointing alone did not catch stale/replayed state:

```text
adversarial_success = 0.500
wrong_commit_rate = 0.100
trace_mismatch_commit_rate = 0.100
```

Adding trace validation closed that gap:

```text
adapter_chain_plus_trace_validation
heldout = 1.000
OOD = 1.000
adversarial = 1.000
wrong_commit_rate = 0.000
trace_mismatch_commit_rate = 0.000
```

The nonzero `drift_rate = 0.100` in the winning system is expected: adversarial
bit-flip rows intentionally create drift, and the valid behavior is to detect
and defer rather than commit.

## Architecture Lock

E48 updates the edge rule:

```text
Edge Adapter Pocket is valid for one edge.

Adapter chains must not be blind pipes.

Every edge hop needs:
  local Agency checkpoint
  trace validation
  rollback/defer on mismatch
```

Canonical reset was not enough by itself in this adversarial setup because a
stale trace can preserve a bit-valid payload while still being invalid history.

## Boundary

This is a controlled symbolic/numeric Edge ABI chain probe. It does not prove
raw language reasoning, deployed AI assistant behavior, model-scale behavior,
AGI, or consciousness.

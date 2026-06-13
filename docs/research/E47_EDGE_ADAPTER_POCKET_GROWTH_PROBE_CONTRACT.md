# E47 Edge Adapter Pocket Growth Probe Contract

## Purpose

E47 tests whether the edge between two frozen nodes should be treated as an
explicit adapter pocket instead of a passive wire bundle.

Core question:

```text
Can a small mutable edge adapter transform a source node mini-matrix into a
consumer node mini-matrix, and can that adapter grow by +1 active bit without
regressing old intents?
```

## Boundary

This is a controlled symbolic/numeric Edge ABI probe. It does not test raw
language reasoning, deployed assistant behavior, AGI, consciousness, or
model-scale behavior.

## Systems

```text
raw_wire_direct_fixed16
raw_wire_progressive_plus1
edge_adapter_fixed16_to16
edge_adapter_progressive_plus1_freeze_old
edge_adapter_progressive_plus1_no_freeze
edge_adapter_block_growth_plus4
identity_oracle_adapter_reference
random_adapter_control
```

## Task

Node A emits an active intent code into a source mini-matrix. The consumer
expects the same intent in canonical little-endian order. The source slots are
intentionally scrambled:

```text
logical target bit -> source slot
0 -> 2
1 -> 4
2 -> 1
3 -> 0
4 -> 3
5 -> 5
6 -> 6
7 -> 7
```

A raw wire reads source slot `j` as target slot `j`, so it should fail on this
scrambled edge. An edge adapter pocket can mutate a mapping:

```text
target_bit[j] = source_bit[adapter_mapping[j]]
```

## Growth

The progressive systems start with five active bits for 32 intents and grow to
eight active bits for 256 intents.

```text
start: 5 active bits
final: 8 active bits
growth: +1 active adapter cell or +4 block
```

## Metrics

Required metrics:

```text
heldout_success
OOD_success
counterfactual_success
adversarial_success
old_intent_success
old_intent_regression
bit_accuracy
source_to_target_transform_accuracy
wrong_commit_rate
false_ask_rate
growth_events
attempts_to_95
accepted/rejected/rollback counts
parameter diff/hash
deterministic replay hash match
```

## Decisions

Allowed decisions:

```text
e47_edge_adapter_pocket_positive
e47_raw_wire_growth_sufficient
e47_adapter_growth_overhead_too_high
e47_adapter_growth_causes_regression
e47_block_adapter_growth_preferred
e47_invalid_artifact_detected
```

Positive adapter evidence requires:

```text
raw wire controls fail
progressive adapter passes heldout/OOD/counterfactual/adversarial >= 0.95
old-intent regression <= 0.01
wrong_commit_rate = 0
deterministic replay passes
checker failure_count = 0
```

## Required Artifacts

```text
backend_manifest.json
adapter_growth_report.json
adapter_event_report.json
system_results.json
row_level_results.jsonl
adapter_curve.jsonl
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

The sample pack must include a replay-checkable subset under:

```text
docs/research/artifact_samples/e47_edge_adapter_pocket_growth_probe/
```

## Hard Requirements

```text
real row-level eval
no gradient descent
no optimizer/backprop
accepted/rejected mutation evidence for mutable adapters
rollback count equals rejected count
deterministic replay passes
target checker failure_count = 0
sample-only checker passes
```

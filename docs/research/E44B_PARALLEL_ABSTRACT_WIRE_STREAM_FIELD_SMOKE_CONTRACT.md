# E44B Parallel Abstract Wire Stream Field Smoke Contract

## Milestone

`E44B_PARALLEL_ABSTRACT_WIRE_STREAM_FIELD_SMOKE`

## Purpose

E44B tests the user's stricter wire-stream question:

```text
Wire#01: 01000111...
Wire#02: 01001010...
...
```

The goal is to sweep `wire_count x bits_per_wire` and produce a table showing
which stream shapes work.

This is a controlled symbolic/numeric Proposal ABI smoke probe. It is not a raw
language, AGI, consciousness, deployed-model, or model-scale claim.

## Task

The task contains 32 abstract intents. Each intent maps to a unique mechanical
target/value pair. This makes the minimum collision-free capacity 5 bits.

The tested shape is:

```text
wire_count x bits_per_wire
```

Examples:

```text
1 wire x 5 bits
5 wires x 1 bit
2 wires x 3 bits
3 wires x 2 bits
```

## Valid Proposal ABI

Each valid system uses:

```text
fixed mechanical header
+ anonymous parallel wire-stream payload
+ Agency validation before commit
```

The fixed header carries mechanics only:

```text
source
cycle
trace
evidence
ground compatibility
support completeness
action
```

The wire streams carry anonymous abstract payload bits. No wire is named as
truth, memory, confidence, answer, or result.

## Metrics

Primary metrics:

```text
agency_decision_success
trace_exact_rate
expected_commit_recovery
false_commit_rate
missed_commit_rate
capacity_bits
capacity_collision_rate
commit_target_value_accuracy
accepted/rejected/rollback mutation counts
deterministic replay hash match
```

## Decision Labels

```text
e44b_parallel_serial_capacity_detected
e44b_wire_shape_tradeoff_detected
e44b_wire_stream_unreliable
e44b_headerless_stream_unreliable
e44b_invalid_artifact_detected
```

## Positive Requirement

`e44b_parallel_serial_capacity_detected` requires:

```text
all shapes with wire_count * bits_per_wire < 5 fail
all shapes with wire_count * bits_per_wire >= 5 pass
headerless 5x1 control fails
random decoder 5x1 control fails
deterministic replay passes
checker failure_count = 0
sample-only checker passes
```

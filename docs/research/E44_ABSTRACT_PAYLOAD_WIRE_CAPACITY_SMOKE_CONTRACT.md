# E44 Abstract Payload Wire Capacity Smoke Contract

## Milestone

`E44_ABSTRACT_PAYLOAD_WIRE_CAPACITY_SMOKE`

## Purpose

E44 tests a narrow architecture question after E43:

```text
If Proposal Field writes use a fixed mechanical header, how many anonymous
abstract payload wires are needed before the Agency Field can safely decode
the intended target/value without literal semantic channels?
```

This is a controlled symbolic/numeric Proposal ABI smoke probe. It is not a
raw language, deployed model, AGI, consciousness, or model-scale claim.

## Fixed Mechanical Header

The fixed header carries mechanical validity information only:

```text
active
action_code
source_pocket_id
cycle_id
trace_ref
evidence_support
ground_compat
support_complete
payload_width
```

The fixed header must not carry literal target/value content in valid learned
systems.

## Abstract Payload Wires

The payload wires are anonymous bits. They may encode a learned abstract intent,
but no wire is named as answer, truth, memory, target, confidence, or result.

The valid learned systems sweep payload widths:

```text
0, 1, 2, 3, 4, 6, 8
```

The task contains 16 abstract intents. Widths below 4 cannot uniquely encode all
intents and should expose a capacity break.

## Compared Systems

```text
oracle_abstract_wire_reference
literal_target_value_header_reference
no_fixed_header_payload_only_w4
fixed_header_no_payload_w0
abstract_payload_w1
abstract_payload_w2
abstract_payload_w3
abstract_payload_w4
abstract_payload_w6
abstract_payload_w8
random_payload_decoder_control
```

`literal_target_value_header_reference` is a reference/control only. It is not a
valid learned architecture because it places the literal target/value in the
header.

## Required Behaviors

Rows cover:

```text
valid_commit
toxic_wrong_payload
stale_replay
ground_conflict
trace_mismatch
partial_support
no_valid_proposal
```

The Agency decision must:

```text
commit valid fresh supported proposals
reject stale/toxic/ground-conflicting/trace-mismatched proposals
ask on partial support
defer when no valid proposal exists
avoid direct Flow writes
```

## Metrics

Primary metrics:

```text
agency_decision_success
action_accuracy
trace_exact_rate
false_commit_rate
missed_commit_rate
commit_target_value_accuracy
expected_commit_recovery
toxic_rejection_accuracy
stale_rejection_accuracy
ground_conflict_rejection
trace_mismatch_rejection
partial_support_ask_accuracy
no_valid_defer_accuracy
uses_fixed_header_rate
payload_collision_rate
accepted/rejected/rollback mutation counts
parameter diff/hash
deterministic replay hash match
```

## Decision Labels

```text
e44_abstract_payload_wire_capacity_detected
e44_fixed_header_only_sufficient
e44_literal_payload_required
e44_abstract_payload_unreliable
e44_invalid_artifact_detected
```

## Positive Requirement

`e44_abstract_payload_wire_capacity_detected` requires:

```text
fixed header without payload does not pass
widths 1, 2, and 3 do not pass
width 4 passes
payload-only without fixed header does not pass safe Agency decisions
deterministic replay passes
checker failure_count = 0
sample-only checker passes
```

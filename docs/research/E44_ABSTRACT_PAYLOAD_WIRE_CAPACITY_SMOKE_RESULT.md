# E44 Abstract Payload Wire Capacity Smoke Result

## Decision

```text
decision = e44_abstract_payload_wire_capacity_detected
minimal_passing_wire_width = 4
run_id = 4b1bc103c17d9a38
target_checker_failure_count = 0
sample_only_checker_failure_count = 0
```

E44 confirms the narrow ABI result: with a fixed mechanical Proposal header,
anonymous abstract payload wires can carry learned intent reliably once the
payload has enough collision-free capacity. For the 16-intent smoke task, the
first passing width was 4 bits.

## Primary Wire Sweep

```text
system                         success   trace     collision  false_commit
fixed_header_no_payload_w0     0.571429  0.571429  1.000000   0.000000
abstract_payload_w1            0.571429  0.571429  1.000000   0.000000
abstract_payload_w2            0.571429  0.571429  1.000000   0.000000
abstract_payload_w3            0.571429  0.571429  1.000000   0.000000
abstract_payload_w4            1.000000  1.000000  0.000000   0.000000
abstract_payload_w6            1.000000  1.000000  0.000000   0.000000
abstract_payload_w8            1.000000  1.000000  0.000000   0.000000
no_fixed_header_payload_only_w4 0.000000 0.000000  0.000000   0.000000
random_payload_decoder_control 0.571429  0.571429  0.000000   0.395833
```

## Confirm Seeds

```text
seed 44022: e44_abstract_payload_wire_capacity_detected, width 4
seed 44023: e44_abstract_payload_wire_capacity_detected, width 4
seed 44024: e44_abstract_payload_wire_capacity_detected, width 4
```

## Interpretation

The fixed header is not the meaning channel. It is the safety and compatibility
channel:

```text
active / action / source / cycle / trace / evidence / ground / support
```

The payload wires are anonymous. They only need enough capacity for the learned
decoder to separate abstract intents. In this task, 1, 2, and 3 payload wires
collided; 4 payload wires separated all 16 intents.

The no-header control failed even with width 4. This means payload capacity is
not sufficient by itself: Agency still needs fixed mechanical validity metadata.

The random decoder control also failed with high false commit rate, so this is
not a "many wires solve it automatically" result. The payload code must be
learned or matched to the Proposal ABI.

## Architecture Lock

E44 supports this low-level lock:

```text
Pocket / LogicAtom
  -> fixed mechanical Proposal header
  -> anonymous abstract payload bits
  -> Proposal Field
  -> Agency Field validation and commit
```

Use fixed channels for mechanics, not semantic labels. Use anonymous payload
wires for learned abstract intent. For this smoke family, 4 payload wires are
the minimum viable width.

## Boundary

This is a controlled symbolic/numeric Proposal ABI smoke. It does not prove raw
language reasoning, deployed AI assistant behavior, model-scale behavior, AGI,
or consciousness.

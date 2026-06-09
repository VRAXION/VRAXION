# E12 Temporal-Coded Input To Flow Solver Confirm Contract

## Purpose

`E12_TEMPORAL_CODED_INPUT_TO_FLOW_SOLVER_CONFIRM` tests the cleanest input path
for the Binary Flow Matrix / pocket pipeline runtime:

```text
anonymous temporal binary event stream
-> binary Flow state
-> scheduled pocket/operator blocks
-> gated writeback
-> output stream
```

The probe is not text parsing, natural-language training, or a neural network
test. The runtime must not receive privileged semantic slots. Debug family names
may exist in harness reports only.

## Search-First Result

Before adding E12, the repo and fetched branches were searched for:

```text
E12
TEMPORAL_CODED_INPUT
temporal coded input
anonymous temporal input
event stream
binary event stream
Flow input lane
temporal Flow solver
input pulse
temporal codebook
randomized codebook
sequence input
output stream decoder
text to binary flow
anonymous binary flow encoding
```

Only adjacent TokenEvent / stream-tokenizer references and an E11C placeholder
for a different latency benchmark were found. No equivalent E12 milestone was
present, so this milestone is created under the requested name.

## Input Model

Each tick contains only binary fields:

```text
clock bit
start/end boundary bit
separator bit
payload bits
noise bits
anonymous structural bits
```

The runtime receives binary codes and temporal structure only. Per-seed
codebooks randomize payload and operator codes, including heldout and
counterfactual codebook splits.

## Task Families

The harness generates these debug families:

```text
COPY_SEQUENCE
REVERSE_SEQUENCE
REWRITE_MAP
CONDITIONAL_CONTROL
DELAYED_BINDING
NOISY_REPAIR
BRANCH_SWITCH
COUNTERFACTUAL_CODEBOOK
```

The runtime sees anonymous binary program codes and structural lanes, not the
family names.

## Systems

```text
OBSERVED_STREAM_DIRECT_BASELINE
STATIC_CODEBOOK_LOOKUP_CONTROL
TEMPORAL_FLOW_NO_GATE
TEMPORAL_FLOW_GATED_WRITEBACK
TEMPORAL_FLOW_GATED_WITH_TRACE_REPAIR
TEMPORAL_FLOW_SCHEDULED_POCKET_PRIMARY
TEMPORAL_FLOW_PRUNED_SCHEDULED_POCKET_PRIMARY
TINY_SEQUENCE_MLP_CONTROL
```

The primary candidate is:

```text
TEMPORAL_FLOW_PRUNED_SCHEDULED_POCKET_PRIMARY
```

## Metrics

```text
input_retention_accuracy
temporal_order_accuracy
output_sequence_accuracy
exact_task_success_rate
trace_validity
delta_validity
binding_accuracy
conditional_control_accuracy
rewrite_map_accuracy
noise_rejection_rate
decoy_rejection_rate
branch_switch_accuracy
heldout_codebook_accuracy
randomized_codebook_generalization
wrong_writeback_rate
destructive_overwrite_rate
branch_contamination_rate
stale_write_rejection_rate
gate_false_accept_rate
gate_false_reject_rate
temporal_drift_rate
oscillation_rate
attractor_collapse_rate
cost_per_tick
deterministic_replay_passed
no_semantic_slot_leak_detected
no_neural_dependency_detected
no_overclaim_boundary_preserved
```

## Positive Gate

The primary passes only if:

```text
exact_task_success_rate >= 0.95
output_sequence_accuracy >= 0.98
trace_validity >= 0.95
temporal_order_accuracy >= 0.98
binding_accuracy >= 0.95
conditional_control_accuracy >= 0.95
rewrite_map_accuracy >= 0.95
noise_rejection_rate >= 0.90
decoy_rejection_rate >= 0.90
heldout_codebook_accuracy >= 0.90
randomized_codebook_generalization >= 0.90
wrong_writeback_rate <= 0.02
destructive_overwrite_rate <= 0.02
branch_contamination_rate == 0
no_semantic_slot_leak_detected == true
deterministic_replay_passed == true
```

## Decisions

Allowed decisions:

```text
e12_temporal_coded_input_to_flow_solver_confirmed
e12_input_retention_or_temporal_order_failure
e12_output_stream_decode_failure
e12_trace_validity_failure
e12_binding_or_rewrite_failure
e12_noise_or_decoy_repair_failure
e12_codebook_generalization_failure
e12_writeback_safety_failure
e12_semantic_slot_leak_detected
e12_invalid_or_incomplete_run
```

## Boundary

E12 is a deterministic synthetic binary temporal-event-stream probe. It does
not claim deployed behavior, natural-language behavior, broad model-scale
behavior, or hardware speedup.

# E12 Temporal-Coded Input To Flow Solver Confirm Result

Status: completed.

## Decision

```text
decision = e12_temporal_coded_input_to_flow_solver_confirmed
next = E13_STREAMING_MULTI_STEP_FLOW_COMPOSITION_CONFIRM
primary_system = TEMPORAL_FLOW_PRUNED_SCHEDULED_POCKET_PRIMARY
positive_gate_passed = true
deterministic_replay_passed = true
checker_failure_count = 0
```

Run root:

```text
target/pilot_wave/e12_temporal_coded_input_to_flow_solver_confirm/
```

## What Was Tested

E12 tests the full anonymous input path:

```text
temporal binary event stream
-> Flow input/buffer state
-> pattern-triggered pocket blocks
-> gated writeback
-> output stream decoder
```

The runtime receives binary tick fields only. Debug task family names are kept
in the harness/report layer.

## Key Metrics

| system | exact | output | trace | order | bind | rewrite | noise | decoy | wrong | branch | cost/tick |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OBSERVED_STREAM_DIRECT_BASELINE | 0.287 | 0.395 | 0.435 | 0.395 | 0.000 | 0.000 | 0.000 | 0.000 | 0.583 | 0.018 | 5.704 |
| STATIC_CODEBOOK_LOOKUP_CONTROL | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 28.518 |
| TEMPORAL_FLOW_NO_GATE | 0.787 | 0.853 | 0.900 | 0.853 | 1.000 | 1.000 | 0.000 | 0.000 | 0.212 | 0.000 | 20.913 |
| TEMPORAL_FLOW_GATED_WRITEBACK | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 22.814 |
| TEMPORAL_FLOW_GATED_WITH_TRACE_REPAIR | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 24.715 |
| TEMPORAL_FLOW_SCHEDULED_POCKET_PRIMARY | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 14.259 |
| TEMPORAL_FLOW_PRUNED_SCHEDULED_POCKET_PRIMARY | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 10.456 |
| TINY_SEQUENCE_MLP_CONTROL | 0.412 | 0.534 | 0.566 | 0.534 | 0.000 | 0.000 | 0.000 | 0.000 | 0.420 | 0.000 | 33.271 |

Primary details:

```text
input_retention_accuracy = 1.000
temporal_order_accuracy = 1.000
output_sequence_accuracy = 1.000
exact_task_success_rate = 1.000
trace_validity = 1.000
binding_accuracy = 1.000
conditional_control_accuracy = 1.000
rewrite_map_accuracy = 1.000
branch_switch_accuracy = 1.000
heldout_codebook_accuracy = 1.000
randomized_codebook_generalization = 1.000
noise_rejection_rate = 1.000
decoy_rejection_rate = 1.000
wrong_writeback_rate = 0.000
destructive_overwrite_rate = 0.000
branch_contamination_rate = 0.000
```

Positive-gate deltas:

```text
exact_success_delta_vs_direct = +0.712500
trace_validity_delta_vs_direct = +0.565132
cost_reduction_vs_no_gate = 0.500000
```

## Family Coverage

The primary reached exact success on every generated family:

```text
COPY_SEQUENCE = 1.000
REVERSE_SEQUENCE = 1.000
REWRITE_MAP = 1.000
CONDITIONAL_CONTROL = 1.000
DELAYED_BINDING = 1.000
NOISY_REPAIR = 1.000
BRANCH_SWITCH = 1.000
COUNTERFACTUAL_CODEBOOK = 1.000
```

## Split Robustness

```text
heldout_codebook exact = 1.000, trace = 1.000
noisy exact = 1.000, trace = 1.000
adversarial exact = 1.000, trace = 1.000
randomized_codebook exact = 1.000, trace = 1.000
```

## Semantic-Leak Audit

```text
runtime_stream_contains_only_bits = true
runtime_receives_forbidden_semantic_slots = false
debug_names_confined_to_harness_reports = true
no_semantic_slot_leak_detected = true
```

The stream tick fields are:

```text
clock
boundary
separator
payload_bits
noise_bits
anonymous structural bits
```

## Interpretation

E12 confirms the clean proxy input path: anonymous temporal binary pulses can be
accumulated into Flow state, processed by scheduled pocket/operator blocks, and
decoded into correct output streams under randomized codebooks, branch
switches, delayed binding, rewrite maps, and noise/decoy corruption.

The result should still be read as a deterministic synthetic proxy. It does not
prove natural-language parsing, deployed behavior, or hardware latency.

## Verification

```text
python3 scripts/probes/run_e12_temporal_coded_input_to_flow_solver_confirm.py
python3 scripts/probes/run_e12_temporal_coded_input_to_flow_solver_confirm_check.py --out target/pilot_wave/e12_temporal_coded_input_to_flow_solver_confirm --write-summary
```

The checker passed with `failure_count = 0`.

Boundary: E12 is a deterministic synthetic binary temporal-event-stream probe
only. It does not make natural-language, deployment, model-scale, or hardware
speedup claims.

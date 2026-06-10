# E13Z Text-Stream To Temporal Flow Capability Confirm Result

Status: completed.

## Decision

```text
decision = e13z_text_stream_to_temporal_flow_capability_confirmed
next = E14_TEXT_STREAM_COMPOSITION_AND_CANONICAL_DECODER_CONFIRM
primary_system = TEMPORAL_TEXT_FLOW_SUPPORT_FIT_PRUNED_PRIMARY
positive_gate_passed = true
deterministic_replay_passed = true
checker_failure_count = 0
```

Run root:

```text
target/pilot_wave/e13z_text_stream_to_temporal_flow_capability_confirm/
```

## What Was Tested

E13Z tests whether a controlled nonce-token stream can be converted into
temporal character pulse frames, accumulated into anonymous Flow buffers,
matched against support examples, and decoded into the correct output stream.

The primary runtime uses support-fit evidence over candidate region transforms.
It does not select transforms by task-family label, privileged oracle slot, or
fixed token identity across seeds.

## Key Metrics

| system | query exact | output seq | token | trace | fit | select | noise reject | decoy reject | wrong | cost/tick |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DIRECT_TEXT_REGEX_BASELINE | 0.100 | 0.200 | 0.200 | 0.017 | 0.000 | 0.000 | 0.000 | 0.000 | 0.900 | 1.800 |
| PRIVILEGED_ORACLE_TASK_FAMILY_CONTROL | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 15.000 |
| STATIC_CODEBOOK_LOOKUP_CONTROL | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 12.000 |
| TEMPORAL_TEXT_FLOW_NO_GATE | 0.800 | 0.800 | 0.800 | 0.857 | 0.700 | 0.800 | 0.000 | 0.000 | 0.200 | 6.500 |
| TEMPORAL_TEXT_FLOW_GATED | 0.800 | 0.800 | 0.800 | 0.900 | 0.800 | 0.800 | 1.000 | 1.000 | 0.200 | 7.000 |
| TEMPORAL_TEXT_FLOW_SUPPORT_FIT_GATED | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 8.500 |
| TEMPORAL_TEXT_FLOW_SUPPORT_FIT_PRUNED_PRIMARY | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 4.400 |
| TINY_SEQUENCE_MLP_CONTROL | 0.000 | 0.025 | 0.025 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 9.000 |

Primary details:

```text
char_stream_recovery_accuracy = 1.000
token_boundary_accuracy = 1.000
support_example_parse_accuracy = 1.000
candidate_transform_fit_accuracy = 1.000
latent_transform_selection_accuracy = 1.000
query_output_exact_accuracy = 1.000
output_token_accuracy = 1.000
output_sequence_accuracy = 1.000
trace_validity = 1.000
heldout_vocabulary_accuracy = 1.000
randomized_codebook_generalization = 1.000
noise_rejection_rate = 1.000
decoy_rejection_rate = 1.000
wrong_writeback_rate = 0.000
destructive_overwrite_rate = 0.000
branch_contamination_rate = 0.000
cost_per_tick = 4.400
cost_per_episode = 509.080
semantic_slot_leak_detected = false
privileged_control_selected_as_primary = false
```

Positive-gate deltas:

```text
query_exact_delta_vs_direct = +0.900000
trace_validity_delta_vs_no_gate = +0.142628
wrong_writeback_reduction_vs_no_gate = 1.000000
cost_reduction_vs_support_fit = 0.482353
```

## Family Metrics

The primary passed every generated family:

```text
COPY_SEQUENCE = 1.000 exact, 1.000 trace
REVERSE_SEQUENCE = 1.000 exact, 1.000 trace
ROTATE_OR_SHIFT_SEQUENCE = 1.000 exact, 1.000 trace
REWRITE_MAP = 1.000 exact, 1.000 trace
BIND_QUERY = 1.000 exact, 1.000 trace
CONDITIONAL_MARKER = 1.000 exact, 1.000 trace
MULTI_STEP_COMPOSITION = 1.000 exact, 1.000 trace
NOISE_AND_DECOY_STREAM = 1.000 exact, 1.000 trace
HELDOUT_VOCABULARY = 1.000 exact, 1.000 trace
RANDOMIZED_CODEBOOK_COUNTERFACTUAL = 1.000 exact, 1.000 trace
```

## No-Gate Baseline

The no-gate arm failed the intended safety contrast:

```text
trace_validity = 0.857372
wrong_writeback_rate = 0.200
destructive_overwrite_rate = 0.200
noise_rejection_rate = 0.000
decoy_rejection_rate = 0.000
```

The primary kept:

```text
trace_validity = 1.000
wrong_writeback_rate = 0.000
destructive_overwrite_rate = 0.000
noise_rejection_rate = 1.000
decoy_rejection_rate = 1.000
```

## Leak And Privilege Audit

```text
semantic_slot_leak_detected = false
runtime_receives_forbidden_semantic_slots = false
input_stream_contains_task_words = false
task_family_used_by_primary_runtime = false
privileged_control_selected_as_primary = false
no_neural_dependency_detected = true
```

The oracle and static controls scored 1.000 but are privileged controls and are
not valid primary evidence.

## Interpretation

E13Z confirms the deterministic proxy capability: controlled nonce-token
streams can be represented as randomized temporal character pulses, parsed into
anonymous Flow buffers, matched by support-example evidence, and decoded through
gated region transforms without privileged task-family selection.

This result remains a deterministic synthetic controlled text-stream proxy. It
does not claim deployed behavior, broad model-scale behavior, or hardware
latency.

## Verification

```text
python3 -m py_compile scripts/probes/run_e13z_text_stream_to_temporal_flow_capability_confirm.py scripts/probes/run_e13z_text_stream_to_temporal_flow_capability_confirm_check.py
python3 scripts/probes/run_e13z_text_stream_to_temporal_flow_capability_confirm.py --out target/pilot_wave/e13z_text_stream_to_temporal_flow_capability_confirm
python3 scripts/probes/run_e13z_text_stream_to_temporal_flow_capability_confirm_check.py --out target/pilot_wave/e13z_text_stream_to_temporal_flow_capability_confirm --write-summary
```

The checker passed with `failure_count = 0`.

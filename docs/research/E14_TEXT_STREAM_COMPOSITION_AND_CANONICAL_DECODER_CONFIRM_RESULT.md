# E14 Text-Stream Composition And Canonical Decoder Confirm Result

Status: completed.

## Decision

```text
decision = e14_text_stream_composition_and_canonical_decoder_confirmed
next = E15_TEXT_STREAM_LONG_HORIZON_MEMORY_AND_REPAIR_CONFIRM
primary_system = COMPOSITION_FLOW_PRUNED_GATED_CANONICAL_DECODER_PRIMARY
positive_gate_passed = true
deterministic_replay_passed = true
checker_failure_count = 0
```

Run root:

```text
target/pilot_wave/e14_text_stream_composition_and_canonical_decoder_confirm/
```

## What Was Tested

E14 tests controlled nonce-token streams that require transform-chain
composition and canonical decoding. The primary recovers pulse streams into
anonymous Flow regions, fits transform chains from support evidence, executes
through guarded lanes, emits a canonical output object, and renders only from
that object.

## Key Metrics

| system | query | comp | chain | order | decoder | render | trace | wrong | cost/tick |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DIRECT_TEXT_REGEX_BASELINE | 0.000 | 0.000 | 0.000 | 0.250 | 0.000 | 1.000 | 0.538 | 1.000 | 1.700 |
| PRIVILEGED_ORACLE_CHAIN_CONTROL | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 16.000 |
| STATIC_CODEBOOK_LOOKUP_CONTROL | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 12.000 |
| E13Z_SINGLE_TRANSFORM_FALLBACK | 0.000 | 0.000 | 0.000 | 0.250 | 0.000 | 1.000 | 0.574 | 1.000 | 5.700 |
| COMPOSITION_FLOW_NO_GATE | 0.833 | 0.833 | 0.917 | 0.917 | 0.833 | 1.000 | 0.912 | 0.167 | 8.100 |
| COMPOSITION_FLOW_GATED | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 8.700 |
| COMPOSITION_FLOW_GATED_CANONICAL_DECODER | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 9.300 |
| COMPOSITION_FLOW_PRUNED_GATED_CANONICAL_DECODER_PRIMARY | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 4.800 |
| RENDERER_ORACLE_CHEAT_CONTROL | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.857 | 1.000 | 16.500 |
| TINY_SEQUENCE_MLP_CONTROL | 0.000 | 0.000 | 0.000 | 0.250 | 0.000 | 1.000 | 0.571 | 1.000 | 9.000 |

Primary details:

```text
char_stream_recovery_accuracy = 1.000
token_boundary_accuracy = 1.000
support_example_parse_accuracy = 1.000
support_consistency_detection_accuracy = 1.000
candidate_transform_fit_accuracy = 1.000
latent_transform_selection_accuracy = 1.000
transform_chain_selection_accuracy = 1.000
chain_order_accuracy = 1.000
chain_step_accuracy = 1.000
composition_exact_accuracy = 1.000
order_sensitive_pair_accuracy = 1.000
heldout_chain_composition_accuracy = 1.000
ambiguous_case_abstain_or_repair_accuracy = 1.000
canonical_output_schema_validity = 1.000
canonical_decoder_exact_accuracy = 1.000
output_sequence_accuracy = 1.000
output_token_accuracy = 1.000
query_output_exact_accuracy = 1.000
renderer_faithfulness = 1.000
renderer_oracle_leak_detected = false
trace_validity = 1.000
wrong_writeback_rate = 0.000
destructive_overwrite_rate = 0.000
branch_contamination_rate = 0.000
heldout_vocab_accuracy = 1.000
randomized_codebook_generalization = 1.000
average_chain_length = 1.750
pruned_operator_count = 6.000
cost_per_tick = 4.800
cost_per_episode = 787.600
```

Positive-gate deltas:

```text
query_exact_delta_vs_direct = +1.000000
composition_delta_vs_single = +1.000000
trace_validity_delta_vs_no_gate = +0.087895
wrong_writeback_reduction_vs_no_gate = 1.000000
cost_reduction_vs_decoder = 0.483871
```

## Task Family Metrics

Primary exact and trace were 1.000 on every family:

```text
TWO_STEP_REVERSE_THEN_MAP
TWO_STEP_MAP_THEN_REVERSE
ROTATE_THEN_MAP
MAP_THEN_ROTATE
BIND_THEN_QUERY_THEN_MAP
CONDITIONAL_COMPOSITION
MULTI_SUPPORT_CHAIN_SELECTION
AMBIGUOUS_SUPPORT_ABSTAIN_OR_REPAIR
NOISE_AND_DECOY_COMPOSITION
HELDOUT_CHAIN_COMPOSITION
RANDOMIZED_CODEBOOK_COUNTERFACTUAL
CANONICAL_DECODER_STRESS
```

## Baseline Contrasts

The E13Z single-transform fallback failed composition and order-sensitive cases:

```text
composition_exact_accuracy = 0.000
order_sensitive_pair_accuracy = 0.000
canonical_decoder_exact_accuracy = 0.000
wrong_writeback_rate = 1.000
```

The no-gate composition arm showed the intended safety gap:

```text
trace_validity = 0.912105
wrong_writeback_rate = 0.166667
destructive_overwrite_rate = 0.166667
gate_false_accept_rate = 1.000
```

The renderer cheat control produced correct oracle-backed outputs but was
detected as invalid:

```text
renderer_faithfulness = 0.000
renderer_oracle_leak_detected = true
```

## Leak And Privilege Audit

```text
semantic_slot_leak_detected = false
runtime_receives_forbidden_semantic_slots = false
task_family_used_by_primary_runtime = false
chain_id_used_by_primary_runtime = false
renderer_oracle_access_in_primary_runtime = false
privileged_control_selected_as_primary = false
```

## Interpretation

E14 confirms the deterministic proxy capability for multi-step controlled
text-stream composition plus canonical decoding. The primary can select and
execute transform chains, handle ambiguous support by abstain/repair policy,
reject decoys through gated writeback, produce a valid canonical object, and
render faithfully from that object.

This result remains a deterministic synthetic controlled text-stream
composition proxy. It does not claim deployed behavior, broad model-scale
behavior, or hardware latency.

## Verification

```text
python3 -m py_compile scripts/probes/run_e14_text_stream_composition_and_canonical_decoder_confirm.py scripts/probes/run_e14_text_stream_composition_and_canonical_decoder_confirm_check.py
python3 scripts/probes/run_e14_text_stream_composition_and_canonical_decoder_confirm.py --out target/pilot_wave/e14_text_stream_composition_and_canonical_decoder_confirm
python3 scripts/probes/run_e14_text_stream_composition_and_canonical_decoder_confirm_check.py --out target/pilot_wave/e14_text_stream_composition_and_canonical_decoder_confirm --write-summary
```

The checker passed with `failure_count = 0`.

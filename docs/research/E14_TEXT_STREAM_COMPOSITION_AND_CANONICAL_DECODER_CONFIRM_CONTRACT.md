# E14 Text-Stream Composition And Canonical Decoder Confirm Contract

## Purpose

`E14_TEXT_STREAM_COMPOSITION_AND_CANONICAL_DECODER_CONFIRM` parks the region
parallel scheduler path and tests a capability extension:

```text
controlled nonce-token stream
-> temporal character pulse Flow input
-> support-query evidence fit
-> transform chain proposal
-> gated Flow execution
-> canonical output object
-> faithful renderer
```

The probe is deterministic and synthetic. It is not a deployed runtime
benchmark, not a regex parser, not a task-family oracle route, and not a neural
training run.

## Search-First Result

Before adding E14, the repo and fetched refs were searched for:

```text
E14_TEXT_STREAM_COMPOSITION
TEXT_STREAM_COMPOSITION_AND_CANONICAL_DECODER
canonical decoder
trace renderer
faithful renderer
renderer faithfulness
transform composition
multi step composition
support fit composition
text stream composition
E13Z
temporal flow decoder
output schema
canonical output object
semantic leak audit
privileged renderer
```

No equivalent E14 implementation was found. The only relevant local hits were
the E13Z next pointer and adjacent non-equivalent output-schema references in
older stable-loop work.

## Runtime Model

The primary receives temporal character/pulse frames only. It must:

```text
recover anonymous token regions
parse support/query regions
fit candidate transforms
select and execute a transform chain
write through guarded lanes
decode a canonical output object
render only from that object
```

The primary runtime must not receive semantic slots, task-family labels,
chain-id labels, oracle expected output, or renderer access to hidden expected
answers.

## Canonical Output Object

The decoder emits:

```json
{
  "status": "ok | ambiguous | repair_failed",
  "output_tokens": [],
  "trace_id": "...",
  "chain_length": 0,
  "confidence": 0.0,
  "decoder_validity": true,
  "error_code": null
}
```

The renderer may only verbalize canonical object fields. It may not repair or
replace output tokens.

## Task Families

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

## Systems

```text
DIRECT_TEXT_REGEX_BASELINE
PRIVILEGED_ORACLE_CHAIN_CONTROL
STATIC_CODEBOOK_LOOKUP_CONTROL
E13Z_SINGLE_TRANSFORM_FALLBACK
COMPOSITION_FLOW_NO_GATE
COMPOSITION_FLOW_GATED
COMPOSITION_FLOW_GATED_CANONICAL_DECODER
COMPOSITION_FLOW_PRUNED_GATED_CANONICAL_DECODER_PRIMARY
RENDERER_ORACLE_CHEAT_CONTROL
TINY_SEQUENCE_MLP_CONTROL
```

The primary candidate is:

```text
COMPOSITION_FLOW_PRUNED_GATED_CANONICAL_DECODER_PRIMARY
```

The oracle, static lookup, and renderer-cheat controls are privileged and
invalid as primary systems.

## Pocket Field Equivalents

```text
pocket_id
read_mask
write_mask
guard_mask
trace_mask
transform_op
chain_step_index
evidence_fit_score
confidence
cost
reason_code
expected_version_hash
```

## Metrics

```text
char_stream_recovery_accuracy
token_boundary_accuracy
support_example_parse_accuracy
support_consistency_detection_accuracy
candidate_transform_fit_accuracy
latent_transform_selection_accuracy
transform_chain_selection_accuracy
chain_order_accuracy
chain_step_accuracy
composition_exact_accuracy
order_sensitive_pair_accuracy
heldout_chain_composition_accuracy
ambiguous_case_abstain_or_repair_accuracy
canonical_output_schema_validity
canonical_decoder_exact_accuracy
output_sequence_accuracy
output_token_accuracy
query_output_exact_accuracy
renderer_faithfulness
renderer_oracle_leak_detected
trace_validity
wrong_writeback_rate
destructive_overwrite_rate
branch_contamination_rate
stale_write_rejection_rate
gate_false_accept_rate
gate_false_reject_rate
temporal_drift_rate
cost_per_tick
cost_per_episode
average_chain_length
pruned_operator_count
semantic_slot_leak_detected
privileged_control_selected_as_primary
deterministic_replay_passed
checker_failure_count
```

## Positive Gate

The primary passes only if:

```text
query_output_exact_accuracy >= 0.90
output_sequence_accuracy >= 0.94
output_token_accuracy >= 0.97
canonical_output_schema_validity >= 0.99
canonical_decoder_exact_accuracy >= 0.95
renderer_faithfulness == 1.00
renderer_oracle_leak_detected == false
trace_validity >= 0.95
candidate_transform_fit_accuracy >= 0.90
latent_transform_selection_accuracy >= 0.90
transform_chain_selection_accuracy >= 0.88
chain_order_accuracy >= 0.90
chain_step_accuracy >= 0.92
composition_exact_accuracy >= 0.88
order_sensitive_pair_accuracy >= 0.90
heldout_chain_composition_accuracy >= 0.82
ambiguous_case_abstain_or_repair_accuracy >= 0.80
noise_decoy_composition_accuracy >= 0.82
heldout_vocab_accuracy >= 0.85
randomized_codebook_generalization >= 0.85
wrong_writeback_rate <= 0.02
destructive_overwrite_rate <= 0.02
branch_contamination_rate == 0
semantic_slot_leak_detected == false
privileged_control_selected_as_primary == false
deterministic_replay_passed == true
```

The primary must beat the direct baseline on query exact accuracy, the E13Z
single-transform fallback on composition and order-sensitive cases, and the
no-gate arm on trace validity and wrong writeback.

## Decisions

Allowed decisions:

```text
e14_text_stream_composition_and_canonical_decoder_confirmed
e14_input_recovery_failure
e14_support_parse_failure
e14_transform_chain_selection_failure
e14_chain_order_failure
e14_decoder_failure
e14_renderer_faithfulness_failure
e14_ambiguous_case_failure
e14_noise_decoy_failure
e14_codebook_generalization_failure
e14_semantic_slot_leak_detected
e14_writeback_safety_failure
e14_invalid_or_incomplete_run
```

Confirmed next:

```text
E15_TEXT_STREAM_LONG_HORIZON_MEMORY_AND_REPAIR_CONFIRM
```

## Boundary

E14 is a deterministic synthetic controlled text-stream composition proxy only.
It does not claim deployed behavior, broad model-scale behavior, or hardware
speedup.

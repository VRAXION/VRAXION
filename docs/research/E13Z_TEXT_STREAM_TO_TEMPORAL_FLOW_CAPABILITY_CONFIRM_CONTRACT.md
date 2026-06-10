# E13Z Text-Stream To Temporal Flow Capability Confirm Contract

## Purpose

`E13Z_TEXT_STREAM_TO_TEMPORAL_FLOW_CAPABILITY_CONFIRM` parks the E14 parallel
scheduler work and tests a capability extension instead:

```text
controlled nonce-token stream
-> randomized character pulse frames over time
-> anonymous temporal Flow state
-> support-example evidence fit
-> candidate region transform proposal
-> gated writeback
-> decoded output stream
```

The probe is deterministic and synthetic. It is not a deployed behavior claim,
not a regex parser claim, and not a neural training run.

## Search-First Result

Before adding E13Z, the repo and fetched refs were searched for:

```text
E13Z
TEXT_STREAM_TO_TEMPORAL_FLOW
text stream temporal flow
controlled text stream
nonsense token
nonce token
support query
text to flow
temporal character stream
temporal text stream
randomized vocabulary
codebook generalization
semantic slot leak
canonical decoder
flow text capability
E14_TEXT
E12_TEXT
```

No equivalent milestone or overlapping implementation was found. The only
nearby context is the existing E12 temporal coded input probe and E13 streaming
grid transition probe.

## Input Model

Episodes are generated as support-query examples with random nonce tokens and a
per-seed randomized character codebook. The debug text is human-readable, but
the primary runtime receives only temporal pulse frames:

```text
start pulse
character code pulses
boundary pulses
token separator pulses
support/example delimiter pulses
query delimiter pulse
guard/noise pulses
end pulse
```

The runtime does not receive semantic slots or instruction tokens. Debug family
names are allowed only in generator, evaluator, and report artifacts.

## Episode Families

```text
COPY_SEQUENCE
REVERSE_SEQUENCE
ROTATE_OR_SHIFT_SEQUENCE
REWRITE_MAP
BIND_QUERY
CONDITIONAL_MARKER
MULTI_STEP_COMPOSITION
NOISE_AND_DECOY_STREAM
HELDOUT_VOCABULARY
RANDOMIZED_CODEBOOK_COUNTERFACTUAL
```

The primary must infer the active transform by fitting candidate pocket
proposals to support examples, then apply the selected transform to the query.

## Systems

```text
DIRECT_TEXT_REGEX_BASELINE
PRIVILEGED_ORACLE_TASK_FAMILY_CONTROL
STATIC_CODEBOOK_LOOKUP_CONTROL
TEMPORAL_TEXT_FLOW_NO_GATE
TEMPORAL_TEXT_FLOW_GATED
TEMPORAL_TEXT_FLOW_SUPPORT_FIT_GATED
TEMPORAL_TEXT_FLOW_SUPPORT_FIT_PRUNED_PRIMARY
TINY_SEQUENCE_MLP_CONTROL
```

The primary candidate is:

```text
TEMPORAL_TEXT_FLOW_SUPPORT_FIT_PRUNED_PRIMARY
```

The oracle and static controls are privileged and invalid as primary systems.

## Pocket Field Equivalents

The implementation reports these equivalent proposal fields:

```text
pocket_id
read_mask
write_mask
guard_mask
trace_mask
transform_op
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
candidate_transform_fit_accuracy
latent_transform_selection_accuracy
query_output_exact_accuracy
output_token_accuracy
output_sequence_accuracy
copy_family_accuracy
reverse_family_accuracy
rotate_shift_family_accuracy
rewrite_map_family_accuracy
bind_query_family_accuracy
conditional_marker_family_accuracy
multi_step_composition_accuracy
noise_rejection_rate
decoy_rejection_rate
heldout_vocabulary_accuracy
randomized_codebook_generalization
trace_validity
wrong_writeback_rate
destructive_overwrite_rate
branch_contamination_rate
stale_write_rejection_rate
gate_false_accept_rate
gate_false_reject_rate
temporal_drift_rate
oscillation_rate
cost_per_tick
cost_per_episode
deterministic_replay_passed
semantic_slot_leak_detected
privileged_control_selected_as_primary
checker_failure_count
```

## Positive Gate

The primary passes only if:

```text
query_output_exact_accuracy >= 0.92
output_sequence_accuracy >= 0.95
output_token_accuracy >= 0.98
trace_validity >= 0.95
char_stream_recovery_accuracy >= 0.98
token_boundary_accuracy >= 0.98
candidate_transform_fit_accuracy >= 0.90
latent_transform_selection_accuracy >= 0.90
copy_family_accuracy >= 0.95
reverse_family_accuracy >= 0.95
rotate_shift_family_accuracy >= 0.90
rewrite_map_family_accuracy >= 0.90
bind_query_family_accuracy >= 0.90
conditional_marker_family_accuracy >= 0.85
multi_step_composition_accuracy >= 0.80
noise_rejection_rate >= 0.85
decoy_rejection_rate >= 0.85
heldout_vocabulary_accuracy >= 0.85
randomized_codebook_generalization >= 0.85
wrong_writeback_rate <= 0.02
destructive_overwrite_rate <= 0.02
branch_contamination_rate == 0
semantic_slot_leak_detected == false
privileged_control_selected_as_primary == false
deterministic_replay_passed == true
```

The primary must also beat:

```text
DIRECT_TEXT_REGEX_BASELINE on query_output_exact_accuracy
TEMPORAL_TEXT_FLOW_NO_GATE on trace_validity
TEMPORAL_TEXT_FLOW_NO_GATE on wrong_writeback_rate
```

## Decisions

Allowed decisions:

```text
e13z_text_stream_to_temporal_flow_capability_confirmed
e13z_text_stream_input_recovery_failure
e13z_token_boundary_failure
e13z_support_fit_failure
e13z_transform_selection_failure
e13z_output_decoder_failure
e13z_noise_decoy_failure
e13z_codebook_generalization_failure
e13z_semantic_slot_leak_detected
e13z_writeback_safety_failure
e13z_invalid_or_incomplete_run
```

Confirmed next:

```text
E14_TEXT_STREAM_COMPOSITION_AND_CANONICAL_DECODER_CONFIRM
```

## Required Artifacts

```text
decision.json
summary.json
aggregate_metrics.json
report.md
e13z_search_report.json
e13z_input_stream_report.json
e13z_vocab_codebook_report.json
e13z_support_query_episode_report.json
e13z_system_comparison_report.json
e13z_task_family_report.json
e13z_trace_validity_report.json
e13z_writeback_safety_report.json
e13z_noise_decoy_report.json
e13z_heldout_generalization_report.json
e13z_semantic_leak_audit_report.json
e13z_deterministic_replay_report.json
e13z_boundary_claims_report.json
```

## Boundary

E13Z is a deterministic synthetic controlled text-stream proxy only. It does
not claim deployed behavior, broad model-scale behavior, or hardware speedup.

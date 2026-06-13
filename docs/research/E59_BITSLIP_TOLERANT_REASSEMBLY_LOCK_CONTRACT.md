# E59 Bit-Slip Tolerant Reassembly Lock Contract

## Purpose

E59 locks the binary ingress fix exposed by E58:

```text
continuous bitstream
-> multi-hypothesis frame reassembly
-> START/LENGTH/CRC/END validation
-> decoded_feature == requested_feature guard
-> ambiguity guard
-> Agency commit only after validation
```

The test is a controlled symbolic/numeric ingress probe. It is not a raw
language, AGI, consciousness, deployed model, or model-scale claim.

## Systems

```text
strict_single_offset_full_guard
end_marker_only_decoder
loose_start_only_decoder
multi_offset_crc_no_feature_guard
multi_offset_crc_requested_no_ambiguity_guard
bitslip_tolerant_reassembly_lock
oracle_frame_reference
random_control
```

## Stages

```text
P0_clean_packet
P1_noise_with_crc
P2_continuous_decoy_false_start
P3_single_bit_insert_before_frame
P4_single_bit_drop_before_frame
P5_payload_slip_with_repeated_frame
P6_adversarial_sync_decoy_before_valid
P7_wrong_feature_valid_crc_only
P8_truncated_packet_must_defer
P9_conflicting_duplicate_frames_must_defer
```

## Protocol

Frame format:

```text
START_SYNC
LENGTH
PAYLOAD
CRC
END_SYNC
```

Payload layout:

```text
feature_id[5] + value[1] + trust[1] + nonce[4]
```

Commit is valid only if all mechanical guards pass:

```text
START_SYNC matches
LENGTH matches payload length
CRC matches
END_SYNC matches
trust bit is valid
decoded_feature == requested_feature
no conflicting valid requested-feature values
```

EOF/END alone is explicitly insufficient.

## Required Artifacts

```text
backend_manifest.json
ingress_protocol_manifest.json
row_level_results.jsonl
system_results.json
stage_metrics.json
reassembly_report.json
false_frame_report.json
requested_feature_guard_report.json
ambiguity_guard_report.json
reassembly_examples.json
failure_examples.json
aggregate_metrics.json
decision.json
summary.json
deterministic_replay.json
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
report.md
```

Sample pack:

```text
docs/research/artifact_samples/e59_bitslip_tolerant_reassembly_lock/
```

## Metrics

```text
closed_loop_success
bitslip_recovery
trace_exact
false_commit_rate
false_frame_commit_rate
wrong_feature_write_rate
false_defer_rate
ambiguity_reject_rate
candidate_count
crc_pass_count
requested_match_count
net_utility
deterministic_replay_match
checker_failure_count
```

## Decision Labels

```text
e59_bitslip_tolerant_reassembly_locked
e59_reassembly_still_false_frame_limited
e59_requested_feature_guard_required
e59_ambiguity_guard_required
e59_invalid_artifact_detected
```

Positive requires:

```text
locked closed-loop success >= 0.995
locked bit-slip recovery >= 0.995
locked false-frame commit <= 0.001
locked wrong-feature write <= 0.001
locked ambiguity reject >= 0.995
strict single-offset control exposes bit-slip failure
no-feature control exposes wrong-feature failure
no-ambiguity control exposes false commit failure
loose/EOF controls expose false-frame risk
deterministic replay passes
checker failure_count = 0
```

## Boundary

This locks a binary ingress reassembly rule inside the current VRAXION
engineering stack. It does not claim natural language generation, general
reasoning, consciousness, deployment readiness, or model-scale behavior.

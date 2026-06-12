# E34D Continuous Binary Stream Framing And Resync Probe Contract

## Summary

E34D continues E34C by attacking the remaining failure: continuous binary stream framing and false-frame commits.

Core question:

```text
Can explicit packet protocol hygiene remove the E34C false-frame failures without changing the Flow/Pocket reasoning layer?
```

## Systems

```text
start_only_baseline
start_end_marker
start_length_end
start_length_crc_end
crc_end_requested_feature_guard
multi_frame_resync_guard
first_sync_shortcut_control
oracle_framing_reference
```

## Splits

```text
packet_clean
packet_noise_10
continuous_stream
continuous_bit_insert
continuous_bit_drop
adversarial_sync_decoy
```

## Required Metrics

```text
closed_loop_success
answer_correct
trace_exact
accepted_flow_write_accuracy
frame_sync_accuracy
false_frame_commit_rate
wrong_feature_write_rate
wrong_confident_answer
avg_steps
accepted/rejected/rollback mutations
deterministic replay
checker failure count
```

## Decision Labels

```text
e34d_framing_resync_guard_positive
e34d_crc_guard_positive_but_resync_brittle
e34d_requested_feature_guard_positive
e34d_eof_length_crc_partial
e34d_framing_still_bottleneck
e34d_shortcut_or_task_artifact_detected
e34d_artifact_invalid
```

## Requirements

```text
real row-level eval
mutation/rollback policy path
progress and heartbeat writeout
deterministic replay
sample-only checker
no gradient descent
no optimizer/backprop
no AGI/consciousness/model-scale claims
```

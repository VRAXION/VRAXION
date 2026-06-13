# E58 Standard IO Regression Binary/Text/Egress Confirm Contract

## Purpose

E58 reruns the standard IO path after the Text Field and Egress Field locks:

```text
binary ingress
text ingress
Agency commit
multi-resolution egress
```

Core questions:

```text
1. Does the current standard path still fail on binary bit slip?
2. Does the bitslip-tolerant reassembly candidate close that gap?
3. Do Text Field mode selection and multi-resolution Egress remain clean?
4. What do concrete multi-resolution outputs look like?
```

## Boundary

E58 is a deterministic integrated IO regression. It is not a raw language
reasoning, AGI, consciousness, deployment, or model-scale claim.

## Systems

```text
legacy_standard_before_io_locks
current_standard_without_bitslip_reassembly
current_standard_with_bitslip_reassembly_candidate
loose_start_only_unsafe
direct_pocket_output_unsafe
oracle_reference
random_control
```

## Stages

```text
B0_binary_packet_clean
B1_binary_packet_noise_10
B2_binary_continuous_decoy
B3_binary_bit_insert_slip
B4_binary_bit_drop_slip
T0_noisy_text_answerable
T1_text_unresolved_must_ask
T2_text_boundary_multiframe
T3_real_like_weak_contrast
O0_multires_output_consistency
O1_stale_proposal_output_attack
```

## Required Example Artifacts

```text
multi_resolution_examples.json
failure_examples.json
```

The multi-resolution examples must show:

```text
compact output
short output
long output
consistency hash
```

## Metrics

```text
closed_loop_success
binary_success
bitslip_success
text_success
egress_success
trace_exact
multi_resolution_consistency
false_commit_rate
wrong_confident_rate
false_ask_rate
stale_output_leak_rate
net_utility
accepted/rejected/rollback counts
deterministic replay
```

## Decisions

```text
e58_standard_path_passes_with_bitslip_reassembly_candidate
e58_standard_path_still_bitslip_limited
e58_text_or_egress_regression_detected
e58_unsafe_shortcut_or_stale_output_detected
e58_invalid_artifact_detected
```

## Positive Criteria

```text
current_standard_with_bitslip_reassembly_candidate closed_loop_success >= 0.98
candidate bitslip_success >= 0.98
candidate text_success >= 0.98
candidate egress_success >= 0.98
candidate false_commit_rate <= 0.01
candidate stale_output_leak_rate <= 0.01
current_standard_without_bitslip_reassembly bitslip_success < 0.70
direct_pocket_output_unsafe exposes stale proposal leakage
checker failure_count = 0
sample-only checker passes
```

## Required Artifacts

```text
backend_manifest.json
standard_io_manifest.json
row_level_results.jsonl
system_results.json
stage_metrics.json
binary_bitslip_report.json
text_regression_report.json
egress_examples_report.json
multi_resolution_examples.json
failure_examples.json
training_history.json
aggregate_metrics.json
decision.json
summary.json
deterministic_replay.json
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
report.md
```

## Hard Requirements

```text
row-level eval
deterministic replay
heartbeat/progress/partial artifacts
no gradient descent
no optimizer/backprop
target checker failure_count = 0
sample-only checker passes
```

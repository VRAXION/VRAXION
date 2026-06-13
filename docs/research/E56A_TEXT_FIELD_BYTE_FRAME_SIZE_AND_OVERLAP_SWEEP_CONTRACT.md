# E56A Text Field Byte Frame Size And Overlap Sweep Contract

## Purpose

E56A tests the Text Field / Byte Field idea after E55 localized the current
pre-monolith text ingress frontier.

Core question:

```text
Does representing incoming text as raw UTF-8 byte frames improve evidence
extraction, and what frame size / overlap is the smallest useful default?
```

## Boundary

This is a controlled Text Field ingress sweep. It does not claim raw open-ended
language reasoning, AGI, consciousness, deployment quality, or model-scale
behavior.

## Architecture Under Test

```text
External text
-> UTF-8 bytes
-> Text Field / Byte Field [frame_count, frame_bytes, 8]
-> Text Ingress Lens Pockets
-> Proposal Field
-> Agency Field
-> Flow/Ground commit only after validation
```

Direct Text Pocket writes to Flow are not allowed.

## Systems

```text
legacy_direct_text_ingress_baseline
text_field_single_64
text_field_single_128
text_field_single_256
text_field_single_512
text_field_4x128_overlap0
text_field_4x128_overlap16
text_field_4x128_overlap32
text_field_4x128_overlap64
keyword_shortcut_control
oracle_text_field_reference
```

## Stages

```text
T0_short_controlled_observation
T1_boundary_split_contrast
T2_adversarial_contrast_clause
T3_real_like_weak_text
T4_long_multisentence_decoy
T5_utf8_accent_noise
```

## Metrics

```text
success
answer_correct
trace_exact
false_commit_rate
wrong_confident_rate
boundary_failure_rate
bytes_processed_per_decision
stress_success
overlap_gain_vs_no_overlap
```

## Decisions

```text
e56a_text_field_byte_frame_positive
e56a_overlap_required_for_boundary_robustness
e56a_large_frame_required
e56a_text_field_no_advantage
e56a_invalid_artifact_detected
```

## Positive Requirements

```text
best Text Field stress_success improves by >= 0.20 over legacy
best Text Field false_commit_rate = 0.0
best Text Field wrong_confident_rate = 0.0
keyword_shortcut_control fails visibly
oracle reference is ceiling
target checker failure_count = 0
sample-only checker passes
deterministic replay passes
```

## Required Artifacts

```text
backend_manifest.json
text_field_schema.json
frame_sweep_manifest.json
stage_generation_report.json
row_level_results.jsonl
frame_sweep_results.json
system_results.json
stage_metrics.json
boundary_failure_report.json
recommendation_report.json
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
no gradient descent
no optimizer/backprop
row-level eval
deterministic replay
progress + heartbeat writeouts
sample pack under docs/research/artifact_samples/
```

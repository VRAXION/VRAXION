# E56C Text Field Mode Selection Adversarial Probe Contract

## Purpose

E56B showed that no single clean Text Field max fits inside the requested
1x-3x slowdown gate. E56C tests the next architectural claim:

```text
Do not lock one universal Text Field max.
Lock multiple Text Field modes and let Agency/Router choose the smallest safe
mode for the current evidence footprint.
```

## Boundary

E56C is a deterministic adversarial mode-selection probe. It does not claim raw
language reasoning, AGI, consciousness, deployment quality, or model-scale
behavior.

## Modes

```text
FAST_DEFAULT_4x128_o32
  unique coverage ~= 416 byte
  work byte = 512
  slowdown = 1.0x

LONG_CAPPED_5x256_o64
  unique coverage ~= 1024 byte
  work byte = 1280
  slowdown = 2.75x

CLEAN_LONG_4x512_o128
  unique coverage ~= 1664 byte
  work byte = 2048
  slowdown = 4.5x

ASK_OR_MULTI_CYCLE
  correct when visible evidence is insufficient or one clean frame is too small
```

## Systems

```text
always_fast_default
always_long_capped
always_clean_long
naive_length_router
clean_long_without_cost_guard
three_mode_agency_router
oracle_mode_selector
random_mode_control
```

## Adversarial Stages

```text
A0_short_answerable
A1_boundary_overlap_answerable
A2_medium_needs_long_capped
A3_long_clean_required
A4_long_lure_relevant_early
A5_missing_evidence_must_ask
A6_oversize_requires_multi_cycle
A7_adversarial_decoy_requires_clean
```

The adversarial focus is:

```text
long input where relevant evidence is early
missing evidence where answering is unsafe
oversize input where one frame is insufficient
adversarial decoy rows where clean integrity is required
```

## Metrics

```text
success
mode_accuracy
trace_exact
false_commit_rate
wrong_confident_rate
false_ask_rate
overpay_rate
mean_cost
net_utility
adversarial_success
adversarial_mode_accuracy
adversarial_false_commit_rate
```

## Positive Criteria

```text
three_mode_agency_router success >= 0.98
three_mode_agency_router mode_accuracy >= 0.98
three_mode_agency_router adversarial_success >= 0.98
false_commit_rate <= 0.01
overpay_rate <= 0.01
net utility matches oracle within 0.02
net utility beats naive length router and always-clean controls by >= 0.10
checker failure_count = 0
sample-only checker passes
deterministic replay passes
```

## Decisions

```text
e56c_three_mode_agency_selector_adversarial_confirmed
e56c_single_clean_long_mode_cost_overfit_detected
e56c_length_router_insufficient_under_adversarial_mix
e56c_clean_long_required_as_default
e56c_mode_policy_unresolved
e56c_invalid_artifact_detected
```

## Required Artifacts

```text
backend_manifest.json
mode_selection_manifest.json
row_level_results.jsonl
system_results.json
stage_metrics.json
adversarial_report.json
mode_policy_recommendation.json
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
no gradient descent
no optimizer/backprop
target checker failure_count = 0
sample-only checker passes
no direct raw-language/model-scale claim
```

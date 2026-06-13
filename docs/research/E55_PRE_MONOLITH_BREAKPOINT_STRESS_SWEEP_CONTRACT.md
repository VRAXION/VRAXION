# E55 Pre-Monolith Breakpoint Stress Sweep Contract

## Purpose

E55 runs before merging the current components into a unified native runtime.

Core question:

```text
Where does the current Flow/Pocket + Proposal/Agency + Pocket Library line
still break under a staged stress sweep?
```

This is not the monolith integration. It is a falsification sweep to avoid
merging unknown bottlenecks into one larger runtime.

## Boundary

E55 is a controlled symbolic/noisy-text/binary/proposal-library stress sweep.
It does not claim raw open-ended language reasoning, AGI, consciousness,
deployment quality, or model-scale behavior.

## Systems

```text
current_pre_monolith_stack
shortcut_or_raw_commit_control
oracle_reference
```

## Stages

```text
S0_symbolic_controlled_evidence
S1_noisy_text_controlled
S2_adversarial_text_contrast
S3_real_like_weak_text
S4_missing_evidence_information_seeking
S5_binary_packet_clean
S6_binary_packet_noise10
S7_binary_continuous_guarded
S8_binary_bit_slip_resync
S9_proposal_agency_adversarial
S10_persistent_library_governance
```

## Required Artifacts

```text
backend_manifest.json
stress_sweep_manifest.json
stage_generation_report.json
row_level_results.jsonl
stage_metrics.json
system_results.json
breakpoint_sweep_report.json
bottleneck_localization_report.json
adversarial_stress_report.json
aggregate_metrics.json
decision.json
summary.json
deterministic_replay.json
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
report.md
```

## Sample Pack

The sample pack must live under:

```text
docs/research/artifact_samples/e55_pre_monolith_breakpoint_stress_sweep/
```

It must contain sample row-level data, sample metrics, replay report, schema,
manifest, and sample-only checker result.

## Decisions

```text
e55_pre_monolith_breakpoints_localized
e55_pre_monolith_text_frontier_still_open
e55_pre_monolith_binary_resync_frontier_open
e55_pre_monolith_all_sweep_clean
e55_pre_monolith_core_regression_detected
e55_invalid_artifact_detected
```

## Pass Requirements

Required clean stages must stay above `0.95` success:

```text
S0_symbolic_controlled_evidence
S1_noisy_text_controlled
S4_missing_evidence_information_seeking
S5_binary_packet_clean
S6_binary_packet_noise10
S7_binary_continuous_guarded
S9_proposal_agency_adversarial
S10_persistent_library_governance
```

At least one frontier stage must expose a breakpoint:

```text
S3_real_like_weak_text
S8_binary_bit_slip_resync
```

## Hard Requirements

```text
no gradient descent
no optimizer/backprop
row-level eval
deterministic replay
progress.jsonl
hardware_heartbeat.jsonl
target checker failure_count = 0
sample-only checker passes
```

# E57 Output Egress Field Multi-Resolution Renderer Probe Contract

## Purpose

E56C locked Text Field mode selection for input. E57 tests the mirrored output
side:

```text
Agency ANSWER_READY / ACT
-> Egress Field
-> output renderer / codec
-> bytes / text / action
```

Core question:

```text
Can Agency-committed state be rendered into compact, short, long, and
multi-resolution output fields without direct Pocket proposal leakage?
```

## Boundary

E57 is a deterministic output/egress field probe. It does not claim raw language
reasoning, AGI, consciousness, deployment quality, or model-scale behavior.

## Output Modes

```text
COMPACT_ACTION
  1x32 byte action field

SHORT_TEXT_1x256
  short byte/text output field

LONG_TEXT_4x256
  longer text/trace output field

MULTI_RES_COMPACT_SHORT_LONG
  compact action + short text + long/detail trace output

ASK_OR_NEED_MORE_INFO
  unresolved output action
```

## Systems

```text
compact_only_output
short_text_only_output
long_text_only_output
direct_pocket_to_text_unsafe
naive_length_egress_router
agency_committed_single_resolution
agency_committed_multi_resolution_renderer
oracle_egress_reference
random_output_control
```

## Stages

```text
R0_compact_action_only
R1_short_text_answer
R2_long_trace_answer
R3_multires_summary_plus_detail
R4_unresolved_must_ask
R5_stale_proposal_leak_attack
R6_utf8_boundary_text
R7_long_input_compact_answer
```

## Metrics

```text
success
render_accuracy
mode_accuracy
multi_resolution_write_success
byte_reconstruction_valid
utf8_valid
trace_backed_output
false_output_rate
wrong_confident_output_rate
false_ask_rate
stale_proposal_leak_rate
overpay_rate
mean_cost
net_utility
```

## Positive Criteria

```text
agency_committed_multi_resolution_renderer success >= 0.98
mode_accuracy >= 0.98
multi_resolution_write_success >= 0.98
false_output_rate <= 0.01
stale_proposal_leak_rate <= 0.01
trace_backed_output >= 0.98
net utility matches oracle within 0.02
net utility beats single-resolution renderer by >= 0.08
checker failure_count = 0
sample-only checker passes
deterministic replay passes
```

## Decisions

```text
e57_multi_resolution_egress_renderer_confirmed
e57_single_resolution_output_sufficient
e57_output_stale_proposal_leak_detected
e57_output_renderer_policy_unresolved
e57_invalid_artifact_detected
```

## Required Artifacts

```text
backend_manifest.json
egress_mode_manifest.json
row_level_results.jsonl
system_results.json
stage_metrics.json
multi_resolution_report.json
egress_policy_recommendation.json
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
output reads only Agency-committed state in primary systems
direct Pocket-to-text exists only as unsafe control
target checker failure_count = 0
sample-only checker passes
```

# E33 Bridge Breakpoint Saturation Ladder Contract

Status: implemented probe contract.

## Purpose

E33 answers the direct question:

```text
Up to which exact step is the Flow/Pocket system clean, and where does it first break?
```

Unlike E31/E32 mixed runs, E33 trains and evaluates every difficulty step separately. This separates:

```text
controlled toy task success
controlled text success
paraphrase/decoy/long-context stress
weak mined real-text failure
```

Important interpretation rule:

```text
If the ladder has no clean S0/S1 region, E33 does not prove that the
previous E24/E27 checked system was broken. It means this isolated
saturation harness failed to reproduce the earlier clean regime and must
be treated as a harness/baseline mismatch finding.
```

## Systems

```text
small_workspace_d96
large_workspace_d192
large_workspace_trace_focus_d192
oracle_text_interpreter_d96
random_static_control
```

`oracle_text_interpreter_d96` is diagnostic only.

## Steps

```text
S0_structured_events_no_text
S1_clean_symbolic_sentences
S2_naturalized_templates
S3_paraphrase_variation
S4_decoy_dense_text
S5_temporal_order_shuffle
S6_missing_info_minimal_pairs
S7_long_context_evidence
S8_indirect_language
S9_weak_mined_real_text
```

Clean means:

```text
resolution_success >= 0.98
trace_exact >= 0.98
```

Perfect means:

```text
resolution_success == 1.0
trace_exact == 1.0
```

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
saturation_plan.json
saturation_ladder_report.json
training_curve_report.json
row_level_results.jsonl
system_results.json
aggregate_metrics.json
decision.json
summary.json
deterministic_replay.json
resource_usage_report.json
report.md
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
```

## Decisions

```text
e33_controlled_bridge_clean_until_real_text_break
e33_breaks_before_real_text
e33_capacity_bottleneck_before_text
e33_ingress_codec_bottleneck_before_text
e33_weak_real_text_data_bottleneck_localized
e33_no_clean_saturation_detected
e33_artifact_invalid
```

## Boundary

E33 is a controlled saturation bridge probe. It does not claim raw language reasoning, AGI, consciousness, deployment quality, or model-scale behavior.

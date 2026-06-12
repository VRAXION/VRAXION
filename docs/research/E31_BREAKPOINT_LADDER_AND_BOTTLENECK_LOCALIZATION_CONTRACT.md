# E31 Breakpoint Ladder And Bottleneck Localization Contract

Status: implemented probe contract.

## Purpose

E31 localizes where the current Flow/Pocket unresolved-state line breaks after E27-E30A.

The probe asks whether the first major failure is primarily:

```text
Ingress Codec / text-to-event translation
Trace Ledger / evidence-span binding
capacity
training objective or data shape
multiple interacting bottlenecks
```

## Systems

```text
baseline_text_ingress_d96_p8
capacity_flow_d192_p8
capacity_pockets_d96_p16
oracle_ingress_d96_p8
oracle_evidence_span_d96_p8
random_static_control
```

`oracle_ingress_d96_p8` and `oracle_evidence_span_d96_p8` are diagnostic controls only. They are not valid primary learned systems.

## Breakpoint Ladder

```text
R0_explicit_controlled_evidence
R1_final_mixed_canonical
R2_naturalized_text_canonical
R3_paraphrase_variation
R4_decoy_density
R5_temporal_disorder
R6_unresolved_answerable_minimal_pairs
R7_long_context_evidence_span
R8_indirect_implication_language
R9_mined_real_text_weak_labels
```

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
breakpoint_ladder_report.json
bottleneck_localization_report.json
oracle_ingress_report.json
oracle_evidence_span_report.json
capacity_sweep_report.json
training_curve_report.json
row_level_results.jsonl
trace_ledger.jsonl
flow_field_snapshot.json
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

## Metrics

Per-system and per-rung:

```text
resolution_success
action_accuracy
trace_exact
trace_bit_accuracy
wrong_confident_answer_on_unresolved
false_ask_on_answerable
```

Localization deltas:

```text
oracle_ingress_resolution_delta
oracle_ingress_trace_exact_delta
oracle_span_resolution_delta
oracle_span_trace_exact_delta
capacity_flow_resolution_delta
capacity_flow_trace_exact_delta
capacity_pocket_resolution_delta
capacity_pocket_trace_exact_delta
```

## Decisions

```text
e31_ingress_codec_bottleneck_localized
e31_trace_ledger_bottleneck_localized
e31_capacity_bottleneck_localized
e31_objective_or_training_bottleneck_localized
e31_no_single_bottleneck_multiple_breaks
e31_no_clear_breakpoint_detected
e31_artifact_invalid
```

## Boundary

E31 is a controlled Flow/Pocket bottleneck localization probe. It is not a chatbot, deployed model, raw language reasoning proof, AGI claim, consciousness claim, or model-scale claim.

# E32 Flow Field Capacity And Trace Ledger Repair Confirm Contract

Status: implemented probe contract.

## Purpose

E31 localized a combined bottleneck:

```text
Flow Field state bandwidth
Trace Ledger exactness
Ingress Codec event formation
evidence-span binding on decoy/long-context rungs
```

E32 tests whether the strongest E31 repair signal, larger Flow Field capacity, can be matched or improved by more targeted objectives.

## Systems

```text
baseline_d96_p8
capacity_flow_d192_p8
trace_ledger_weighted_d96_p8
trace_ledger_weighted_d192_p8
span_bucket_aux_d96_p8
ingress_event_aux_d96_p8
combined_capacity_aux_d192_p8
random_static_control
```

## Metrics

Primary metrics:

```text
heldout resolution_success
heldout action_accuracy
heldout trace_exact
heldout trace_bit_accuracy
wrong_confident_answer_on_unresolved
false_ask_on_answerable
per-rung heldout metrics
```

Repair deltas are measured against `baseline_d96_p8`.

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
repair_plan.json
repair_comparison_report.json
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

## Decisions

```text
e32_capacity_only_repair_confirmed
e32_trace_ledger_auxiliary_positive
e32_span_auxiliary_positive
e32_ingress_auxiliary_positive
e32_combined_capacity_auxiliary_positive
e32_no_repair_confirmed
e32_artifact_invalid
```

## Boundary

E32 is a controlled Flow/Pocket repair probe. It does not test raw language reasoning, AGI, consciousness, deployment quality, or model-scale behavior.

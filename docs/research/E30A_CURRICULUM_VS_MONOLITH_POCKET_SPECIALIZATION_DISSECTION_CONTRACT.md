# E30A Curriculum vs Monolith Pocket Specialization Dissection Contract

## Purpose

E30A compares training paths on the same Flow/Pocket architecture:

```text
monolith_direct_final
curriculum_staged_final
random_order_curriculum_control
reverse_curriculum_control
random_static_control
```

The goal is not simply to beat a baseline. The goal is to inspect whether staged curriculum produces more specialized Pocket Operators, cleaner Arbiter behavior, stronger Trace Ledger validity, and more local ablation effects than direct final-task training.

## Canonical Naming

E30A uses the canonical Flow/Pocket naming scheme:

```text
Ground Field
Flow Field
Pocket Operator
Lens Pocket
Writer Pocket
Arbiter
Trace Ledger
Ingress Codec
Egress Codec
```

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
row_level_results.jsonl
trace_ledger.jsonl
arbiter_decision_trace.jsonl
pocket_activation_map.json
pocket_ablation_table.json
field_writeback_map.json
flow_field_snapshot.json
ground_field_snapshot.json
conflict_map.json
unresolved_state_map.json
training_curve_report.json
system_results.json
aggregate_metrics.json
deterministic_replay.json
resource_usage_report.json
decision.json
summary.json
report.md
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
```

## Metrics

- heldout resolution success
- trap resolution success
- phrase-holdout resolution success
- action accuracy
- trace exactness
- wrong confident answer on unresolved rows
- false ask on answerable rows
- Pocket Operator specialization score
- ablation locality score
- Arbiter activation entropy
- deterministic replay hash match

## Decision Labels

```text
e30a_curriculum_specialization_positive
e30a_curriculum_accuracy_only_no_specialization
e30a_monolith_sufficient
e30a_no_clear_specialization_signal
e30a_artifact_invalid
```

## Boundary

E30A is a controlled naturalized-text dissection probe. It is not a chatbot, production system, raw language reasoning proof, AGI claim, consciousness claim, or model-scale claim.

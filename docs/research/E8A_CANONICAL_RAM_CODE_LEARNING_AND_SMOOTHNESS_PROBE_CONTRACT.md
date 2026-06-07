# E8A Canonical RAM Code Learning And Smoothness Probe Contract

## Purpose

`E8A_CANONICAL_RAM_CODE_LEARNING_AND_SMOOTHNESS_PROBE` follows E7Z.

E7Z showed:

```text
oracle low-bit RAM code is expressive
learned low-bit systems do not discover that code from final-answer pressure alone
```

E8A tests whether intermediate RAM-code supervision, staged
producer/consumer distillation, smooth-to-hard curriculum, simplified teacher
codes, and mutation repair after distillation can make the canonical RAM
language learnable.

## Boundary

This is a controlled numeric pocket-router proxy.

```text
No semantic labels.
No dense graph as primary.
No new router architecture.
No image/MNIST/language task migration.
No oracle write at inference for learned systems.
No AGI, consciousness, deployed-model, or model-scale claims.
```

Oracle writes may be used only as teacher targets or diagnostic references.

## Systems

```text
current_best_baseline
oracle_low_bit_reference
producer_distill_binary
producer_distill_ternary
producer_distill_int4
consumer_distill_binary
producer_consumer_staged_binary
producer_consumer_staged_ternary
producer_consumer_staged_int4
soft_to_hard_int4_to_ternary_to_binary
contrastive_ram_code_alignment
progressive_code_freeze
mutation_only_from_random_lowbit
mutation_repair_after_distillation
full_end_to_end_control
dense_graph_danger_control
```

## Teacher Codes

```text
current_oracle_projection_code
simplified_canonical_code
```

The simplified teacher exists to test whether the E7Z oracle projection is too
jagged or arbitrary for a pocket system to learn.

## Required Metrics

```text
composition usefulness
heldout/OOD/counterfactual/adversarial usefulness
answer accuracy
route accuracy
producer code accuracy / oracle-code similarity
per-cell bundle MAE
support sign mismatch rate
next-pocket compatibility error
consumer read accuracy with oracle-coded input
consumer read accuracy with learned code
1-bit flip fitness-drop proxy
2-bit flip fitness-drop proxy
local neighborhood valid rate
capture basin radius proxy
progressive freeze rounds
mutation repair gain
accepted/rejected/rollback mutation counts
dense graph control comparison
deterministic replay
checker failure_count
```

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
producer_distillation_report.json
consumer_distillation_report.json
staged_composition_report.json
smoothness_report.json
mutation_repair_report.json
code_teacher_comparison_report.json
system_results.json
row_level_samples.json
aggregate_metrics.json
decision.json
summary.json
report.md
deterministic_replay.json
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
```

## Decision Labels

```text
e8a_canonical_ram_code_distillation_positive
e8a_consumer_read_bottleneck
e8a_producer_write_bottleneck
e8a_soft_to_hard_code_curriculum_required
e8a_int4_code_required
e8a_binary_canonical_code_learned
e8a_mutation_repair_after_distillation_positive
e8a_mutation_only_code_learning_viable
e8a_current_oracle_code_too_jagged
e8a_canonical_ram_code_learning_failed
e8a_graph_soup_regression_detected
```

## Guardrails

```text
real row-level eval
no semantic labels as model inputs
oracle used only as teacher/diagnostic
learned systems evaluated without oracle writes
mutation repair uses no backprop
mutation-only control included
deterministic replay required
checker failure_count must be 0
```

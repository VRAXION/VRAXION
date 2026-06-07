# E8B Producer Write-Code Dynamics And Loss Audit Contract

## Purpose

E8A showed producer-side canonical RAM-code writing is the main bottleneck.
E8B diagnoses why the producer plateaus below oracle quality.

This is a controlled numeric producer-write audit, not a new router architecture.

## Systems

```text
current_projection_teacher_baseline
simplified_canonical_teacher
factorized_teacher_code
codebook_teacher
local_smooth_teacher_code
per_cell_supervised_loss
weighted_per_cell_loss
contrastive_plus_per_cell_loss
consumer_aware_loss
code_similarity_loss
soft_continuous_to_int4
int4_to_ternary_to_binary
progressive_cell_freeze_after_similarity_threshold
mutation_repair_after_similarity_threshold
mutation_only_from_random_lowbit
dense_graph_danger_control
consumer_distill_reference
oracle_low_bit_reference
```

## Required Diagnostics

Every trained producer system must log, per seed/skill/epoch:

```text
train_loss
validation_loss
train_code_similarity
validation_code_similarity
ood_code_similarity
train_bundle_mae
validation_bundle_mae
ood_bundle_mae
per_cell_mae
per_cell_sign_accuracy
validation_support_sign_mismatch
validation_support_silence_rate
validation_write_entropy
gradient_norm
gradient_variance
gradient_cosine
gradient_cosine_negative_rate
```

Every trained producer must also report:

```text
loss_start
loss_end
loss_drop
validation_code_similarity_start
validation_code_similarity_end
validation_code_similarity_best
best_epoch
final_gap_to_best
tail_gain
tail_range
```

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
producer_dynamics_report.json
gradient_diagnostics_report.json
teacher_comparison_report.json
loss_variant_report.json
curriculum_report.json
mutation_repair_report.json
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
e8b_current_projection_teacher_too_hard
e8b_factorized_or_codebook_teacher_positive
e8b_per_cell_supervision_positive
e8b_producer_generalization_bottleneck
e8b_producer_capacity_or_loss_bottleneck
e8b_gradient_conflict_or_jagged_target_confirmed
e8b_teacher_or_architecture_mismatch
e8b_seeded_mutation_repair_positive
e8b_mutation_only_producer_learning_viable
e8b_graph_soup_regression_detected
```

## Guardrails

```text
no semantic labels as model input
no new router architecture
no oracle write at inference for learned systems
oracle allowed only as teacher target / diagnostic reference
mutation repair must not use backprop
deterministic replay must pass
checker failure_count must be 0
```

## Boundary

E8B only tests producer-side RAM-code learning dynamics in a controlled numeric
pocket-router proxy. It makes no raw-language, AGI, consciousness,
deployed-model, or model-scale claim.

# E8C Producer Target Decomposition And Consumer Compatibility Probe Contract

## Purpose

E8B showed producer RAM-code learning improves smoothly, then plateaus below
useful downstream composition, and smaller-batch diagnostics expose gradient
direction conflict.

E8C tests whether decomposing the mechanical producer target into smaller
subtargets reduces that conflict and improves downstream consumer compatibility.

## Systems

```text
current_full_code_teacher_baseline
local_smooth_full_code_teacher
per_skill_decomposed_heads
primary_then_support_staged_teacher
support_cells_only_after_primary_plateau
consumer_sensitivity_weighted_targets
route_step_local_teacher_targets
codebook_decomposed_targets
low_conflict_batch_curriculum
consumer_compatibility_weighted_loss
mutation_repair_after_consumer_compatible_plateau
mutation_only_decomposed_lowbit
dense_graph_danger_control
consumer_distill_reference
oracle_low_bit_reference
```

## Mechanical Target Groups

```text
primary cell = first bundle cell
support cells = remaining anonymous bundle cells
consumer-sensitivity cells = bundle cells ranked by frozen next-read impact
route-step-local cells = bundle cells read by the next pocket in that route step
```

These are mechanical groups only. No semantic lane labels are model inputs.

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
target_decomposition_report.json
consumer_sensitivity_report.json
producer_dynamics_report.json
gradient_diagnostics_report.json
compatibility_report.json
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

## Required Metrics

```text
composition usefulness
answer accuracy
route accuracy
heldout/OOD/counterfactual/adversarial usefulness
oracle code similarity
consumer compatibility score
next-pocket compatibility error
per-cell MAE and sign accuracy
gradient norm
gradient variance
gradient cosine
gradient negative-rate
tail_gain
tail_range
mutation accepted/rejected/rollback counts
parameter diff/hash
deterministic replay hash match
```

Gradient diagnostics must use a low-batch diagnostic path so cosine/variance are
not silently degenerate.

## Decision Labels

```text
e8c_target_decomposition_positive
e8c_consumer_sensitivity_weighting_positive
e8c_route_step_local_targets_positive
e8c_gradient_conflict_reduced_but_usefulness_low
e8c_producer_architecture_bottleneck
e8c_consumer_interface_bottleneck
e8c_mutation_repair_after_compatibility_plateau_positive
e8c_mutation_only_decomposed_learning_viable
e8c_current_code_interface_still_wrong
e8c_graph_soup_regression_detected
```

## Guardrails

```text
no new router architecture
no semantic labels as model input
no oracle write at inference for learned systems
oracle allowed only as teacher target / diagnostic reference
mutation repair must not use backprop
dense graph remains danger control only
checker failure_count must be 0
deterministic replay must pass
```

## Boundary

E8C is a controlled symbolic/numeric producer-write probe. It makes no raw
language, image, AGI, consciousness, deployed-model, or model-scale claim.

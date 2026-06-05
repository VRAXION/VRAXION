# E7A9 Binary Freeze Policy Upper-Bound Audit Contract

## Purpose

E7A9 decides whether binary matrix-core is worth pursuing as a quality path, or whether it should remain a compression side branch behind int4/ternary.

The probe uses the existing E7 matrix-core only. It does not add architecture or operator-cell complexity.

## Systems

Required methods:

```text
float32_matrix_core
int8_direct
int4_direct
ternary_qat_reference
binary_direct
binary_qat_baseline
binary_qat_best_effort
binary_distance_paramwise_freeze
binary_sensitivity_paramwise_freeze
binary_qat_warmstart_paramwise_freeze
binary_direct_mutation_repair
mixed_input_int4_state_binary_output_int4
mixed_input_ternary_state_binary_output_int4
```

## Highest-Quality Binary QAT Reference

The best-effort binary QAT reference must include:

```text
float warm-start
gradual binary ramp
STE fake quantization
teacher distillation from float matrix-core
per-channel scale at conversion
best validation checkpoint
multiple deterministic splits
```

## Paramwise Freeze

Paramwise freeze methods freeze one binary parameter at a time according to:

```text
distance-only order
sensitivity-aware order
QAT-warmstart distance order
```

Each frozen parameter is followed by mutation/rollback repair. Rollback is required if heldout or OOD/counterfactual/adversarial accuracy drops beyond guard thresholds.

## Mixed Precision

Mixed precision candidates:

```text
input_projection int4, state binary, carry/state_bias/output int4
input_projection ternary, state binary, carry/state_bias/output int4
```

## Metrics

Required metrics:

```text
heldout/OOD/counterfactual/adversarial accuracy
eval average
gap to float
gap to int4
gap to ternary
effective bit cost
compression vs float32
freeze rounds
rollback rounds
frozen parameter ratio
input_projection frozen ratio
mutation accepted/rejected/rollback counts
deterministic replay
```

## Decision Rules

Allowed decisions:

```text
e7a9_binary_quality_competitive
e7a9_binary_not_quality_competitive
e7a9_mixed_precision_matrix_core_preferred
e7a9_binary_paramwise_freeze_positive
e7a9_qat_upper_bound_remains_preferred
e7a9_invalid_artifact_detected
```

Binary is quality competitive if:

```text
binary_best >= ternary_best - 0.015
OR binary_best >= int4_best - 0.025
AND no heldout/OOD/counterfactual/adversarial collapse
```

If mixed precision beats the best binary method by at least 0.005 and passes generalization gates, prefer mixed precision.

## Required Artifacts

```text
e7a9_backend_manifest.json
e7a9_task_generation_report.json
e7a9_method_comparison_report.json
e7a9_binary_freeze_schedule_report.json
e7a9_qat_upper_bound_report.json
e7a9_precision_tradeoff_report.json
e7a9_mixed_precision_report.json
e7a9_mutation_history.json
e7a9_no_synthetic_metric_audit.json
e7a9_runtime_report.json
e7a9_deterministic_replay_report.json
aggregate_metrics.json
decision.json
summary.json
report.md
progress.jsonl
row-level samples for heldout/OOD/counterfactual/adversarial
```

## Checker Requirements

The checker fails on:

```text
missing method or artifact
missing row-level eval samples
missing bit-cost estimates
missing accepted/rejected/rollback counts
rollback mismatch
mutation-only path using optimizer/backprop
deterministic replay mismatch
hardcoded improvement flags
new architecture flag
forbidden broad claims
failure_count != 0
```

## Boundary

E7A9 is only a controlled low-bit matrix-core quality/compression tradeoff audit.

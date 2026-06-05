# E7A8 Progressive Quant Freeze Plateau Repair Contract

## Purpose

E7A8 tests whether progressive quantize/freeze/plateau repair can close the gap between post-quantization mutation repair and QAT for the existing E7 matrix-core.

This is only a low-bit repair strategy test. It does not add architecture, operator-cell complexity, or broad model claims.

## Context

E7A7 found:

```text
decision = e7a7_qat_preferred_over_post_repair
main low-bit damage = input_projection
QAT > post-quant mutation repair
```

E7A8 asks:

```text
Can progressive freeze scheduling and input-projection-aware repair improve mutation/rollback low-bit recovery?
```

## Methods

Required methods:

```text
baseline_float_matrix_core
direct_low_bit_quant
post_quant_mutation_repair
distance_only_progressive_freeze
sensitivity_aware_progressive_freeze
input_projection_aware_progressive_freeze
blockwise_scale_mutation_repair
qat_reference
```

Target modes:

```text
ternary
binary
```

Optional target:

```text
int3
```

## Progressive Freeze Semantics

All progressive methods operate on the target low-bit candidate. The freeze schedule determines which low-bit parameters are locked and which remain mutable during plateau repair.

Distance-only:

```text
score = normalized quantization distance
freeze easiest parameters first
```

Sensitivity-aware:

```text
score = quantization distance + validation sensitivity
freeze low-distance and low-sensitivity parameters first
```

Input-projection-aware:

```text
same as sensitivity-aware
but input_projection freezes more slowly
and receives input-biased mutation attention
```

Plateau:

```text
validation improvement < 0.001 for patience 8 generations
```

Rollback:

```text
rollback freeze round if heldout drops > 0.01
or OOD/counterfactual/adversarial drops > 0.015
```

## Metrics

Required metrics:

```text
heldout accuracy
OOD accuracy
counterfactual accuracy
adversarial accuracy
eval average
quantization drop from float
recovery delta vs direct quant
gap to QAT
freeze rounds count
frozen parameter ratio per round
input_projection frozen ratio
recurrent_state frozen ratio
output_head frozen ratio
rollback count
accepted/rejected mutation count
plateau generations per round
runtime
deterministic replay
```

## Decisions

Allowed decisions:

```text
e7a8_input_projection_aware_progressive_freeze_positive
e7a8_sensitivity_aware_freeze_positive
e7a8_distance_only_freeze_sufficient
e7a8_blockwise_scale_repair_positive
e7a8_progressive_freeze_no_advantage
e7a8_progressive_freeze_overfit_or_brittle
e7a8_invalid_artifact_detected
```

Decision rules:

```text
input_projection_aware_positive:
  beats post-quant repair
  and closes >= 50% of the gap to QAT

sensitivity_aware_positive:
  beats distance-only and post-quant repair

distance_only_sufficient:
  distance-only is within 0.005 of sensitivity-aware
  and beats post-quant repair for at least one target level

blockwise_scale_positive:
  blockwise scale mutation beats post-quant repair by > 0.01

no_advantage:
  no progressive/scale method beats post-quant repair

overfit_or_brittle:
  freeze schedule causes repeated freeze-round rollbacks
```

## Required Artifacts

```text
e7a8_backend_manifest.json
e7a8_task_generation_report.json
e7a8_method_comparison_report.json
e7a8_freeze_schedule_report.json
e7a8_input_projection_damage_recovery_report.json
e7a8_mutation_repair_report.json
e7a8_mutation_history.json
e7a8_no_synthetic_metric_audit.json
e7a8_runtime_report.json
e7a8_deterministic_replay_report.json
aggregate_metrics.json
decision.json
summary.json
report.md
progress.jsonl
row-level eval samples for heldout/OOD/counterfactual/adversarial
```

## Checker Requirements

The checker fails on:

```text
missing artifact
missing method or target level
missing row-level samples
missing accepted/rejected/rollback counts
rollback mismatch
mutation-only repair using optimizer/backprop
deterministic replay mismatch
hardcoded improvement flags
new architecture flag
forbidden broad claims
failure_count != 0
```

## Boundary

E7A8 is only a low-bit matrix-core repair strategy test. It does not prove broad reasoning behavior.

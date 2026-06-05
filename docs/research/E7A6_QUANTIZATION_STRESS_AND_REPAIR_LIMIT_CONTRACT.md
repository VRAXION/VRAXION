# E7A6 Quantization Stress And Repair Limit Contract

## Purpose

E7A6 maps how far the E7A4/E7A5 plain matrix-core can be quantized before it breaks, and whether mutation-only repair becomes useful after quantization damage.

This is a controlled symbolic/numeric proxy. It does not make final E7 claims.

## Systems

```text
float32_matrix_core
quantized_no_repair
quantized_mutation_repair
random_control
```

## Quantization Levels

```text
int8    symmetric integer q in [-127, 127]
int4    symmetric integer q in [-7, 7]
int3    symmetric integer q in [-3, 3]
ternary q in {-1, 0, 1}
binary  q in {-1, +1}, with zero-only tensors allowed to remain zero
```

Every level is evaluated before repair and after mutation-only repair.

## Task

The probe reuses the E7A3/E7A4 deterministic symbolic/numeric task:

- train
- validation
- heldout
- OOD
- counterfactual
- adversarial

All metrics must come from row-level evaluation.

## Metrics

Primary metrics:

- float32 eval accuracy
- quantized eval accuracy
- repair eval accuracy
- quantization drop vs float32
- repair drop vs float32
- repair delta vs quantized
- heldout/OOD/counterfactual/adversarial accuracy
- parameter count and nonzero parameter count
- accepted/rejected mutation counts
- rollback count
- parameter diff/hash
- deterministic replay hash match

Solve thresholds:

```text
heldout >= 0.90
OOD >= 0.85
counterfactual >= 0.85
adversarial >= 0.80
```

A quantized level is stable without repair only if it passes solve thresholds and its eval drop from float32 is at most `0.02`.

Mutation repair is positive only if it recovers a previously unstable level or improves eval accuracy by at least `0.02`.

## Allowed Decisions

```text
e7a6_int8_only_stable
e7a6_int4_stable_without_repair
e7a6_int3_or_lower_stable_without_repair
e7a6_mutation_repair_recovers_low_bit_core
e7a6_mutation_repair_partial_low_bit_recovery
e7a6_mutation_repair_not_useful
e7a6_quantization_breakpoint_mapped
e7a6_invalid_artifact_detected
```

## Required Artifacts

Artifact root:

```text
target/pilot_wave/e7a6_quantization_stress_and_repair_limit/
```

Required top-level artifacts:

- `e7a6_backend_manifest.json`
- `e7a6_task_generation_report.json`
- `e7a6_float_training_report.json`
- `e7a6_quantization_stress_report.json`
- `e7a6_mutation_repair_report.json`
- `e7a6_frontier_report.json`
- `e7a6_mutation_history.json`
- `e7a6_no_synthetic_metric_audit.json`
- `e7a6_deterministic_replay_report.json`
- `aggregate_metrics.json`
- `decision.json`
- `summary.json`
- `report.md`
- `progress.jsonl`
- row-level samples for heldout/OOD/counterfactual/adversarial

Per width:

- float32 candidate summary
- float32 state summary
- training history
- random control summary

Per quantization level and width:

- quantized candidate
- no-repair summary
- mutation history
- mutation repair initial/final candidates
- mutation repair summary
- parameter diff

## Checker Requirements

The checker fails on:

- missing artifact
- missing quantization level
- missing row-level samples
- no accepted/rejected mutations for repair variants
- rollback mismatch
- missing parameter diff/hash
- deterministic replay mismatch
- replay progress missing
- mutation repair using optimizer/backprop
- random control passing
- hardcoded improvement flags
- final E7 or model-scale claims

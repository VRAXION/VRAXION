# E7A4 Quantized Bridge From Backprop To Mutation Matrix Contract

## Purpose

E7A4 tests the bridge suggested by E7A3:

```text
float MLP reference
float matrix-core backprop
quantized matrix-core without repair
quantized matrix-core with mutation repair
```

The goal is to separate three questions:

```text
1. Is the matrix-core architecture learnable with gradients?
2. Does the learned matrix-core survive integer quantization?
3. Can mutation + rollback repair a quantized learned matrix-core?
```

This is a controlled toy substrate test. It does not claim a final E7 architecture, natural-language reasoning, consciousness, AGI, or model-scale behavior.

## Systems

- `float_mlp_backprop_reference`: ordinary E7A3 float MLP reference.
- `float_matrix_core_backprop`: differentiable version of the E7A3 matrix hidden replacement.
- `quantized_matrix_core_no_repair`: learned matrix-core quantized to symmetric integer tensors with fixed per-tensor scales.
- `quantized_matrix_core_mutation_repair`: quantized matrix-core after integer mutation + rollback repair.
- `random_control`: task-ease/leakage control.

## Task And Metrics

The runner reuses the E7A3 margin-filtered toy task.

Required metrics:

- train / validation / heldout / OOD / counterfactual / adversarial accuracy
- generalization gap
- parameter count
- matrix shape and matrix cell count
- smallest passing width
- best width by eval accuracy
- quantization delta
- mutation repair delta
- accepted/rejected mutation counts
- rollback count
- deterministic replay

Solve threshold:

```text
heldout >= 0.90
OOD >= 0.85
counterfactual >= 0.85
adversarial >= 0.80
```

## Required Artifacts

Run root:

```text
target/pilot_wave/e7a4_quantized_bridge_from_backprop_to_mutation_matrix/
```

Required reports:

- `e7a4_backend_manifest.json`
- `e7a4_task_generation_report.json`
- `e7a4_bridge_report.json`
- `e7a4_quantization_report.json`
- `e7a4_mutation_repair_report.json`
- `e7a4_training_history.json`
- `e7a4_mutation_history.json`
- `e7a4_no_synthetic_metric_audit.json`
- `e7a4_deterministic_replay_report.json`
- `aggregate_metrics.json`
- `decision.json`
- `summary.json`
- `report.md`

Long-ish runs must write progress during training and repair.

## Allowed Decisions

- `e7a4_reference_not_solved_redesign_required`
- `e7a4_float_matrix_core_not_learned`
- `e7a4_quantized_matrix_core_preserved_without_repair`
- `e7a4_mutation_repair_improves_quantized_matrix_core`
- `e7a4_mutation_repair_recovers_quantized_matrix_core`
- `e7a4_quantization_breaks_and_mutation_repair_failed`
- `e7a4_task_too_easy_or_leaky`

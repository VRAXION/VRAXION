# E7A5 Operator Cell Matrix Incremental Scan Contract

## Purpose

E7A4 showed that a plain matrix-core can be learned by backprop and can survive post-training quantization. E7A5 tests the next narrower question:

```text
Does a matrix cell that contains a small local operator add real value over a plain matrix-core baseline?
```

This is a controlled symbolic/numeric toy proxy. It is not a claim about AGI, consciousness, natural-language reasoning, or model-scale behavior.

## Variants

The scan is incremental. Each step adds one main degree of freedom.

```text
plain_matrix_core_baseline
  h_next = tanh(h @ W + drive)

soft_mask_matrix
  h_next = tanh(h @ (W * sigmoid(C)) + drive)

edge_bias_shared_activation
  edge_ij = sigmoid(C_ij) * tanh(W_ij * h_j + B_ij)

per_cell_activation_soft_mixture
  edge_ij = sigmoid(C_ij) * sum_k softmax(A_ij,k) * psi_k(W_ij * h_j + B_ij)

source_target_trace_operand_cell
  edge_ij = sigmoid(C_ij) * sum_k softmax(A_ij,k) * psi_k(a_ij*h_j + b_ij*h_i + c_ij*trace_i + d_ij)

random_control
  row-level random logits sanity control
```

## Training And Repair Modes

Each differentiable variant runs:

- float/backprop training
- post-training symmetric integer quantization
- mutation-only repair after quantization

Mutation repair must not call backprop, optimizers, or gradient steps. It must report accepted mutations, rejected mutations, rollback count, mutation attempts, and parameter diff/hash.

## Task

E7A5 reuses the E7A3/E7A4 deterministic toy substrate task and adds a stress split with small numeric perturbation.

Required splits:

- train
- validation
- heldout
- OOD
- counterfactual
- adversarial
- stress

All metrics must come from row-level evaluation.

## Metrics

Primary metrics:

- heldout accuracy
- OOD accuracy
- counterfactual accuracy
- adversarial accuracy
- stress accuracy
- mean eval accuracy across non-train evaluation splits
- generalization gap
- parameter count
- parameter ratio versus plain baseline
- parameter-normalized delta versus plain baseline
- quantization delta
- mutation repair delta
- accepted/rejected/rollback mutation counts
- deterministic replay hash match

Solve thresholds:

```text
heldout >= 0.90
OOD >= 0.85
counterfactual >= 0.85
adversarial >= 0.80
stress >= 0.85
```

## Positive Evidence Rules

Extra parameters do not count as success by themselves. An operator-cell variant is positive only if it beats the plain matrix-core baseline by a meaningful margin:

- `+0.02` absolute eval accuracy, and parameter-normalized delta is positive, or
- same accuracy with materially fewer effective parameters, or
- better quantization survival than baseline, or
- mutation repair succeeds where the plain core fails.

## Allowed Decisions

```text
e7a5_edge_bias_shared_activation_positive
e7a5_per_cell_activation_positive
e7a5_operand_cell_positive
e7a5_operator_cell_no_advantage_detected
e7a5_operator_cell_overfit_or_search_noise
e7a5_mutation_repair_value_confirmed
e7a5_invalid_artifact_detected
```

## Required Artifacts

Artifact root:

```text
target/pilot_wave/e7a5_operator_cell_matrix_incremental_scan/
```

Required top-level artifacts:

- `e7a5_backend_manifest.json`
- `e7a5_task_generation_report.json`
- `e7a5_incremental_comparison_report.json`
- `e7a5_quantization_report.json`
- `e7a5_mutation_repair_report.json`
- `e7a5_operator_cell_audit.json`
- `e7a5_training_history.json`
- `e7a5_mutation_history.json`
- `e7a5_no_synthetic_metric_audit.json`
- `e7a5_deterministic_replay_report.json`
- `aggregate_metrics.json`
- `decision.json`
- `summary.json`
- `report.md`
- `progress.jsonl`
- row-level samples for heldout/OOD/counterfactual/adversarial/stress

Per width and variant:

- float candidate summary
- float state summary
- training history
- quantized candidate and summary
- mutation history
- mutation repair initial/final candidates
- mutation repair parameter diff

## Checker Requirements

The checker fails on:

- missing artifact
- missing required variant
- missing row-level samples
- missing accepted/rejected mutation counts
- rollback mismatch
- missing parameter diff/hash
- deterministic replay mismatch
- mutation-only repair using optimizer/backprop
- hardcoded improvement flags
- random control passing
- static synthetic metrics
- final E7, AGI, consciousness, natural-language reasoning, or model-scale claims

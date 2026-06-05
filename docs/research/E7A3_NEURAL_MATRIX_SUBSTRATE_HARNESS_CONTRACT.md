# E7A3 Neural Matrix Substrate Harness Contract

## Purpose

E7A3 tests the minimal viable route from standard neural function toward a matrix-medium replacement on toy tasks.

It compares:

- `float_mlp_backprop`: ordinary float MLP, trained with backprop.
- `integer_mlp_mutation`: ordinary MLP topology, integer weights, mutation + rollback only.
- `integer_matrix_hidden_replacement_mutation`: input/output shell with hidden layers replaced by a recurrent integer matrix core, mutation + rollback only.
- `random_control`: leakage/task-ease control.

The question is size and substrate viability. The default task is deliberately linearly dominant with a small nonlinear perturbation and a target-margin filter, so the float MLP reference should solve before mutation paths are interpreted:

```text
How large does the hidden matrix/state need to be before the task solves?
Does integer mutation solve the same toy task?
Does matrix hidden replacement solve or match the integer MLP mutation path?
```

## Sweep

Default hidden/state widths:

```text
4, 8, 16, 32
```

For neural systems, width is hidden width. For matrix hidden replacement, width is state dimension and the main hidden matrix is `width x width`.

## Required Metrics

Required metrics:

- train / validation / heldout / OOD / counterfactual / adversarial accuracy
- generalization gap
- parameter count
- matrix shape and matrix cell count
- smallest passing width
- best width by eval accuracy
- accepted/rejected mutations and rollback count
- parameter diff for mutation systems
- deterministic replay

Solve threshold:

```text
heldout >= 0.90
OOD >= 0.85
counterfactual >= 0.85
adversarial >= 0.80
```

## Constraints

- Backprop is allowed only for `float_mlp_backprop`.
- Integer mutation systems must not call optimizer/backprop.
- Every long run must write progress and partial artifacts during the run.
- CPU mutation jobs should overlap the GPU gradient lane when parallel mode is used.
- Final E7 verdict remains intentionally deferred.

## Required Artifacts

Run root:

```text
target/pilot_wave/e7a3_neural_matrix_substrate_harness/
```

Required reports:

- `e7a3_backend_manifest.json`
- `e7a3_task_generation_report.json`
- `e7a3_size_sweep_report.json`
- `e7a3_substrate_comparison_report.json`
- `e7a3_matrix_size_report.json`
- `e7a3_mutation_history.json`
- `e7a3_training_history.json`
- `e7a3_no_synthetic_metric_audit.json`
- `e7a3_deterministic_replay_report.json`
- `aggregate_metrics.json`
- `decision.json`
- `summary.json`
- `report.md`

## Allowed Decisions

- `e7a3_backprop_reference_solved_only`
- `e7a3_integer_mutation_network_viable`
- `e7a3_matrix_hidden_replacement_viable`
- `e7a3_matrix_hidden_replacement_matches_integer_network`
- `e7a3_no_mutation_path_detected`
- `e7a3_reference_not_solved_redesign_required`
- `e7a3_task_too_easy_or_leaky`

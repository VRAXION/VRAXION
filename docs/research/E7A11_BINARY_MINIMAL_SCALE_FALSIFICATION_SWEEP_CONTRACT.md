# E7A11 Binary Minimal-Scale Falsification Sweep Contract

## Purpose

E7A11 tries to falsify the strongest E7A10 claim: minimal-scale binary QAT at larger width can beat an int4 matrix-core reference under equal or lower measured bit budget.

## Claim Under Test

```text
binary_minimal_scale_qat width scaling is not just an E7A10 seed/task artifact.
```

## Sweep Axes

- multiple seed groups
- baseline and shifted symbolic task families
- input dimension changes
- class-count changes
- int4 references at width32/48/64
- binary minimal-scale QAT at width32/64/96/128
- ternary and direct-binary controls
- heldout/OOD/counterfactual/adversarial splits

## Required Systems

- `float32_matrix_core`
- `int4_direct`
- `int3_direct`
- `ternary_block_scale_qat`
- `binary_direct_block_scale`
- `binary_minimal_scale_qat`

## Required Artifacts

- `e7a11_backend_manifest.json`
- `e7a11_task_family_report.json`
- `e7a11_case_results.json`
- `e7a11_bit_budget_falsification_report.json`
- `e7a11_width_scaling_report.json`
- `e7a11_no_synthetic_metric_audit.json`
- `e7a11_runtime_report.json`
- `e7a11_deterministic_replay_report.json`
- `aggregate_metrics.json`
- `decision.json`
- `summary.json`
- `report.md`
- `progress.jsonl`

## Decision Rules

- `e7a11_binary_minimal_survives_falsification`: at least 70% of cases are positive and median reference-width32 margin is at least 0.005.
- `e7a11_binary_minimal_partially_survives`: mixed evidence, but at least half the cases remain positive.
- `e7a11_binary_minimal_seed_or_task_artifact_detected`: at least half the cases falsify the E7A10-style same-budget binary win.
- `e7a11_int4_restored_preference`: int4 wins more cases than binary.
- `e7a11_task_family_redesign_required`: the sweep is inconclusive or too unstable to interpret.
- `e7a11_invalid_artifact_detected`: checker, replay, row-level, or bit-budget gate fails.

## Boundary

This is a controlled symbolic/numeric matrix-core falsification sweep. It does not claim anything about natural-language reasoning, AGI, consciousness, or model-scale behavior.

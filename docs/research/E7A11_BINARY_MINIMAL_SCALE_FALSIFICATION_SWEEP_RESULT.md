# E7A11 Binary Minimal-Scale Falsification Sweep Result

## Decision

```text
decision = e7a11_binary_minimal_seed_or_task_artifact_detected
deterministic_replay_passed = true
checker_failure_count = 0
```

Artifact root:

```text
target/pilot_wave/e7a11_binary_minimal_scale_falsification_sweep/
```

## Summary

E7A11 tried to break the strongest E7A10 result: `binary_minimal_scale_qat width64` beating the int4 width32 reference under equal or lower bit budget.

The falsification sweep changed seed groups, task families, input dimensions, and class count while keeping row-level heldout/OOD/counterfactual/adversarial evaluation.

```text
positive_cases = 2 / 6
falsified_cases = 3 / 6
median_reference32_margin = -0.009375
mean_reference32_margin = -0.009375
```

## Case Results

| case | family | dim | classes | int4 width32 eval | best same-budget binary | binary eval | margin |
|---|---|---:|---:|---:|---|---:|---:|
| `baseline_seed_a` | baseline | 10 | 4 | 0.922917 | `binary_minimal_scale_qat width64` | 0.927083 | +0.004167 |
| `baseline_seed_b` | baseline | 10 | 4 | 0.920833 | `binary_minimal_scale_qat width64` | 0.922917 | +0.002083 |
| `five_class_12x5` | five_class | 12 | 5 | 0.900000 | `binary_minimal_scale_qat width64` | 0.885417 | -0.014583 |
| `high_dim_16x4` | high_dim | 16 | 4 | 0.958333 | `binary_minimal_scale_qat width64` | 0.937500 | -0.020833 |
| `interaction_10x4` | interaction | 10 | 4 | 0.987500 | `binary_minimal_scale_qat width64` | 0.964583 | -0.022917 |
| `wide_12x4` | wide_mix | 12 | 4 | 0.972917 | `binary_minimal_scale_qat width64` | 0.968750 | -0.004167 |

## Interpretation

E7A10 was not garbage: the baseline task family still gave two positive same-budget binary cases. But the win did not survive the falsification sweep. Once the task family changed, int4 restored an advantage in the interaction, high-dimensional, and five-class cases.

The best current reading is:

```text
minimal-scale binary width scaling is viable on the original proxy family,
but it is not robust enough to become the main path yet.
```

This demotes the E7A10 result from "binary same-budget preferred" to "promising but task-family fragile."

## Recommendation

Do not discard binary entirely. Keep three separate paths:

```text
practical path: int4 matrix-core
research path: binary with better QAT/scale policy
diagnostic path: identify which task properties break minimal-scale binary
```

The next useful experiment is a binary QAT failure audit: compare minimal-scale, global-scale, block-scale, and channel-scale binary on the E7A11 failing families only. The question is whether the failure is caused by fixed minimal scale, insufficient width, binary signs, or QAT objective weakness.

## Boundary

This is a controlled symbolic/numeric matrix-core falsification sweep. It does not claim anything about natural-language reasoning, AGI, consciousness, or model-scale behavior.

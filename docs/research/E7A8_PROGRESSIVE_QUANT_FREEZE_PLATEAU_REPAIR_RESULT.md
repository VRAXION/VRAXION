# E7A8 Progressive Quant Freeze Plateau Repair Result

## Decision

```text
decision = e7a8_sensitivity_aware_freeze_positive
deterministic_replay_passed = true
checker_failure_count = 0
```

Run root:

```text
target/pilot_wave/e7a8_progressive_quant_freeze_plateau_repair
```

## Methods Ran

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

Target levels:

```text
ternary
binary
```

Best float32 matrix-core:

```text
width = 32
eval_accuracy = 0.958333
```

## Method Comparison

| level | method | width | eval | drop from float | recovery vs direct | gap to QAT |
|---|---|---:|---:|---:|---:|---:|
| `ternary` | `direct_low_bit_quant` | 32 | 0.915000 | 0.043333 | 0.000000 | |
| `ternary` | `post_quant_mutation_repair` | 32 | 0.928333 | 0.030000 | 0.013333 | 0.028333 |
| `ternary` | `distance_only_progressive_freeze` | 32 | 0.925833 | 0.032500 | 0.010833 | 0.030833 |
| `ternary` | `sensitivity_aware_progressive_freeze` | 32 | 0.932500 | 0.025833 | 0.017500 | 0.024167 |
| `ternary` | `input_projection_aware_progressive_freeze` | 32 | 0.923333 | 0.035000 | 0.008333 | 0.033333 |
| `ternary` | `blockwise_scale_mutation_repair` | 32 | 0.915833 | 0.042500 | 0.000833 | 0.040833 |
| `ternary` | `qat_reference` | 32 | 0.956667 | 0.001667 | 0.041667 | 0.000000 |
| `binary` | `direct_low_bit_quant` | 32 | 0.846667 | 0.111667 | 0.000000 | |
| `binary` | `post_quant_mutation_repair` | 32 | 0.898333 | 0.060000 | 0.051667 | 0.025833 |
| `binary` | `distance_only_progressive_freeze` | 32 | 0.917500 | 0.040833 | 0.070833 | 0.006667 |
| `binary` | `sensitivity_aware_progressive_freeze` | 32 | 0.895833 | 0.062500 | 0.049167 | 0.028333 |
| `binary` | `input_projection_aware_progressive_freeze` | 32 | 0.885833 | 0.072500 | 0.039167 | 0.038333 |
| `binary` | `blockwise_scale_mutation_repair` | 32 | 0.855833 | 0.102500 | 0.009167 | 0.068333 |
| `binary` | `qat_reference` | 32 | 0.924167 | 0.034167 | 0.077500 | 0.000000 |

## Interpretation

Progressive freeze did beat post-quant mutation repair, but not uniformly with the same schedule:

```text
ternary best non-QAT:
  sensitivity_aware_progressive_freeze = 0.932500
  post_quant_mutation_repair = 0.928333
  delta = +0.004167

binary best non-QAT:
  distance_only_progressive_freeze = 0.917500
  post_quant_mutation_repair = 0.898333
  delta = +0.019167
```

Sensitivity-aware freeze was positive on ternary, but not binary. Binary was best with distance-only freeze and got close to QAT:

```text
binary QAT = 0.924167
binary distance-only = 0.917500
gap = 0.006667
```

The input-projection-aware heuristic did not help in this version:

```text
ternary input-aware vs distance = -0.002500
binary input-aware vs distance = -0.031667
```

Blockwise scale mutation also did not become the main improvement:

```text
ternary scale vs post-quant = -0.012500
binary scale vs post-quant = -0.042500
```

## What Broke Or Failed

The explicit input-projection-aware schedule underperformed even though input projection remained sensitive. The likely issue is not that input projection is irrelevant, but that the current special handling freezes it too conservatively or biases mutations in a way that hurts generalization.

The binary sensitivity-aware run was brittle:

```text
freeze_round_rollback_count = 3
solve_passed = false
```

Distance-only was surprisingly robust on binary, which means the low-bit schedule may need separate policies for ternary and binary.

## Next Recommended Experiment

The next narrow step should not add architecture. It should isolate why binary likes distance-only while ternary likes sensitivity-aware:

```text
E7A9_LEVEL_SPECIFIC_FREEZE_POLICY_AUDIT
```

Core tests:

```text
separate ternary policy from binary policy
audit freeze ordering by block and round
test input_projection late-freeze vs early-repair instead of current input-aware heuristic
test sensitivity weights 0.15 / 0.35 / 0.55
keep QAT as reference only
```

## Boundary

E7A8 is only a controlled low-bit matrix-core repair strategy test. It does not make broad reasoning or model-scale claims.

# E7A9 Binary Freeze Policy Upper-Bound Audit Result

## Decision

```text
decision = e7a9_binary_quality_competitive
deterministic_replay_passed = true
checker_failure_count = 0
```

Run root:

```text
target/pilot_wave/e7a9_binary_freeze_policy_upper_bound_audit
```

## What Ran

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

Widths:

```text
16, 32
```

The decision is based on the best deterministic width, which was width 32.

## Best Method Table

| method | eval | gap to int4 | gap to ternary | compression vs float32 |
|---|---:|---:|---:|---:|
| `float32_matrix_core` | 0.948333 | | | 1.000x |
| `int8_direct` | 0.950000 | | | 3.939x |
| `int4_direct` | 0.950000 | 0.000000 | | 7.758x |
| `ternary_qat_reference` | 0.945833 | 0.004167 | 0.000000 | 15.061x |
| `binary_direct` | 0.850833 | 0.099167 | 0.095000 | 28.453x |
| `binary_qat_baseline` | 0.938333 | 0.011667 | 0.007500 | 28.453x |
| `binary_qat_best_effort` | 0.944167 | 0.005833 | 0.001667 | 12.928x |
| `binary_distance_paramwise_freeze` | 0.930000 | 0.020000 | 0.015833 | 28.453x |
| `binary_sensitivity_paramwise_freeze` | 0.942500 | 0.007500 | 0.003333 | 28.453x |
| `binary_qat_warmstart_paramwise_freeze` | 0.947500 | 0.002500 | -0.001667 | 12.928x |
| `binary_direct_mutation_repair` | 0.908333 | 0.041667 | 0.037500 | 28.453x |
| `mixed_input_int4_state_binary_output_int4` | 0.938333 | 0.011667 | 0.007500 | 15.024x |
| `mixed_input_ternary_state_binary_output_int4` | 0.913333 | 0.036667 | 0.032500 | 18.667x |

## Main Findings

Binary direct quantization is not good enough:

```text
binary_direct = 0.850833
solve_passed = false
```

But binary becomes quality-competitive when trained/repaired properly:

```text
binary_qat_warmstart_paramwise_freeze = 0.947500
int4_direct                          = 0.950000
ternary_qat_reference                = 0.945833
```

This satisfies the E7A9 binary competitiveness rule:

```text
binary_best >= ternary_best - 0.015
binary_best >= int4_best - 0.025
no heldout/OOD/counterfactual/adversarial collapse
```

## Pure Binary vs Per-Channel Binary

The best overall binary used per-channel scale, so its compression estimate is lower:

```text
binary_qat_warmstart_paramwise_freeze:
  eval = 0.947500
  compression = 12.928x
```

But a more compact pure/global-scale binary path also remained competitive:

```text
binary_sensitivity_paramwise_freeze:
  eval = 0.942500
  compression = 28.453x
```

So the result is not just "binary works if we spend many scale parameters." There is also a high-compression binary branch that stayed close to ternary/int4.

## Paramwise Freeze Result

Paramwise one-by-one freeze helped, especially when warm-started from binary QAT:

```text
binary_qat_best_effort              = 0.944167
binary_qat_warmstart_paramwise_freeze = 0.947500
delta = +0.003333
```

The pure sensitivity-aware paramwise path was also strong:

```text
binary_qat_baseline                 = 0.938333
binary_sensitivity_paramwise_freeze = 0.942500
```

Distance-only was weaker:

```text
binary_distance_paramwise_freeze = 0.930000
```

## Mixed Precision

The tested mixed-precision options did not win:

```text
mixed input int4 / state binary / output int4     = 0.938333
mixed input ternary / state binary / output int4  = 0.913333
```

This does not rule out mixed precision generally, but these two simple layouts were not better than binary QAT warmstart or int4.

## Interpretation

Binary is no longer just a weak compression sidequest on this proxy. The best binary route reached near-int4 and slightly above the ternary QAT reference:

```text
binary_best = 0.947500
int4        = 0.950000
ternary     = 0.945833
```

Practical conclusion:

```text
quality-first path: int4 remains simplest and slightly best
compression-first path: binary is now viable
balanced path: binary QAT + paramwise freeze deserves follow-up
```

## Next Recommended Experiment

Run:

```text
E7A10_BINARY_SCALE_OVERHEAD_AND_DEPLOYABILITY_AUDIT
```

Purpose:

```text
separate true binary benefit from scale-overhead benefit
compare global scale, per-block scale, per-channel scale
measure bit cost honestly
test whether binary_sensitivity_paramwise_freeze can close the remaining gap without per-channel scale overhead
```

## Boundary

E7A9 is only a controlled low-bit matrix-core quality/compression tradeoff audit. It does not make broad reasoning or model-scale claims.

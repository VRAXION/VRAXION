# E7A4 Quantized Bridge From Backprop To Mutation Matrix Result

## Decision

```text
decision = e7a4_quantized_matrix_core_preserved_without_repair
checker = failure_count 0
deterministic_replay = passed
final_e7_verdict = intentionally deferred
```

Run root:

```text
target/pilot_wave/e7a4_quantized_bridge_from_backprop_to_mutation_matrix
```

E7A4 is a toy bridge test. It does not confirm a final matrix-medium architecture.

## Main Result

The bridge worked:

```text
float_mlp_backprop_reference solved = true
float_matrix_core_backprop solved = true
quantized_matrix_core_no_repair solved = true
quantized_matrix_core_mutation_repair solved = true
random_control solved = false
```

Smallest passing width:

```text
float_mlp_backprop_reference = 4
float_matrix_core_backprop = 4
quantized_matrix_core_no_repair = 4
quantized_matrix_core_mutation_repair = 4
random_control = none
```

Best eval accuracy:

| system | best width | matrix cells | eval accuracy |
|---|---:|---:|---:|
| `float_mlp_backprop_reference` | 32 | 1024 | 0.974166666667 |
| `float_matrix_core_backprop` | 16 | 256 | 0.967500000000 |
| `quantized_matrix_core_no_repair` | 16 | 256 | 0.967500000000 |
| `quantized_matrix_core_mutation_repair` | 32 | 1024 | 0.965000000000 |
| `random_control` | 32 | 0 | 0.261666666667 |

## Width Sweep

Float matrix-core backprop:

| width | eval | heldout | OOD | counterfactual | adversarial | pass |
|---:|---:|---:|---:|---:|---:|---|
| 4 | 0.917500000000 | 0.923333333333 | 0.916666666667 | 0.923333333333 | 0.906666666667 | true |
| 8 | 0.957500000000 | 0.956666666667 | 0.956666666667 | 0.966666666667 | 0.950000000000 | true |
| 16 | 0.967500000000 | 0.963333333333 | 0.963333333333 | 0.976666666667 | 0.966666666667 | true |
| 32 | 0.967500000000 | 0.963333333333 | 0.963333333333 | 0.976666666667 | 0.966666666667 | true |

Quantized matrix-core without repair:

| width | eval | heldout | OOD | counterfactual | adversarial | pass |
|---:|---:|---:|---:|---:|---:|---|
| 4 | 0.915833333333 | 0.923333333333 | 0.916666666667 | 0.920000000000 | 0.903333333333 | true |
| 8 | 0.957500000000 | 0.956666666667 | 0.956666666667 | 0.966666666667 | 0.950000000000 | true |
| 16 | 0.967500000000 | 0.966666666667 | 0.960000000000 | 0.976666666667 | 0.966666666667 | true |
| 32 | 0.967500000000 | 0.963333333333 | 0.963333333333 | 0.976666666667 | 0.966666666667 | true |

Quantized matrix-core with mutation repair:

| width | eval | heldout | OOD | counterfactual | adversarial | pass |
|---:|---:|---:|---:|---:|---:|---|
| 4 | 0.916666666667 | 0.930000000000 | 0.913333333333 | 0.916666666667 | 0.906666666667 | true |
| 8 | 0.955833333333 | 0.960000000000 | 0.956666666667 | 0.960000000000 | 0.946666666667 | true |
| 16 | 0.964166666667 | 0.960000000000 | 0.956666666667 | 0.966666666667 | 0.973333333333 | true |
| 32 | 0.965000000000 | 0.953333333333 | 0.970000000000 | 0.980000000000 | 0.956666666667 | true |

## Quantization

Quantization did not materially damage the learned matrix-core:

```text
width 4  quantization_delta = -0.001666666667
width 8  quantization_delta =  0.000000000000
width 16 quantization_delta =  0.000000000000
width 32 quantization_delta =  0.000000000000
```

Mutation repair was not needed for solve. It stayed solved, but did not improve the already-good quantized core:

```text
width 4  repair_delta_vs_quantized =  0.000833333334
width 8  repair_delta_vs_quantized = -0.001666666667
width 16 repair_delta_vs_quantized = -0.003333333333
width 32 repair_delta_vs_quantized = -0.002500000000
```

## Interpretation

E7A4 resolves the ambiguity from E7A3:

```text
The matrix-core architecture is learnable with backprop.
The learned matrix-core survives integer quantization.
Mutation repair is not required on this toy task because quantization barely hurts.
Naive mutation from scratch failed in E7A3, but mutation from a learned seed remains viable as a repair path.
```

The key scientific update is:

```text
E7A3 failure was not evidence against matrix-core.
It was evidence against from-scratch naive integer mutation search.
```

Recommended next:

```text
E7A5_QUANTIZATION_STRESS_AND_REPAIR_LIMIT
```

That should deliberately make quantization harsher:

```text
int8 -> int4 -> int3 -> binary/ternary
then test whether mutation repair can recover quality.
```

This would finally test whether mutation adds value after the learned matrix is damaged enough to need repair.

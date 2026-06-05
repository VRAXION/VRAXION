# E7A3 Neural Matrix Substrate Harness Result

## Decision

```text
decision = e7a3_backprop_reference_solved_only
checker = failure_count 0
deterministic_replay = passed
final_e7_verdict = intentionally deferred
```

Run root:

```text
target/pilot_wave/e7a3_neural_matrix_substrate_harness
```

E7A3 is a toy substrate-size harness. It does not confirm a final matrix-medium architecture.

## Main Result

On the margin-filtered linearly dominant toy task:

```text
float_mlp_backprop solved
integer_mlp_mutation did not solve
integer_matrix_hidden_replacement_mutation did not solve
random_control did not solve
```

Smallest passing width:

```text
float_mlp_backprop = 4
integer_mlp_mutation = none
integer_matrix_hidden_replacement_mutation = none
random_control = none
```

Best eval accuracy:

| system | best width | matrix cells | eval accuracy |
|---|---:|---:|---:|
| `float_mlp_backprop` | 32 | 1024 | 0.974166666667 |
| `integer_mlp_mutation` | 8 | 64 | 0.674166666667 |
| `integer_matrix_hidden_replacement_mutation` | 8 | 64 | 0.645000000000 |
| `random_control` | 4 | 0 | 0.255000000000 |

## Width Sweep

Float MLP backprop:

| width | eval | heldout | OOD | counterfactual | adversarial | pass |
|---:|---:|---:|---:|---:|---:|---|
| 4 | 0.928333333333 | 0.933333333333 | 0.933333333333 | 0.936666666667 | 0.910000000000 | true |
| 8 | 0.957500000000 | 0.963333333333 | 0.956666666667 | 0.963333333333 | 0.946666666667 | true |
| 16 | 0.968333333333 | 0.963333333333 | 0.963333333333 | 0.980000000000 | 0.966666666667 | true |
| 32 | 0.974166666667 | 0.966666666667 | 0.970000000000 | 0.980000000000 | 0.980000000000 | true |
| 64 | 0.973333333334 | 0.960000000000 | 0.980000000000 | 0.976666666667 | 0.976666666667 | true |

Integer MLP mutation:

| width | eval | heldout | OOD | counterfactual | adversarial | pass |
|---:|---:|---:|---:|---:|---:|---|
| 4 | 0.610833333333 | 0.616666666667 | 0.643333333333 | 0.590000000000 | 0.593333333333 | false |
| 8 | 0.674166666667 | 0.686666666667 | 0.723333333333 | 0.670000000000 | 0.616666666667 | false |
| 16 | 0.614166666667 | 0.610000000000 | 0.616666666667 | 0.646666666667 | 0.583333333333 | false |
| 32 | 0.509166666667 | 0.526666666667 | 0.516666666667 | 0.473333333333 | 0.520000000000 | false |
| 64 | 0.450000000000 | 0.493333333333 | 0.420000000000 | 0.446666666667 | 0.440000000000 | false |

Integer matrix hidden replacement mutation:

| width | eval | heldout | OOD | counterfactual | adversarial | pass |
|---:|---:|---:|---:|---:|---:|---|
| 4 | 0.603333333333 | 0.590000000000 | 0.576666666667 | 0.603333333333 | 0.643333333333 | false |
| 8 | 0.645000000000 | 0.660000000000 | 0.656666666667 | 0.650000000000 | 0.613333333333 | false |
| 16 | 0.528333333333 | 0.530000000000 | 0.596666666667 | 0.523333333333 | 0.463333333333 | false |
| 32 | 0.457500000000 | 0.440000000000 | 0.426666666667 | 0.510000000000 | 0.453333333333 | false |
| 64 | 0.390000000000 | 0.410000000000 | 0.400000000000 | 0.370000000000 | 0.380000000000 | false |

## Interpretation

This answers the immediate size question at toy scale:

```text
A standard float MLP needs only width 4 to pass this clean toy task.
Best float performance is around width 32.
The equivalent hidden matrix at width 4 is 4x4 = 16 cells.
The best tested hidden matrix is 32x32 = 1024 cells.
```

But the mutation paths did not solve:

```text
integer MLP mutation peaked at width 8, eval 0.674166666667
matrix hidden replacement mutation peaked at width 8, eval 0.645000000000
```

The degradation at larger widths suggests the naive integer mutation operator is not scaling with parameter count. That is a mutation/search problem before it is a matrix-medium theory problem.

## Scientific Read

E7A3 gives a clean three-phase harness:

```text
1. standard float neural reference: works
2. integer mutation neural network: not yet viable
3. mutation matrix hidden replacement: not yet viable
```

So the current minimal viable route is not to add more matrix primitives. It is to improve the mutation operator or use a bridge path, such as quantizing a trained float network and then mutating from that seed.

Recommended next:

```text
E7A4_QUANTIZED_BRIDGE_FROM_BACKPROP_TO_MUTATION_MATRIX
```

That should test:

```text
float MLP -> rounded integer MLP -> mutation repair
float hidden layer -> extracted/rounded matrix core -> mutation repair
```

This would directly test whether the matrix substrate is reachable once backprop has already found a useful hidden representation.

# E7A5 Operator Cell Matrix Incremental Scan Result

## Decision

```text
decision = e7a5_operator_cell_no_advantage_detected
deterministic_replay_passed = true
checker_failure_count = 0
final_e7_verdict = intentionally deferred
```

Run root:

```text
target/pilot_wave/e7a5_operator_cell_matrix_incremental_scan/
```

## Evidence Summary

| variant | best width | float eval | delta vs plain | param ratio | param-normalized delta | best quantized eval | best repair eval |
|---|---:|---:|---:|---:|---:|---:|---:|
| `plain_matrix_core_baseline` | 32 | 0.964000 | 0.000000 | 1.000 | 0.000000 | 0.962667 | 0.963333 |
| `soft_mask_matrix` | 16 | 0.961333 | -0.002667 | 0.501 | -0.002667 | 0.961333 | 0.962000 |
| `edge_bias_shared_activation` | 32 | 0.964667 | 0.000667 | 2.330 | -0.003562 | 0.964000 | 0.965333 |
| `per_cell_activation_soft_mixture` | 16 | 0.964667 | 0.000667 | 1.332 | -0.000768 | 0.965333 | 0.966000 |
| `source_target_trace_operand_cell` | 32 | 0.962000 | -0.002000 | 7.005 | -0.011733 | 0.962000 | 0.961333 |

## Interpretation

The operator-cell variants did not clear the positive threshold. The best raw float gain over the plain matrix-core was only `+0.000667`, far below the required `+0.02`, and it disappeared after parameter-normalized comparison.

Quantization remained stable across variants. Mutation repair produced only tiny changes, not a decisive repair-only advantage.

The result does not falsify operator-cell matrices in general. It says that this incremental E7A5 scan did not find evidence that edge bias, per-cell activation mixtures, or source/target/trace operands add meaningful value over the simpler E7A4-style matrix-core on this controlled toy task.

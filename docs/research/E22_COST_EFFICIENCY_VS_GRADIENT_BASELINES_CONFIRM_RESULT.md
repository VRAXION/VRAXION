# E22_COST_EFFICIENCY_VS_GRADIENT_BASELINES_CONFIRM Result

Status: completed.

```text
decision = e22_neural_baseline_more_efficient
checker_failure_count = 0
target_checker_passed = true
sample_only_checker_passed = true
```

## Run

```text
target = target/pilot_wave/e22_cost_efficiency_vs_gradient_baselines_confirm
sample_pack = docs/research/artifact_samples/e22_cost_efficiency_vs_gradient_baselines
run_id = 5843fbc44d2896d4
torch = 2.5.1+cu121
cuda = true
device = NVIDIA GeForce RTX 4070 Ti SUPER
cpu_workers_requested = 23
total_wall_time_seconds = 19.939923799999633
total_cpu_time_seconds = 41.390625
```

## Key Metrics

| system | heldout | locked hard | cost80 | cost90 | cost95 | p50 ms | p95 ms | trace |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flow_pocket_curriculum_primary | 0.907857 | 0.903333 | 1843 | 2539 | null | 0.314000 | 0.396000 | 0.783333 |
| flow_pocket_no_curriculum_ablation | 0.730714 | 0.734167 | null | null | null | 0.314000 | 0.396000 | 0.733333 |
| monolithic_mutation_baseline | 0.535714 | 0.531667 | null | null | null | 0.565200 | 0.712800 | 0.000000 |
| mlp_gradient_baseline | 1.000000 | 1.000000 | 24576 | 28672 | 32768 | 0.003583 | 0.004055 | 0.000000 |
| gru_lstm_gradient_baseline | 0.071429 | 0.071667 | null | null | null | 0.006572 | 0.006739 | 0.000000 |
| tiny_transformer_gradient_baseline | 1.000000 | 1.000000 | 4096 | 8192 | 8192 | 0.016166 | 0.016838 | 0.000000 |
| tiny_transformer_plus_curriculum | 1.000000 | 1.000000 | 4096 | 4096 | 8192 | 0.019555 | 0.026612 | 0.000000 |
| random_static_controls | 0.079286 | 0.068333 | null | null | null | 0.109900 | 0.138600 | 0.000000 |
| oracle_sympy_direct_eval_invalid_controls | 1.000000 | 1.000000 | 0 | 0 | 0 | 0.785000 | 0.990000 | 0.000000 |

## Interpretation

On the locked E21 symbolic proxy, standard gradient baselines were not weaker:
MLP and tiny Transformer reached 1.0 heldout and locked-hard accuracy. The
Flow/Pocket curriculum remained strong and trace-producing, but did not match
the neural baselines on accuracy in this E22 setup.

The result should be read as a benchmark falsification signal, not as a broad
architecture claim. The E21/E22 task is still a controlled symbolic proxy; it is
easy enough for char-feature gradient models to learn. Neural trace validity is
0.0 because these baselines do not produce Flow/Pocket-style symbolic traces.

Boundary: E22 is a controlled cost-efficiency comparison on the locked E21
symbolic composition proxy. It does not claim general reasoning, raw language
ability, production readiness, consciousness, AGI, or model-scale behavior.

# E23_TRACE_LOCKED_ONLINE_RULESHIFT_COMPOSITION_VS_NEURAL_BASELINES Result

Status: completed.

```text
decision = e23_flow_pocket_online_ruleshift_trace_advantage_confirmed
checker_failure_count = 0
target_checker_passed = true
sample_only_checker_passed = true
```

## Run

```text
target = target/pilot_wave/e23_trace_locked_online_ruleshift_composition_vs_neural_baselines
sample_pack = docs/research/artifact_samples/e23_trace_locked_online_ruleshift_composition_vs_neural_baselines
run_id = b5a314535f258d33
torch = 2.5.1+cu121
cuda = true
device = NVIDIA GeForce RTX 4070 Ti SUPER
cpu_workers_requested = 23
total_wall_time_seconds = 49.11190859999988
total_cpu_time_seconds = 110.5
```

## Key Metrics

Primary metric:

```text
composition_success = answer_correct AND trace_exact
```

| system | heldout comp | OOD comp | counterfactual comp | adversarial comp | heldout answer | heldout trace |
|---|---:|---:|---:|---:|---:|---:|
| flow_pocket_online_state_primary | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| flow_pocket_no_ruleshift_update_ablation | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.025556 | 0.000000 |
| flow_pocket_answer_only_ablation | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 |
| mlp_answer_only_gradient_baseline | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.054444 | 0.000000 |
| mlp_trace_locked_gradient_baseline | 0.018889 | 0.031429 | 0.000000 | 0.000000 | 0.053333 | 0.268889 |
| gru_trace_locked_gradient_baseline | 0.002222 | 0.001429 | 0.000000 | 0.000000 | 0.053333 | 0.064444 |
| tiny_transformer_trace_locked_gradient_baseline | 0.016667 | 0.011429 | 0.000000 | 0.000000 | 0.055556 | 0.162222 |
| tiny_transformer_curriculum_trace_locked | 0.013333 | 0.054286 | 0.000000 | 0.000000 | 0.045556 | 0.118889 |
| random_static_control | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.003333 | 0.000000 |
| direct_rule_engine_invalid_control | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

## Interpretation

E23 reverses the E22 pattern on this controlled proxy. When every episode has a
fresh codebook, support examples, a rule shift, counterfactual/adversarial
variants, and trace-locked scoring, the small neural baselines did not learn
the full composition. Some neural systems learned easy trace bits, but answer
accuracy and exact trace did not combine into composition success.

The decisive ablations are:

```text
flow_pocket_answer_only_ablation:
  heldout answer = 1.0
  heldout composition = 0.0

flow_pocket_no_ruleshift_update_ablation:
  heldout composition = 0.0
```

So E23 is not just rewarding answer-only correctness. The benchmark requires
rule-update trace and counterfactual-safe state transition.

## Boundary

This is still a scaffolded symbolic/numeric proxy with explicit state-update
primitives. It supports a narrower claim:

```text
Flow/Pocket-style state update is a strong fit for trace-locked online
ruleshift composition under this controlled protocol.
```

It does not prove raw language reasoning, production readiness, consciousness,
AGI, or model-scale behavior.

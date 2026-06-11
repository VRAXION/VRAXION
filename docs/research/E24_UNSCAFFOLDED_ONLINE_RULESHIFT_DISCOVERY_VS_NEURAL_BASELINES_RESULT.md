# E24_UNSCAFFOLDED_ONLINE_RULESHIFT_DISCOVERY_VS_NEURAL_BASELINES Result

Status: completed.

```text
decision = e24_flow_pocket_unsccaffolded_discovery_confirmed
checker_failure_count = 0
target_checker_passed = true
sample_only_checker_passed = true
```

## Run

```text
target = target/pilot_wave/e24_unscaffolded_online_ruleshift_discovery_vs_neural_baselines
sample_pack = docs/research/artifact_samples/e24_unscaffolded_online_ruleshift_discovery_vs_neural_baselines
run_id = 6f7dc35ddc5b3ae9
torch = 2.5.1+cu121
cuda = true
device = NVIDIA GeForce RTX 4070 Ti SUPER
cpu_workers_requested = 23
total_wall_time_seconds = 59.42676709999978
total_cpu_time_seconds = 124.828125
```

## Key Metrics

Primary metric:

```text
composition_success = answer_correct AND trace_exact
```

| system | heldout comp | OOD comp | counterfactual comp | adversarial comp | heldout answer | heldout trace |
|---|---:|---:|---:|---:|---:|---:|
| flow_pocket_unsccaffolded_discovery_primary | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| flow_pocket_marker_shortcut_ablation | 0.215000 | 0.213750 | 0.000000 | 0.000000 | 0.668000 | 0.215000 |
| flow_pocket_stale_rule_retention_ablation | 0.167000 | 0.166250 | 0.500000 | 0.000000 | 0.171000 | 0.167000 |
| flow_pocket_answer_only_ablation | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 |
| mlp_trace_locked_gradient_baseline | 0.000000 | 0.001250 | 0.000000 | 0.000000 | 0.057000 | 0.016000 |
| gru_trace_locked_gradient_baseline | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.052000 | 0.000000 |
| tiny_transformer_trace_locked_gradient_baseline | 0.004000 | 0.002500 | 0.000000 | 0.000000 | 0.071000 | 0.065000 |
| tiny_transformer_curriculum_trace_locked | 0.012000 | 0.010000 | 0.000000 | 0.000000 | 0.070000 | 0.139000 |
| random_static_control | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.002000 | 0.000000 |
| direct_rule_engine_invalid_control | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

## Interpretation

E24 preserves the E23 Flow/Pocket advantage after removing direct rule-change
assignments. The primary system has to infer changes from visible
support/evidence contradictions, false alarms, delayed shifts, and adversarial
decoys. Marker-only and stale-rule ablations collapse, which means the benchmark
is not rewarding a simple "notice means shift" shortcut.

Important ablation:

```text
flow_pocket_answer_only_ablation:
  heldout answer = 1.0
  heldout composition_success = 0.0
```

So answer-only correctness is explicitly insufficient.

The neural baselines learned some common trace bits, but did not combine answer
and exact trace into composition success on heldout/OOD/counterfactual/adversarial
splits.

## Boundary

This is still a controlled symbolic/numeric proxy with candidate operation
families and visible support rows. The supported claim is narrow:

```text
Flow/Pocket-style visible-evidence state discovery is a strong fit for this
trace-locked online ruleshift discovery protocol.
```

It does not prove raw language reasoning, production readiness, AGI,
consciousness, or model-scale behavior.

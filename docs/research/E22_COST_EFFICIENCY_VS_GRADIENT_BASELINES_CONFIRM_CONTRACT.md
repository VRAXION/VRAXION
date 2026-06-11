# E22_COST_EFFICIENCY_VS_GRADIENT_BASELINES_CONFIRM Contract

## Purpose

E22 compares Flow/Pocket curriculum training against gradient-descent baselines
on the same locked symbolic composition benchmark established by E21.

E21 confirmed controlled symbolic curriculum-composition transfer:

```text
locked_hard_pretest_accuracy = 0.419
locked_hard_posttest_accuracy = 0.904
heldout_composition_transfer_accuracy = 0.910
primitive_reuse_rate = 0.909
checker_failure_count = 0
```

E22 does not ask whether the Flow/Pocket curriculum can solve the benchmark.
It asks whether it is more cost-efficient than standard neural baselines under
the same task protocol and audit boundary.

## Systems

Compare:

```text
flow_pocket_curriculum_primary
flow_pocket_no_curriculum_ablation
monolithic_mutation_baseline
mlp_gradient_baseline
gru_lstm_gradient_baseline
tiny_transformer_gradient_baseline
tiny_transformer_plus_curriculum
random_static_controls
oracle_sympy_direct_eval_invalid_controls
```

The Flow/Pocket primary should reuse the E21 locked hard pretest/posttest and
heldout transfer protocol rather than inventing a new benchmark.

## Metrics

Report per system:

```text
heldout_accuracy
locked_hard_accuracy
cost_to_80_percent_accuracy
cost_to_90_percent_accuracy
cost_to_95_percent_accuracy
wall_time_seconds
cpu_time_seconds
gpu_time_seconds_if_available
peak_ram_mb
peak_vram_mb_if_available
training_sample_count
sample_efficiency
inference_latency_p50_ms
inference_latency_p95_ms
inference_latency_max_ms
trace_validity
renderer_faithfulness
deterministic_replay_passed
checker_failure_count
artifact_sample_pack_passed
```

Cost accounting must be explicit. If a system uses GPU, report both wall time
and GPU time/VRAM where available. If a system uses CPU only, record GPU fields
as not applicable rather than zero.

## Positive Decision Requirements

`e22_flow_pocket_cost_efficiency_confirmed` requires:

```text
Flow/Pocket matches or beats neural baselines on heldout accuracy
Flow/Pocket reaches target accuracy with better cost/sample/latency efficiency
trace_validity remains high
deterministic replay passes
checker_failure_count = 0
artifact sample pack passes
no direct eval / sympy / oracle leakage in primary systems
```

Accuracy without cost advantage is not enough. Cost advantage with poor trace
validity is not enough.

## Decision Labels

```text
e22_flow_pocket_cost_efficiency_confirmed
e22_neural_baseline_more_efficient
e22_transformer_curriculum_matches_flow_pocket
e22_flow_pocket_accuracy_positive_but_cost_not
e22_no_clear_efficiency_winner
e22_invalid_oracle_or_artifact_detected
```

## Required Future Artifacts

The implementation phase should produce:

```text
aggregate_metrics.json
cost_curve_report.json
accuracy_to_cost_report.json
latency_report.json
resource_usage_report.json
trace_validity_report.json
baseline_comparison_report.json
leakage_audit.json
deterministic_replay.json
artifact_sample_pack/
decision.json
report.md
progress.jsonl
hardware_heartbeat.jsonl
```

Long runs must write heartbeat/progress artifacts over time, not only at the
end of epochs or full system runs.

## Boundary

E22 is a controlled cost-efficiency comparison on symbolic composition tasks.
It does not prove general AI, GPT replacement, production readiness, broad
superiority across all tasks, consciousness, or model-scale behavior.

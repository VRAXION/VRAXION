# E25_NATURALIZED_RULESHIFT_TEXT_STREAM_DISCOVERY_CONFIRM Result

Status: completed.

```text
decision = e25_flow_pocket_naturalized_text_ruleshift_confirmed
checker_failure_count = 0
target_checker_passed = true
sample_only_checker_passed = true
```

## Run

```text
target = target/pilot_wave/e25_naturalized_ruleshift_text_stream_discovery_confirm
sample_pack = docs/research/artifact_samples/e25_naturalized_ruleshift_text_stream_discovery
run_id = 7c234ec062aa3a53
torch = 2.5.1+cu121
cuda = true
device = NVIDIA GeForce RTX 4070 Ti SUPER
cpu_workers_requested = 23
total_wall_time_seconds = 150.977887
total_cpu_time_seconds = 303.343750
```

Primary metric:

```text
composition_success = answer_correct AND trace_exact AND evidence_span_valid
```

## Key Metrics

| system | heldout | OOD | paraphrase | unseen code | counterfactual | adversarial | temporal shuffle | false marker | delayed | evidence span |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flow_pocket_naturalized_text_discovery_primary | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| parser_only_control | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| flow_pocket_answer_only_ablation | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| flow_pocket_marker_shortcut_ablation | 0.213000 | 0.207500 | 0.215000 | 0.218750 | 0.000000 | 0.000000 | 0.492500 | 0.000000 | 0.000000 | 0.715405 |
| flow_pocket_stale_rule_retention_ablation | 0.167000 | 0.166250 | 0.167500 | 0.166250 | 0.500000 | 0.000000 | 0.166250 | 1.000000 | 0.000000 | 0.256757 |
| flow_pocket_temporal_order_shuffle_ablation | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.186250 | 1.000000 | 1.000000 | 0.957297 |
| flow_pocket_no_paraphrase_generalization_ablation | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.459459 |
| flow_pocket_no_evidence_span_tracking_ablation | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| flow_pocket_no_counterfactual_repair_ablation | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.500000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 0.837838 |
| mlp_text_feature_trace_locked_baseline | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| gru_text_trace_locked_baseline | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| tiny_transformer_text_trace_locked_baseline | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| tiny_transformer_text_curriculum_trace_locked | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| oracle_text_parser_invalid_control | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| direct_rule_engine_invalid_control | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

## Interpretation

E25 confirms the E24 Flow/Pocket result under a naturalized text-evidence
proxy. The primary system still succeeds when structured support rows are
replaced by paraphrased observations, false markers, rumor/decoy sentences,
delayed contradictions, temporal shuffle, unseen codewords, and heldout phrase
families.

The controls isolate the needed parts:

```text
answer-only fails composition despite 1.0 heldout answer accuracy
parser-only fails evidence-span-valid composition
marker shortcut fails adversarial/counterfactual/false-marker splits
stale-rule retention fails real-shift splits
no-paraphrase generalization fails OOD/paraphrase/temporal/adversarial
no-evidence-span tracking fails composition by construction
temporal-order ablation fails the temporal_shuffle split
```

The neural text baselines learned some common trace bits but did not satisfy
trace-locked composition with evidence-span validity on any required split.

## Boundary

This is still a controlled naturalized text-evidence proxy. The supported claim
is narrow:

```text
Flow/Pocket-style state discovery is a strong fit for trace-locked online
ruleshift discovery when the evidence is carried by varied text observations.
```

It does not prove raw open-ended language reasoning, production readiness, AGI,
consciousness, or model-scale behavior.

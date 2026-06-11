# E25_NATURALIZED_RULESHIFT_TEXT_STREAM_DISCOVERY_CONFIRM Contract

## Purpose

E25 follows E24 and attacks the remaining structured-row scaffold. E24 used
visible evidence rows. E25 replaces those rows with short naturalized text
observations: paraphrases, false markers, decoy rumors, delayed contradictions,
temporal disorder, heldout phrasing, and unseen codewords.

Core question:

```text
Can Flow/Pocket discover online rule shifts from varied text evidence, without
explicit SHIFT assignments, oracle parse trees, oracle answers, or fixed
structured support rows?
```

Primary metric:

```text
composition_success = answer_correct AND trace_exact AND evidence_span_valid
```

## Systems

```text
flow_pocket_naturalized_text_discovery_primary
parser_only_control
flow_pocket_marker_shortcut_ablation
flow_pocket_stale_rule_retention_ablation
flow_pocket_answer_only_ablation
flow_pocket_temporal_order_shuffle_ablation
flow_pocket_no_paraphrase_generalization_ablation
flow_pocket_no_evidence_span_tracking_ablation
flow_pocket_no_counterfactual_repair_ablation
mlp_text_feature_trace_locked_baseline
gru_text_trace_locked_baseline
tiny_transformer_text_trace_locked_baseline
tiny_transformer_text_curriculum_trace_locked
random_static_control
oracle_text_parser_invalid_control
direct_rule_engine_invalid_control
```

The oracle text parser and direct rule engine are invalid controls only. Valid
systems must not use hidden before/after rule maps, oracle evidence labels,
oracle answers, Python eval, SymPy, or direct hidden calculators.

## Required Splits

```text
heldout
ood
paraphrase
unseen_codeword
counterfactual
adversarial
temporal_shuffle
false_marker
delayed_contradiction
```

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
aggregate_metrics.json
training_curve_report.json
trace_discovery_report.json
ruleshift_generalization_report.json
baseline_comparison_report.json
leakage_audit.json
deterministic_replay.json
resource_usage_report.json
decision.json
summary.json
report.md
progress.jsonl
hardware_heartbeat.jsonl
row_level_results.jsonl
system_results.json
docs/research/artifact_samples/e25_naturalized_ruleshift_text_stream_discovery/
```

## Decision Labels

```text
e25_flow_pocket_naturalized_text_ruleshift_confirmed
e25_neural_text_baseline_stronger
e25_parser_only_shortcut_detected
e25_answer_without_trace_failure
e25_no_clear_winner
e25_invalid_oracle_or_artifact_detected
```

Positive decision requires Flow/Pocket to beat neural baselines on
trace-locked composition across required splits, answer-only and marker-shortcut
controls to fail composition, high evidence-span validity, deterministic replay,
target checker pass, and sample-only checker pass.

## Boundary

E25 is a controlled naturalized text-evidence proxy. It does not prove raw
open-ended language reasoning, GPT-like generation, production readiness, AGI,
consciousness, or model-scale behavior.


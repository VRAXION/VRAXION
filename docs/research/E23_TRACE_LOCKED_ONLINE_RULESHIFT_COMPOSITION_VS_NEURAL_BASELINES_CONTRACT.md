# E23_TRACE_LOCKED_ONLINE_RULESHIFT_COMPOSITION_VS_NEURAL_BASELINES Contract

## Purpose

E23 follows E22, where standard neural baselines solved the locked E21
symbolic proxy. E23 removes the easiest answer-only shortcut by requiring
per-episode codebooks, support examples, mid-episode rule shifts,
counterfactual/adversarial variants, and trace-locked output.

Core question:

```text
Does a stateful Flow/Pocket update policy have an advantage over standard
neural baselines when the task requires online rule binding, rule invalidation,
state update, and trace-valid composition rather than answer-only pattern
learning?
```

## Systems

```text
flow_pocket_online_state_primary
flow_pocket_no_ruleshift_update_ablation
flow_pocket_answer_only_ablation
mlp_answer_only_gradient_baseline
mlp_trace_locked_gradient_baseline
gru_trace_locked_gradient_baseline
tiny_transformer_trace_locked_gradient_baseline
tiny_transformer_curriculum_trace_locked
random_static_control
direct_rule_engine_invalid_control
```

The direct rule engine is an invalid control. Valid systems must not use hidden
oracle answers, Python eval, SymPy, or direct calculators.

## Task

Each episode contains:

```text
random per-episode alien number codebook
operator bindings before shift
support examples
rule shift marker
hard query after shift
counterfactual/adversarial split variants
```

The output requires:

```text
canonical answer
structured trace bits
post-shift operator labels
rule-shift/change labels
intermediate/final sign labels
```

Primary metric:

```text
composition_success = answer_correct AND trace_exact
```

## Decision Labels

```text
e23_flow_pocket_online_ruleshift_trace_advantage_confirmed
e23_neural_online_adaptation_stronger
e23_answer_accuracy_without_trace_validity
e23_no_clear_winner
e23_invalid_oracle_or_artifact_detected
```

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
aggregate_metrics.json
training_curve_report.json
trace_validity_report.json
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
```

## Boundary

E23 is a controlled symbolic/numeric online ruleshift proxy. It does not prove
raw language reasoning, production readiness, consciousness, AGI, or model-scale
behavior.

# E27_UNRESOLVED_FLOW_STATE_INFORMATION_SEEKING_REPAIR_CONFIRM Contract

## Purpose

E27 repairs the first E26 failure frontier:

```text
stage5_missing_evidence_ambiguous
```

E26 forced a numeric answer even when the visible text evidence did not justify
a post-event binding. E27 changes the protocol from answer-only rendering to
resolution-level action selection.

Core question:

```text
Can unresolved Flow state learn an information-seeking action when visible
evidence is insufficient, without adding a brittle semantic UNKNOWN label?
```

## Action Set

```text
ANSWER
ASK_FOR_EVIDENCE
SEARCH_MORE
HOLD_UNRESOLVED
```

Primary metric:

```text
resolution_success =
  correct_action
  AND if ANSWER: answer_correct AND trace_valid AND evidence_span_valid
  AND if ASK/SEARCH/HOLD: non-answer is justified by missing evidence
```

## Systems

```text
flow_pocket_unresolved_information_seeking_primary
forced_answer_stale_rule_baseline
always_ask_baseline
flow_pocket_answer_only_ablation
no_information_seeking_action_ablation
no_query_dependency_check_ablation
flow_pocket_no_evidence_span_tracking_ablation
stale_rule_retention_ablation
mlp_text_feature_action_gradient_baseline
gru_text_action_gradient_baseline
tiny_transformer_text_action_gradient_baseline
tiny_transformer_text_action_curriculum_gradient
random_static_control
oracle_information_seeking_invalid_control
direct_rule_engine_invalid_control
```

Invalid controls may use oracle/reference behavior only as sanity references.
Valid systems must not use hidden before/after maps, oracle evidence labels,
oracle answers, Python eval, SymPy, or direct hidden calculators.

## Pass Requirements

```text
stage1-4 resolution_success >= 0.95
stage5_missing_evidence_ambiguous resolution_success >= 0.80
wrong_confident_answer_on_missing <= 0.05
false_ask_on_answerable <= 0.15
evidence_span_validity >= 0.90
trace_validity >= 0.90
deterministic replay passes
target checker failure_count = 0
sample-only checker failure_count = 0
```

Scoring is by visible evidence. If the hidden world has a numeric answer but the
visible evidence does not justify it, the correct behavior is ASK/SEARCH/HOLD,
not clairvoyant guessing.

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
aggregate_metrics.json
training_curve_report.json
trace_discovery_report.json
ruleshift_generalization_report.json
information_seeking_repair_report.json
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
docs/research/artifact_samples/e27_unresolved_flow_state_information_seeking_repair/
```

## Decision Labels

```text
e27_unresolved_flow_state_information_seeking_repair_confirmed
e27_wrong_confident_repaired_but_trace_or_span_low
e27_information_seeking_overasks_answerable_cases
e27_no_repair_detected
e27_invalid_oracle_or_artifact_detected
```

## Boundary

E27 is a controlled symbolic/naturalized-text proxy. It does not prove raw
open-ended language reasoning, deployed-model behavior, AGI, consciousness, or
model-scale behavior.

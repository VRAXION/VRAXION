# E26_HARD_SKIP_TEXT_REASONING_FAILURE_MAP Contract

## Purpose

E26 follows E25 by intentionally jumping beyond the confirmed naturalized
ruleshift text proxy. The goal is not to chase another all-green benchmark. The
goal is to find the first clean failure frontier.

Core question:

```text
Where does Flow/Pocket text-mediated state discovery first break when the test
moves from E25-style visible evidence to harder text reasoning families?
```

Primary metric:

```text
composition_success = answer_correct AND trace_exact AND evidence_span_valid
```

## Hard-Skip Families

```text
stage1_bridge_single_shift
stage2_multi_rule_document
stage3_long_decoy_dense
stage4_temporal_disorder
stage5_missing_evidence_ambiguous
stage6_indirect_language
stage7_long_context_memory
```

E26 is valid if it produces a replay-checked family-level failure map. A
partial decision is not a failed experiment when it includes the first failing
family, the failure signature, and a concrete repair recommendation.

## Systems

```text
flow_pocket_hard_skip_primary
parser_only_control
flow_pocket_marker_shortcut_ablation
flow_pocket_stale_rule_retention_ablation
flow_pocket_answer_only_ablation
flow_pocket_no_memory_ablation
flow_pocket_no_paraphrase_generalization_ablation
flow_pocket_no_evidence_span_tracking_ablation
mlp_text_feature_gradient_baseline
gru_text_gradient_baseline
tiny_transformer_text_gradient_baseline
tiny_transformer_text_curriculum_gradient
random_static_control
oracle_text_parser_invalid_control
direct_rule_engine_invalid_control
```

The oracle text parser and direct rule engine are invalid controls only. Valid
systems must not use hidden before/after rule maps, oracle evidence labels,
oracle answers, Python eval, SymPy, or direct hidden calculators.

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
aggregate_metrics.json
training_curve_report.json
trace_discovery_report.json
ruleshift_generalization_report.json
failure_map_report.json
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
docs/research/artifact_samples/e26_hard_skip_text_reasoning_failure_map/
```

## Decision Labels

```text
e26_hard_skip_text_reasoning_confirmed
e26_hard_skip_text_reasoning_partial
e26_hard_skip_text_reasoning_failed_with_failure_map
e26_invalid_oracle_or_artifact_detected
```

## Required Validation

The checker must verify:

```text
target checker failure_count = 0
sample-only checker failure_count = 0
deterministic replay hash match
row-level eval present
hard family list matches contract
first failing family is recomputable from row metrics
invalid controls are marked invalid
valid systems do not use oracle/direct-eval/SymPy leakage
no structured support rows or explicit shift assignments
progress and hardware heartbeat artifacts are non-empty
```

## Boundary

E26 is a controlled hard-skip naturalized text reasoning failure-map probe. It
does not prove raw open-ended language reasoning, GPT-like generation,
production readiness, AGI, consciousness, or model-scale behavior.

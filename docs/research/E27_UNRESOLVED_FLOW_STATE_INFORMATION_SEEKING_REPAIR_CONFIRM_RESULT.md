# E27_UNRESOLVED_FLOW_STATE_INFORMATION_SEEKING_REPAIR_CONFIRM Result

Status: completed.

```text
decision = e27_unresolved_flow_state_information_seeking_repair_confirmed
checker_failure_count = 0
target_checker_passed = true
sample_only_checker_passed = true
```

## Run

```text
target = target/pilot_wave/e27_unresolved_flow_state_information_seeking_repair_confirm
sample_pack = docs/research/artifact_samples/e27_unresolved_flow_state_information_seeking_repair
run_id = cf19745ca99d8b73
torch = 2.5.1+cu121
cuda = true
device = NVIDIA GeForce RTX 4070 Ti SUPER
cpu_workers_requested = 23
total_wall_time_seconds = 161.195873
total_cpu_time_seconds = 282.609375
```

Primary metric:

```text
resolution_success =
  correct_action
  AND if ANSWER: answer_correct AND trace_valid AND evidence_span_valid
  AND if ASK/SEARCH/HOLD: non-answer justified by missing evidence
```

## Key Result

```text
flow_pocket_unresolved_information_seeking_primary:
  stage1-4 min resolution_success = 1.000000
  stage5_missing_evidence_ambiguous = 1.000000
  wrong_confident_answer_on_missing = 0.000000
  false_ask_on_answerable = 0.000000
  evidence_span_validity = 1.000000
  trace_validity = 1.000000
```

## Comparison

| system | s1 | s2 | s3 | s4 | s5 | s6 | s7 | action | wrong confident | false ask | span |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flow_pocket_unresolved_information_seeking_primary | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| forced_answer_stale_rule_baseline | 0.333333 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.857143 | 0.142857 | 0.000000 | 0.190476 |
| always_ask_baseline | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.142857 | 0.000000 | 0.857143 | 1.000000 |
| no_information_seeking_action_ablation | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 0.857143 | 0.142857 | 0.000000 | 1.000000 |
| no_query_dependency_check_ablation | 0.333333 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 0.021111 | 0.857143 | 0.142857 | 0.000000 | 0.764921 |
| flow_pocket_no_evidence_span_tracking_ablation | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.857143 | 0.142857 | 0.000000 | 0.000000 |
| mlp_text_feature_action_gradient_baseline | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| gru_text_action_gradient_baseline | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| tiny_transformer_text_action_gradient_baseline | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| tiny_transformer_text_action_curriculum_gradient | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| oracle_information_seeking_invalid_control | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| direct_rule_engine_invalid_control | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |

## Interpretation

E27 confirms the intended E26 repair. The system no longer treats missing
post-event evidence as permission to guess. When a query depends on an
unproven post-event binding, the primary system selects `ASK_FOR_EVIDENCE`
instead of rendering a confident numeric answer.

The controls isolate the mechanism:

```text
forced-answer/stale-rule fails stage5 and keeps wrong confident answers
always-ask passes stage5 but over-asks answerable cases
no-information-seeking passes answerable stages but fails stage5
no-query-dependency-check fails stage5 and long-context evidence routing
no-evidence-span fails resolution despite many correct raw answers/actions
```

The neural action baselines learned the action channel in this setup
(`correct_action_rate = 1.0`) but still failed resolution because evidence span
validity stayed at zero. This is important: E27 does not reward a bare ASK
classifier. It requires action, trace, and evidence contract together.

## Boundary

This is still a controlled symbolic/naturalized-text proxy. The supported claim
is narrow:

```text
Flow/Pocket can repair the E26 missing-evidence failure by using an
information-seeking action when visible evidence is insufficient.
```

It does not prove raw open-ended language reasoning, production readiness, AGI,
consciousness, or model-scale behavior.

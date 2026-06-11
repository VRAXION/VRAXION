# E26_HARD_SKIP_TEXT_REASONING_FAILURE_MAP Result

Status: completed.

```text
decision = e26_hard_skip_text_reasoning_partial
checker_failure_count = 0
target_checker_passed = true
sample_only_checker_passed = true
```

## Run

```text
target = target/pilot_wave/e26_hard_skip_text_reasoning_failure_map
sample_pack = docs/research/artifact_samples/e26_hard_skip_text_reasoning_failure_map
run_id = 8011bc83a9cd5bc4
torch = 2.5.1+cu121
cuda = true
device = NVIDIA GeForce RTX 4070 Ti SUPER
cpu_workers_requested = 23
total_wall_time_seconds = 155.316021
total_cpu_time_seconds = 285.531250
```

Primary metric:

```text
composition_success = answer_correct AND trace_exact AND evidence_span_valid
```

## Failure Map

```text
best_valid_system = flow_pocket_hard_skip_primary
best_stage_passed = 4
first_failing_family = stage5_missing_evidence_ambiguous
failure_signature = missing_or_underdetermined_evidence
recommended_repair = add explicit unknown/underdetermined-state handling before answer rendering
```

| family | primary composition_success |
|---|---:|
| stage1_bridge_single_shift | 1.000000 |
| stage2_multi_rule_document | 1.000000 |
| stage3_long_decoy_dense | 1.000000 |
| stage4_temporal_disorder | 1.000000 |
| stage5_missing_evidence_ambiguous | 0.000000 |
| stage6_indirect_language | 0.000000 |
| stage7_long_context_memory | 0.063333 |

## Baseline Comparison

| system | s1 | s2 | s3 | s4 | s5 | s6 | s7 | evidence span |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flow_pocket_hard_skip_primary | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.063333 | 0.580476 |
| parser_only_control | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| flow_pocket_answer_only_ablation | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| flow_pocket_no_memory_ablation | 0.333333 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.047619 |
| flow_pocket_no_paraphrase_generalization_ablation | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.142857 |
| flow_pocket_no_evidence_span_tracking_ablation | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mlp_text_feature_gradient_baseline | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| gru_text_gradient_baseline | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| tiny_transformer_text_gradient_baseline | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| tiny_transformer_text_curriculum_gradient | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| oracle_text_parser_invalid_control | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| direct_rule_engine_invalid_control | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

## Interpretation

E26 answers the "hard skip" question: Flow/Pocket does not simply continue
through every harder text-reasoning family at 100 percent. It passes the first
four hard-skip stages, including multi-rule text documents, dense decoys, and
temporal disorder, then fails exactly when the query requires explicit handling
of missing or underdetermined evidence.

This is useful because the next repair is now localized:

```text
Do not add a larger neural baseline or a broader text parser first.
Add an explicit unknown / underdetermined-state branch before answer rendering,
then retest stage5 while preserving stages1-4.
```

The neural baselines did not pass trace-locked composition on any hard family
under this run. They did use CUDA and their losses moved, but composition stayed
at zero, so the result is not merely an unrun dependency placeholder.

## Boundary

This is still a controlled naturalized text reasoning proxy. The supported claim
is narrow:

```text
Flow/Pocket currently handles the first four E26 hard-skip text families, but
the next failure frontier is missing/underdetermined evidence, not generic
language understanding.
```

It does not prove raw open-ended language reasoning, production readiness, AGI,
consciousness, or model-scale behavior.

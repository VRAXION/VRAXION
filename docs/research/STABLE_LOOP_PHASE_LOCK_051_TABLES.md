# STABLE_LOOP_PHASE_LOCK_051 Tables

These tables are copied from 050 machine-audited output, not new hand-generated
claims. The 050 runner generated its paper tables from child 049
machine-readable artifacts:

```text
child_049/summary.json
child_049/metrics.jsonl
child_049/leakage_audit.jsonl
child_049/collapse_metrics.json
child_049/prediction_distribution.json
```

## Main Table

| Arm | Heldout | OOD | Family Min | Hard Distractor | Long OOD | Unique Outputs | Top Output Rate | Output Entropy | Collapse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ADVERSARIAL_FROZEN_ROUTE_GRAMMAR_TRAIN_AND_INFER | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 75 / 75 | 0.073 | 5.404 | false |
| ADVERSARIAL_FROZEN_ROUTE_GRAMMAR_ROLLBACK_GATED | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 75 / 75 | 0.073 | 5.404 | false |

## Ablation And Failure-Control Table

| Control | Expected Failure | Observed Metric |
| --- | --- | --- |
| NO_ROUTE_GRAMMAR_ADVERSARIAL_FROZEN_BASELINE | no-route collapse | collapse_detected = true, family_min_accuracy = 0.000, top_output_rate = 1.000 |
| FROZEN_EVAL_048_REFERENCE | previous scale reference collapses | collapse_detected = true, family_min_accuracy = 0.000, top_output_rate = 0.894 |
| ROUTE_GRAMMAR_SHUFFLED_LABELS | labels destroyed | heldout_exact_accuracy = 0.000, ood_exact_accuracy = 0.000 |
| RANDOM_LABEL_CONTROL | random labels fail family-min | family_min_accuracy = 0.000 |
| RANDOM_PHASE_RULE_CONTROL | aggregate-looking false positive caught | family_min_accuracy = 0.000, long_ood_accuracy = 0.000 |
| ALWAYS_SPACE_CONTROL | static space output caught | space_only_rate = 1.000 |
| ALWAYS_MAJORITY_CONTROL | majority shortcut caught | majority_output_rate = 1.000 |
| COPY_LAST_TOKEN_CONTROL | copy shortcut caught | copy_last_token_rate = 1.000 |

## Leakage-Audit Table

| Audit Field | Value |
| --- | ---: |
| train_eval_id_overlap_count | 0 |
| train_eval_input_overlap_count | 0 |
| train_eval_near_duplicate_count | 0 |
| train_eval_semantic_overlap_count | 0 |
| max_train_eval_token_jaccard | 0.667 |
| near_duplicate_threshold | 0.92 |

Expected source hashes:

```text
corpus_sha256_normalized_lf = 6b44848ab9483e8267103538ca58198b198a7651e9f20025168143fef4e5cd56
runner_sha256_normalized_lf = 4777b479294bc571751582dee53b05a121eae1465a45e24870432f4828b81046
```

## Claim Boundary

Supports:

```text
reviewer-facing reproduction package for bounded 049/050 adversarial frozen-eval result
```

Does not support:

```text
production default training
public beta promotion
production API readiness
full VRAXION
language grounding
consciousness
biological/FlyWire equivalence
physical quantum behavior
```

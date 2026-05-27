# D41 Rule Family Inference Router Scale Confirm Result

Local result file for D41 rule family inference router scale confirmation.

The runnable evidence is produced by:

```bash
python scripts/probes/run_d41_rule_family_inference_router_scale_confirm.py --out target/pilot_wave/d41_rule_family_inference_router_scale_confirm/smoke --seeds 8701,8702,8703,8704,8705,8706,8707,8708 --support-counts 1,2,3,5 --train-rows-per-seed 1200 --test-rows-per-seed 1200 --ood-rows-per-seed 1200 --generations 600 --population 160 --workers auto --cpu-target saturate --heartbeat-sec 20
python scripts/probes/run_d41_rule_family_inference_router_scale_confirm_check.py --check-only --out target/pilot_wave/d41_rule_family_inference_router_scale_confirm/smoke
```

## Result Status

decision = `rule_family_inference_router_scale_confirmed`

verdict = `D41_RULE_FAMILY_INFERENCE_ROUTER_SCALE_CONFIRMED`

next = `D42_RAW_SUPPORT_EVIDENCE_EXTRACTION_PROTOTYPE`

artifact_path = `target/pilot_wave/d41_rule_family_inference_router_scale_confirm/smoke`

This was the full D41 smoke configuration, not scale-lite:

- seeds: `8701,8702,8703,8704,8705,8706,8707,8708`
- support counts: `1,2,3,5`
- train/test/OOD rows per seed: `1200/1200/1200`
- generations: `600`
- population: `160`
- workers: `8`
- wall clock seconds: `595.5983819961548`
- failed jobs: `0`

## Upstream Check

D40 upstream fields were present in `d40_upstream_manifest.json`:

- d40_decision = `rule_family_inference_router_prototype_positive`
- d40_verdict = `D40_RULE_FAMILY_INFERENCE_ROUTER_PROTOTYPE_POSITIVE`
- d40_next = `D41_RULE_FAMILY_INFERENCE_ROUTER_SCALE_CONFIRM`
- D40 learned selected-pocket test/OOD = `1.0 / 1.0`
- D40 rule-family test/OOD = `1.0 / 1.0`
- D40 wrong-support follow rate = `1.0`
- D40 same-query-different-support = `1.0`

## Dataset Invariants

| Metric | Value |
| --- | ---: |
| duplicate_target_pocket_rate | 0.0 |
| missing_target_pocket_rate | 0.0 |
| expected_selected_points_to_target_rate | 1.0 |
| ambiguous_support_rate | 0.0 |
| multi_family_support_tie_rate | 0.0 |
| intended_family_unique_evidence_rate | 1.0 |
| counterfactual_target_collision_rate | 0.0 |
| wrong_support_query_mismatch_rate | 1.0 |

## OOD Audit

| Metric | Value |
| --- | ---: |
| support_rule_oracle_test_accuracy | 1.0 |
| support_rule_oracle_ood_accuracy | 1.0 |
| support_selected_pocket_oracle_test_accuracy | 1.0 |
| support_selected_pocket_oracle_ood_accuracy | 1.0 |
| known_rule_oracle_test_accuracy | 1.0 |
| known_rule_oracle_ood_accuracy | 1.0 |
| ood_label_rule_changed | false |

## Arm Metrics

| Arm | train | test | OOD |
| --- | ---: | ---: | ---: |
| RANDOM_BASELINE | 0.1090625 | 0.11 | 0.10916666666666666 |
| QUERY_ONLY_MONOLITHIC_BASELINE | 0.20802083333333335 | 0.19802083333333334 | 0.19541666666666668 |
| SUPPORT_EVIDENCE_ORACLE_RULE_SELECTOR | 1.0 | 1.0 | 1.0 |
| TRUE_FAMILY_ORACLE_UPPER_BOUND | 1.0 | 1.0 | 1.0 |
| MUTABLE_LEARNED_RULE_FAMILY_INFERENCE | 1.0 | 1.0 | 1.0 |
| MUTABLE_LEARNED_RULE_INFERENCE_PLUS_LEARNED_ROUTER | 0.996875 | 0.996875 | 0.996875 |
| SHUFFLED_SUPPORT_EVIDENCE_CONTROL | 0.0 | 0.0 | 0.0 |
| NO_SUPPORT_EVIDENCE_CONTROL | 0.2 | 0.2 | 0.2 |
| WRONG_SUPPORT_CONTROL | 0.0 | 0.0 | 0.0 |
| SAME_QUERY_DIFFERENT_SUPPORT_COUNTERFACTUAL | 1.0 | 1.0 | 1.0 |
| SUPPORT_COUNT_GENERALIZATION_REPORT_ONLY | 1.0 | 1.0 | 1.0 |
| SUPPORT_MARGIN_STRATA_REPORT_ONLY | 1.0 | 1.0 | 1.0 |

## Learned Inference Metrics

| Metric | Value |
| --- | ---: |
| learned_rule_family_train_accuracy | 1.0 |
| learned_rule_family_test_accuracy | 1.0 |
| learned_rule_family_ood_accuracy | 1.0 |
| learned_selected_pocket_train_accuracy | 1.0 |
| learned_selected_pocket_test_accuracy | 1.0 |
| learned_selected_pocket_ood_accuracy | 1.0 |
| min_seed_learned_test_accuracy | 1.0 |
| min_seed_learned_ood_accuracy | 1.0 |
| min_support_count_accuracy | 1.0 |
| min_margin_strata_accuracy | 1.0 |
| low_margin_error_rate | 0.0 |
| median_score_margin | 8.189332715776182 |

## Learned Breakdowns

Per support count:

| support_count_1 | support_count_2 | support_count_3 | support_count_5 |
| ---: | ---: | ---: | ---: |
| 1.0 | 1.0 | 1.0 | 1.0 |

Per margin stratum:

| clean_unanimous | margin_high | margin_low | noisy_majority |
| ---: | ---: | ---: | ---: |
| 1.0 | 1.0 | 1.0 | 1.0 |

Per family selected-pocket accuracy:

| row | col | pair | mirror | diag |
| ---: | ---: | ---: | ---: | ---: |
| 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

## Control Deltas

| Delta | Value |
| --- | ---: |
| learned_vs_query_only_test_delta | 0.8019791666666667 |
| learned_vs_shuffled_support_test_delta | 1.0 |
| learned_vs_no_support_test_delta | 0.8 |
| same_query_different_support_accuracy | 1.0 |
| wrong_support_follow_rate_test | 1.0 |
| wrong_support_selected_pocket_test_accuracy | 0.0 |
| wrong_support_query_mismatch_rate_test | 1.0 |

## Rule Identity

| Metric | Value |
| --- | ---: |
| rule_identity_alignment_score_mean | 1.0 |
| rule_identity_alignment_score_min | 1.0 |
| rule_diagonal_mass_mean | 0.9990556554112107 |
| rule_off_diagonal_mass_mean | 0.0009443445887893093 |
| rule_entropy_mean | 0.0048705596526348064 |

The learned argmax mapping was identity for every seed:

```json
{"8701":{"row":"row","col":"col","pair":"pair","mirror":"mirror","diag":"diag"},"8702":{"row":"row","col":"col","pair":"pair","mirror":"mirror","diag":"diag"},"8703":{"row":"row","col":"col","pair":"pair","mirror":"mirror","diag":"diag"},"8704":{"row":"row","col":"col","pair":"pair","mirror":"mirror","diag":"diag"},"8705":{"row":"row","col":"col","pair":"pair","mirror":"mirror","diag":"diag"},"8706":{"row":"row","col":"col","pair":"pair","mirror":"mirror","diag":"diag"},"8707":{"row":"row","col":"col","pair":"pair","mirror":"mirror","diag":"diag"},"8708":{"row":"row","col":"col","pair":"pair","mirror":"mirror","diag":"diag"}}
```

## Mutation Metrics

accepted_mutations_by_type:

```json
{"prune_small_weights":1,"rule_bias_delta":8,"rule_column_delta":7,"rule_column_swap":5,"rule_row_delta":227,"rule_row_swap":5,"rule_weight_delta":774}
```

rejected_mutations_by_type:

```json
{"prune_small_weights":16253,"rule_bias_delta":16220,"rule_column_delta":16022,"rule_column_swap":16208,"rule_row_delta":15670,"rule_row_swap":16155,"rule_weight_delta":66765}
```

mutation_acceptance_rate = `0.00625`

convergence_generation_median = `20.0`

seed_variance:

```json
{"ood_accuracy_variance":0.0,"rule_family_ood_accuracy_variance":0.0,"rule_family_test_accuracy_variance":0.0,"test_accuracy_variance":0.0}
```

## Boundary

A positive D41 proves only that support-evidence-based rule-family inference scale-confirms while feeding the known-rule router path on a controlled symbolic pocket task. It does not prove raw visual Raven reasoning, full hidden-rule Raven solving, natural-language reasoning, DNA/genome success, Raven solved, architecture superiority, consciousness, or general intelligence.

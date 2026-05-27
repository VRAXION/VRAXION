# D40 Rule Family Inference Router Prototype Result

Local result file for D40 rule family inference router prototype.

The runnable evidence is produced by:

```bash
python scripts/probes/run_d40_rule_family_inference_router_prototype.py --out target/pilot_wave/d40_rule_family_inference_router_prototype/smoke --seeds 8601,8602,8603,8604,8605 --support-count 3 --train-rows-per-seed 700 --test-rows-per-seed 700 --ood-rows-per-seed 700 --generations 500 --population 128 --workers auto --cpu-target saturate --heartbeat-sec 20
python scripts/probes/run_d40_rule_family_inference_router_prototype_check.py --check-only --out target/pilot_wave/d40_rule_family_inference_router_prototype/smoke
```

## Result Status

Full D40 smoke was run locally, not scale-lite.

- artifact path: `target/pilot_wave/d40_rule_family_inference_router_prototype/smoke`
- seeds: `8601,8602,8603,8604,8605`
- support_count: 3
- rows per seed: train 700, test 700, OOD 700
- generation cap: 500
- population: 128
- workers: 8
- completed jobs: 50
- failed jobs: 0
- decision: `rule_family_inference_router_prototype_positive`
- verdict: `D40_RULE_FAMILY_INFERENCE_ROUTER_PROTOTYPE_POSITIVE`
- next: `D41_RULE_FAMILY_INFERENCE_ROUTER_SCALE_CONFIRM`

## Dataset Invariants

- duplicate_target_pocket_rate = 0.0
- missing_target_pocket_rate = 0.0
- expected_selected_points_to_target_rate = 1.0
- ambiguous_support_rate = 0.0
- multi_family_support_tie_rate = 0.0
- intended_family_unique_evidence_rate = 1.0
- support_evidence_margin_count_min = 3
- support_evidence_margin_normalized_min = 1.0

## Support Evidence Audit

- support_rule_oracle_test_accuracy = 1.0
- support_rule_oracle_ood_accuracy = 1.0
- support_selected_pocket_oracle_test_accuracy = 1.0
- support_selected_pocket_oracle_ood_accuracy = 1.0

## OOD Rule Invariance

- support_rule_oracle_test_accuracy = 1.0
- support_rule_oracle_ood_accuracy = 1.0
- known_rule_oracle_test_accuracy = 1.0
- known_rule_oracle_ood_accuracy = 1.0
- ood_label_rule_changed = false

## Metrics

| Arm | train | test | ood |
| --- | ---: | ---: | ---: |
| RANDOM_BASELINE | 0.1151 | 0.1134 | 0.1137 |
| QUERY_ONLY_MONOLITHIC_BASELINE | 0.2251 | 0.2006 | 0.1974 |
| SUPPORT_EVIDENCE_ORACLE_RULE_SELECTOR | 1.0000 | 1.0000 | 1.0000 |
| TRUE_FAMILY_ORACLE_UPPER_BOUND | 1.0000 | 1.0000 | 1.0000 |
| MUTABLE_LEARNED_RULE_FAMILY_INFERENCE | 1.0000 | 1.0000 | 1.0000 |
| MUTABLE_LEARNED_RULE_INFERENCE_PLUS_LEARNED_ROUTER | 1.0000 | 1.0000 | 1.0000 |
| SHUFFLED_SUPPORT_EVIDENCE_CONTROL | 0.0000 | 0.0000 | 0.0000 |
| NO_SUPPORT_EVIDENCE_CONTROL | 0.2000 | 0.2000 | 0.2000 |
| WRONG_SUPPORT_CONTROL | 0.0000 | 0.0000 | 0.0000 |
| SAME_QUERY_DIFFERENT_SUPPORT_COUNTERFACTUAL | 1.0000 | 1.0000 | 1.0000 |

## Learned Rule Inference

- learned_rule_family_train_accuracy = 1.0
- learned_rule_family_test_accuracy = 1.0
- learned_rule_family_ood_accuracy = 1.0
- learned_selected_pocket_train_accuracy = 1.0
- learned_selected_pocket_test_accuracy = 1.0
- learned_selected_pocket_ood_accuracy = 1.0
- min_seed_learned_test_accuracy = 1.0
- min_seed_learned_ood_accuracy = 1.0
- per-family selected-pocket accuracy: row 1.0, col 1.0, pair 1.0, mirror 1.0, diag 1.0

## Controls

- learned_vs_query_only_test_delta = 0.7994285714285714
- learned_vs_shuffled_support_test_delta = 1.0
- learned_vs_no_support_test_delta = 0.8
- same_query_different_support_accuracy = 1.0
- wrong_support_follow_rate_test = 1.0
- wrong_support_query_mismatch_rate_test = 1.0
- wrong_support_selected_pocket_test_accuracy = 0.0

## Rule Identity

- rule_identity_alignment_score_mean = 1.0
- rule_identity_alignment_score_min = 1.0
- rule_diagonal_mass_mean = 0.9964355334608136
- rule_off_diagonal_mass_mean = 0.0035644665391864234
- rule_entropy_mean = 0.013258297067010937
- rule_argmax_mapping_by_seed: every seed maps row->row, col->col, pair->pair, mirror->mirror, diag->diag

## Mutation Metrics

- accepted_mutations_by_type = `{prune_small_weights: 0, rule_bias_delta: 0, rule_column_delta: 1, rule_column_swap: 1, rule_row_delta: 39, rule_row_swap: 1, rule_weight_delta: 263}`
- rejected_mutations_by_type = `{prune_small_weights: 3465, rule_bias_delta: 3390, rule_column_delta: 3340, rule_column_swap: 3325, rule_row_delta: 3281, rule_row_swap: 3430, rule_weight_delta: 18504}`
- mutation_acceptance_rate = 0.0078125
- convergence_generation_median = 20
- seed_variance.test_accuracy_variance = 0.0
- seed_variance.ood_accuracy_variance = 0.0

## Boundary

A positive D40 proves only that support-evidence-based rule-family inference can feed the known-rule router path on a controlled symbolic pocket task. It does not prove raw visual Raven reasoning, full hidden-rule Raven solving, natural-language reasoning, DNA/genome success, Raven solved, architecture superiority, consciousness, or general intelligence.

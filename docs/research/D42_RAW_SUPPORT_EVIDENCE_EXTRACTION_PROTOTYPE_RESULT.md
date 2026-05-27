# D42 Raw Support Evidence Extraction Prototype Result

Local result file for D42 raw support evidence extraction prototype.

The runnable evidence is produced by:

```bash
python scripts/probes/run_d42_raw_support_evidence_extraction_prototype.py --out target/pilot_wave/d42_raw_support_evidence_extraction_prototype/smoke --seeds 8801,8802,8803,8804,8805 --support-counts 1,2,3,5 --train-rows-per-seed 800 --test-rows-per-seed 800 --ood-rows-per-seed 800 --generations 700 --population 160 --workers auto --cpu-target saturate --heartbeat-sec 20
python scripts/probes/run_d42_raw_support_evidence_extraction_prototype_check.py --check-only --out target/pilot_wave/d42_raw_support_evidence_extraction_prototype/smoke
```

## Result Status

decision = `raw_support_evidence_extraction_prototype_positive`

verdict = `D42_RAW_SUPPORT_EVIDENCE_EXTRACTION_PROTOTYPE_POSITIVE`

next = `D43_RAW_SUPPORT_EVIDENCE_EXTRACTION_SCALE_CONFIRM`

artifact_path = `target/pilot_wave/d42_raw_support_evidence_extraction_prototype/smoke`

This was the full D42 smoke configuration, not scale-lite:

- seeds: `8801,8802,8803,8804,8805`
- support counts: `1,2,3,5`
- train/test/OOD rows per seed: `800/800/800`
- generations requested: `700`
- population: `160`
- workers: `8`
- wall clock seconds: `66.12237215042114`
- failed jobs: `0`

## Upstream Check

D41 upstream fields were present in `d41_upstream_manifest.json`:

- d41_decision = `rule_family_inference_router_scale_confirmed`
- d41_verdict = `D41_RULE_FAMILY_INFERENCE_ROUTER_SCALE_CONFIRMED`
- d41_next = `D42_RAW_SUPPORT_EVIDENCE_EXTRACTION_PROTOTYPE`
- D41 learned selected-pocket test/OOD = `1.0 / 1.0`
- D41 plus-router selected-pocket test/OOD = `0.996875 / 0.996875`
- D41 support-count and margin strata pass = `true / true`
- D41 wrong-support and same-query controls pass = `true / true`

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
| support_evidence_oracle_test_accuracy | 1.0 |
| support_evidence_oracle_ood_accuracy | 1.0 |
| support_selected_pocket_oracle_test_accuracy | 1.0 |
| support_selected_pocket_oracle_ood_accuracy | 1.0 |
| known_rule_oracle_test_accuracy | 1.0 |
| known_rule_oracle_ood_accuracy | 1.0 |
| ood_label_rule_changed | false |

## Arm Metrics

| Arm | train | test | OOD |
| --- | ---: | ---: | ---: |
| RANDOM_BASELINE | 0.11674999999999999 | 0.10475 | 0.10925 |
| QUERY_ONLY_BASELINE | 0.229 | 0.185 | 0.2025 |
| PRECOMPUTED_SUPPORT_EVIDENCE_UPPER_BOUND | 1.0 | 1.0 | 1.0 |
| ORACLE_RAW_SUPPORT_EVIDENCE_EXTRACTOR | 1.0 | 1.0 | 1.0 |
| MUTABLE_LEARNED_RAW_SUPPORT_EVIDENCE_EXTRACTOR | 1.0 | 1.0 | 1.0 |
| SHUFFLED_CENTER_CONTROL | 0.07975 | 0.07525 | 0.063 |
| SHUFFLED_FORMULA_CANDIDATE_CONTROL | 0.0 | 0.0 | 0.0 |
| NO_CENTER_CONTROL | 0.2 | 0.2 | 0.2 |
| NO_FORMULA_CANDIDATE_CONTROL | 0.2 | 0.2 | 0.2 |
| WRONG_SUPPORT_CONTROL | 0.0 | 0.0 | 0.0 |
| SAME_QUERY_DIFFERENT_RAW_SUPPORT_COUNTERFACTUAL | 1.0 | 1.0 | 1.0 |
| RAW_SUPPORT_PLUS_LEARNED_ROUTER_COMPOSITION | 1.0 | 1.0 | 1.0 |

## Learned Raw Extractor Metrics

| Metric | Value |
| --- | ---: |
| learned_raw_extractor_rule_family_train_accuracy | 1.0 |
| learned_raw_extractor_rule_family_test_accuracy | 1.0 |
| learned_raw_extractor_rule_family_ood_accuracy | 1.0 |
| learned_raw_extractor_selected_pocket_train_accuracy | 1.0 |
| learned_raw_extractor_selected_pocket_test_accuracy | 1.0 |
| learned_raw_extractor_selected_pocket_ood_accuracy | 1.0 |
| min_seed_learned_raw_extractor_test_accuracy | 1.0 |
| min_seed_learned_raw_extractor_ood_accuracy | 1.0 |
| min_support_count_accuracy | 1.0 |
| min_margin_strata_accuracy | 1.0 |

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
| learned_vs_query_only_test_delta | 0.815 |
| learned_vs_shuffled_center_test_delta | 0.92475 |
| learned_vs_no_center_test_delta | 0.8 |
| learned_vs_shuffled_formula_candidate_test_delta | 1.0 |
| learned_vs_no_formula_candidate_test_delta | 0.8 |
| same_query_different_raw_support_accuracy | 1.0 |
| wrong_support_follow_rate_test | 1.0 |
| wrong_support_selected_pocket_test_accuracy | 0.0 |
| wrong_support_query_mismatch_rate_test | 1.0 |

## Extractor Identity

Equality kernel:

- equality_kernel_diagonal_mass = `0.999998687091761`
- equality_kernel_off_diagonal_mass = `0.0000013129082390347395`
- equality_kernel_entropy = `0.000009932060434714901`
- equality_kernel_argmax_mapping = `{"0":"0","1":"1","2":"2","3":"3","4":"4","5":"5","6":"6","7":"7","8":"8"}`

Channel gate:

- channel_gate_identity_alignment_score = `1.0`
- channel_gate_diagonal_mass = `0.9999993462437965`
- channel_gate_off_diagonal_mass = `0.0000006537562034634846`
- channel_gate_argmax_mapping_by_seed = identity mapping for `8801,8802,8803,8804,8805`

## Mutation Metrics

accepted_mutations_by_type:

```json
{"channel_column_delta":0,"channel_column_swap":0,"channel_row_delta":250,"channel_row_swap":0,"channel_weight_delta":0,"equality_column_delta":0,"equality_column_swap":0,"equality_row_delta":450,"equality_row_swap":0,"equality_weight_delta":0,"prune_small_weights":0,"rule_bias_delta":0}
```

rejected_mutations_by_type:

```json
{"channel_column_delta":0,"channel_column_swap":0,"channel_row_delta":0,"channel_row_swap":0,"channel_weight_delta":0,"equality_column_delta":0,"equality_column_swap":0,"equality_row_delta":0,"equality_row_swap":0,"equality_weight_delta":0,"prune_small_weights":0,"rule_bias_delta":0}
```

mutation_acceptance_rate = `1.0`

convergence_generation_median = `0`

seed_variance:

```json
{"ood_accuracy_variance":0.0,"rule_family_ood_accuracy_variance":0.0,"rule_family_test_accuracy_variance":0.0,"test_accuracy_variance":0.0}
```

## Boundary

A positive D42 proves only that a learned raw symbolic support-evidence extractor can feed the D41 support-rule-router stack on a controlled symbolic task, with fixed formula primitive candidates available. It does not prove raw visual Raven reasoning, formula primitive discovery, full hidden-rule Raven solving, natural-language reasoning, DNA/genome success, Raven solved, architecture superiority, consciousness, or general intelligence.

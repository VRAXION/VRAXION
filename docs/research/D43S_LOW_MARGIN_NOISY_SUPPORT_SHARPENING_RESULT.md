# D43S Low-Margin Noisy Support Sharpening Result

Status: positive.

Artifact: `target/pilot_wave/d43s_low_margin_noisy_support_sharpening/smoke`

D43S targeted the D43 failure tail: low-margin noisy-majority support evidence where the learned soft equality/channel field compressed small intended evidence gaps. The first full replay reproduced the tail. The hardened candidate then switched the combined model from margin-weighted soft equality evidence to a learned hard-vote readout: the learned equality kernel supplies the symbol argmax, and each support board contributes one vote. This keeps the learned raw extractor boundary while avoiding symbol-specific margin weighting that can distort 3-vs-2 noisy-majority cases.

## Decision

- decision: `low_margin_noisy_support_tail_hardened`
- verdict: `D43S_LOW_MARGIN_NOISY_SUPPORT_TAIL_HARDENED`
- next: `D44_FORMULA_PRIMITIVE_DISCOVERY_PLAN`
- failed jobs: `0`

## Dataset And OOD Audits

- duplicate_target_pocket_rate: `0.0`
- missing_target_pocket_rate: `0.0`
- expected_selected_points_to_target_rate: `1.0`
- ambiguous_support_rate: `0.0`
- multi_family_support_tie_rate: `0.0`
- intended_family_unique_evidence_rate: `1.0`
- counterfactual_target_collision_rate: `0.0`
- wrong_support_query_mismatch_rate: `1.0`
- support_evidence_oracle_test_accuracy: `1.0`
- support_evidence_oracle_ood_accuracy: `1.0`
- support_selected_pocket_oracle_test_accuracy: `1.0`
- support_selected_pocket_oracle_ood_accuracy: `1.0`
- known_rule_oracle_test_accuracy: `1.0`
- known_rule_oracle_ood_accuracy: `1.0`
- ood_label_rule_changed: `false`

## Arm Metrics

| Arm | Train | Test | OOD | Noisy Low |
| --- | ---: | ---: | ---: | ---: |
| D43_BASELINE_REPLAY | 0.9980 | 0.9678 | 0.9774 | 0.9540 |
| MARGIN_PRESERVING_OBJECTIVE | 0.9990 | 0.9926 | 0.9936 | 0.9894 |
| TEMPERATURE_SHARPENED_EVIDENCE | 0.9828 | 0.9748 | 0.9820 | 0.9640 |
| FAMILY_BALANCED_EDGE_OVERSAMPLING | 0.9922 | 0.9892 | 0.9890 | 0.9846 |
| COMBINED_SHARPENED_MODEL | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| HARD_VOTE_ORACLE_UPPER_BOUND | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| SHUFFLED_CENTER_CONTROL | 0.0762 | 0.0776 | 0.0816 | 0.1000 |
| SHUFFLED_FORMULA_CANDIDATE_CONTROL | 0.0002 | 0.0000 | 0.0000 | 0.0000 |
| NO_CENTER_CONTROL | 0.2000 | 0.2000 | 0.2000 | 0.2143 |
| SAME_QUERY_DIFFERENT_RAW_SUPPORT_COUNTERFACTUAL | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| WRONG_SUPPORT_CONTROL | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Tail Comparison

- baseline noisy_majority_margin_low_accuracy: `0.954`
- combined noisy_majority_margin_low_accuracy: `1.0`
- noisy_low_delta: `0.04600000000000004`
- easy_case_regression: `0.0`
- baseline test tail errors: `161`
- baseline OOD tail errors: `113`
- combined test tail errors: `0`
- combined OOD tail errors: `0`
- baseline support_count_3_low_margin_accuracy: `0.999`
- baseline support_count_5_low_margin_accuracy: `0.894`
- combined support_count_3_low_margin_accuracy: `1.0`
- combined support_count_5_low_margin_accuracy: `1.0`
- baseline low_margin_error_rate: `0.0462`
- combined low_margin_error_rate: `0.0`
- baseline median_evidence_gap: `0.11405166092461377`
- combined median_evidence_gap: `0.8823529411764706`

## Controls

- shuffled_center_test_accuracy: `0.0776`
- shuffled_formula_candidate_test_accuracy: `0.0`
- no_center_test_accuracy: `0.2`
- same_query_different_raw_support_accuracy: `1.0`
- wrong_support_follow_rate_test: `1.0`
- wrong_support_follow_rate_ood: `1.0`
- wrong_support_selected_pocket_test_accuracy: `0.0`
- wrong_support_selected_pocket_ood_accuracy: `0.0`

## Learned Extractor Reports

- equality_kernel_argmax_mapping: `0->0, 1->1, 2->2, 3->3, 4->4, 5->5, 6->6, 7->7, 8->8`
- equality_kernel_diagonal_mass: `0.23288127065887412`
- equality_kernel_off_diagonal_mass: `0.7671187293411259`
- equality_kernel_entropy: `0.9580870754643579`
- channel_gate_identity_alignment_score: `0.68`
- channel_gate_identity_alignment_score_min: `0.6`
- channel_gate_diagonal_mass: `0.5336752148559308`
- channel_gate_off_diagonal_mass: `0.4663247851440692`
- channel_gate_entropy: `0.6193888696814767`
- mutation_acceptance_rate: `0.00625`
- convergence_generation_median: `20`

## Boundary

D43S only tests low-margin noisy-support tail hardening for D43 on a controlled symbolic task with fixed formula primitive candidates.

It does not prove formula primitive discovery, raw visual Raven reasoning, Raven solved, DNA/genome success, consciousness, AGI, general intelligence, or architecture superiority.

# D39 Learned Router Layer Scale Confirm Result

Local result file for D39 learned router layer scale confirmation.

The runnable evidence is produced by:

```bash
python scripts/probes/run_d39_learned_router_layer_scale_confirm.py --out target/pilot_wave/d39_learned_router_layer_scale_confirm/smoke --seeds 8501,8502,8503,8504,8505,8506,8507,8508 --train-rows-per-seed 800 --test-rows-per-seed 800 --ood-rows-per-seed 800 --generations 500 --population 128 --workers auto --cpu-target saturate --heartbeat-sec 20
python scripts/probes/run_d39_learned_router_layer_scale_confirm_check.py --check-only --out target/pilot_wave/d39_learned_router_layer_scale_confirm/smoke
```

## Result Status

Full D39 smoke was run locally, not scale-lite.

- artifact path: `target/pilot_wave/d39_learned_router_layer_scale_confirm/smoke`
- seeds: `8501,8502,8503,8504,8505,8506,8507,8508`
- rows per seed: train 800, test 800, OOD 800
- generation cap: 500
- population: 128
- workers: 8
- completed jobs: 48
- failed jobs: 0
- decision: `learned_conditioning_router_field_scale_confirmed`
- verdict: `D39_LEARNED_CONDITIONING_ROUTER_FIELD_SCALE_CONFIRMED`
- next: `D40_RULE_FAMILY_INFERENCE_ROUTER_PROTOTYPE`

The learned mutation/search jobs converged at generation 20 for all eight seeds and stopped after 61 executed generations per seed.

## Dataset Invariants

- duplicate_target_pocket_rate = 0.0
- missing_target_pocket_rate = 0.0
- expected_selected_points_to_target_rate = 1.0

## OOD Rule Invariance

- known_rule_oracle_test_accuracy = 1.0
- known_rule_oracle_ood_accuracy = 1.0
- ood_label_rule_changed = false

## Metrics

| Arm | train | test | ood |
| --- | ---: | ---: | ---: |
| MONOLITHIC_FORMULA_BASELINE | 0.3352 | 0.2998 | 0.3127 |
| ORACLE_GATED_RULE_FORMULA_UPPER_BOUND | 1.0000 | 1.0000 | 1.0000 |
| MUTABLE_LEARNED_ROUTER_GATE | 1.0000 | 1.0000 | 1.0000 |
| SHUFFLED_GATE_CONTROL | 0.1077 | 0.1141 | 0.1136 |
| NO_FAMILY_INPUT_CONTROL | 0.3497 | 0.3177 | 0.3206 |
| EXPLICIT_TARGET_STATE_UPPER_BOUND | 1.0000 | 1.0000 | 1.0000 |

## Learned Router Details

- min_seed_learned_gate_test_accuracy = 1.0
- min_seed_learned_gate_ood_accuracy = 1.0
- per-family accuracy: row 1.0, col 1.0, pair 1.0, mirror 1.0, diag 1.0
- gate_identity_alignment_score_mean = 1.0
- gate_identity_alignment_score_min = 1.0
- diagonal_gate_mass_mean = 0.999279112642582
- off_diagonal_gate_mass_mean = 0.0007208873574180186
- gate_entropy_mean = 0.0034639393694700887
- seed_variance.test_accuracy_variance = 0.0
- seed_variance.ood_accuracy_variance = 0.0

## Deltas

- monolithic_vs_learned_test_delta = 0.70015625
- learned_vs_shuffled_test_delta = 0.8859375
- learned_vs_no_family_test_delta = 0.68234375

## Mutation Metrics

- accepted_mutations_by_type = `{gate_bias_delta: 1, gate_column_delta: 1, gate_column_swap: 1, gate_row_delta: 0, gate_row_swap: 2, gate_weight_delta: 483, pocket_bias_delta: 0, prune_small_weights: 0}`
- rejected_mutations_by_type = `{gate_bias_delta: 4831, gate_column_delta: 4682, gate_column_swap: 4833, gate_row_delta: 4731, gate_row_swap: 4684, gate_weight_delta: 26943, pocket_bias_delta: 4753, prune_small_weights: 4743}`
- mutation_acceptance_rate = 0.008041128394410757
- convergence_generation_median = 20.0

## Boundary

A positive D39 proves only that the learned conditioning/router field scale-confirms on a controlled known-rule symbolic pocket task. It does not prove hidden-rule Raven reasoning, natural-language reasoning, DNA/genome success, Raven solved, architecture superiority, consciousness, or general intelligence.

# Phase D8.0 Archive + Psi Replay Audit

Verdict: **D8_PSI_ARCHIVE_OFFLINE_PASS**

## Summary

- D8.0 is offline-only: no live search, no SAF changes, no raw candidate scan.
- State identity is `source::run_id::panel_index`; parent identity is previous panel in the same run.
- The audit tests whether panel fingerprints predict future gain and whether archive replay beats score-only retention.
- Archive replay uses seed-held-out `Psi` predictions where available, with in-sample fallback only for missing predictions.

## Coverage

```json
{
  "used_roots": [
    "phase_b_full_20260424",
    "phase_b1_horizon_ties_20260425",
    "phase_d1_activation_20260425",
    "phase_d2_cross_h_activation_20260426",
    "phase_d3_klock_coarse_20260426",
    "phase_d3_fine_k_20260426",
    "phase_d4_softness_20260426",
    "phase_d7_operator_bandit_20260427"
  ],
  "runs": 297,
  "panel_rows": 5840,
  "H_values": [
    128,
    256,
    384
  ],
  "sources": [
    "phase_b1_horizon_ties_20260425",
    "phase_b_full_20260424",
    "phase_d1_activation_20260425",
    "phase_d2_cross_h_activation_20260426",
    "phase_d3_fine_k_20260426",
    "phase_d3_klock_coarse_20260426",
    "phase_d4_softness_20260426",
    "phase_d7_operator_bandit_20260427"
  ],
  "archive_size": 64,
  "bucket_bins": 4,
  "basin_delta": 0.005,
  "psi_replay_prediction": "seed-held-out CV where available; in-sample per-H fallback only for missing predictions"
}
```

## Decision

```json
{
  "psi_seed_cv_spearman": 0.6396861242611915,
  "score_seed_cv_spearman": 0.3100859817259444,
  "time_seed_cv_spearman": 0.6264919733177691,
  "negative_control_max_abs_spearman": 0.08390027219940974,
  "psi_beats_baselines": true,
  "controls_fail": true,
  "selector_stats": {
    "A1_PHI_BUCKET": {
      "h_deltas": {
        "128": -1.5625000000000014e-05,
        "256": 9.375000000000464e-05,
        "384": 0.0005312499999999862
      },
      "pass_h_count": 2,
      "material_regression": false
    },
    "A2_BUCKET_PSI": {
      "h_deltas": {
        "128": 0.020312499999999997,
        "256": 0.025390625000000007,
        "384": 0.023656249999999983
      },
      "pass_h_count": 3,
      "material_regression": false
    },
    "A3_BUCKET_PSI_NOVELTY": {
      "h_deltas": {
        "128": 0.020203125000000002,
        "256": 0.026921875000000005,
        "384": 0.020453124999999985
      },
      "pass_h_count": 3,
      "material_regression": false
    }
  }
}
```

## Psi Validation

```text
           target                   model     cv  valid    n        r2  spearman   pearson  groups
future_gain_local                 psi_phi   seed   True 5840  0.068659  0.343608  0.290401       5
future_gain_local                 psi_phi source   True 5840  0.128663  0.380787  0.359422       8
future_gain_local                 psi_phi  phase   True 5840  0.128663  0.380787  0.359422       8
future_gain_local                 psi_phi      H   True 5840 -0.295568  0.210666  0.210492       3
future_gain_local              score_only   seed   True 5840  0.014904  0.225349  0.152936       5
future_gain_local              score_only source   True 5840  0.037332  0.237851  0.196280       8
future_gain_local              score_only  phase   True 5840  0.037332  0.237851  0.196280       8
future_gain_local              score_only      H   True 5840  0.004597  0.232218  0.172229       3
future_gain_local               time_only   seed   True 5840  0.060150  0.291361  0.247538       5
future_gain_local               time_only source   True 5840  0.074861  0.311575  0.273634       8
future_gain_local               time_only  phase   True 5840  0.074861  0.311575  0.273634       8
future_gain_local               time_only      H   True 5840  0.068417  0.304935  0.263096       3
  future_gain_mid                 psi_phi   seed   True 5840  0.109447  0.457185  0.369373       5
  future_gain_mid                 psi_phi source   True 5840  0.222585  0.514603  0.472171       8
  future_gain_mid                 psi_phi  phase   True 5840  0.222585  0.514603  0.472171       8
  future_gain_mid                 psi_phi      H   True 5840 -0.715891  0.264986  0.237222       3
  future_gain_mid              score_only   seed   True 5840  0.022815  0.273898  0.195435       5
  future_gain_mid              score_only source   True 5840  0.065108  0.287260  0.258283       8
  future_gain_mid              score_only  phase   True 5840  0.065108  0.287260  0.258283       8
  future_gain_mid              score_only      H   True 5840  0.038772  0.280068  0.242617       3
  future_gain_mid               time_only   seed   True 5840  0.093495  0.402778  0.310662       5
  future_gain_mid               time_only source   True 5840  0.125809  0.431260  0.354739       8
  future_gain_mid               time_only  phase   True 5840  0.125809  0.431260  0.354739       8
  future_gain_mid               time_only      H   True 5840  0.112648  0.419652  0.337622       3
future_gain_final                 psi_phi   seed   True 5840  0.350598  0.639686  0.600950       5
future_gain_final                 psi_phi source   True 5840  0.423582  0.672565  0.651777       8
future_gain_final                 psi_phi  phase   True 5840  0.423582  0.672565  0.651777       8
future_gain_final                 psi_phi      H   True 5840 -0.234938  0.414043  0.429047       3
future_gain_final              score_only   seed   True 5840  0.090670  0.310086  0.317213       5
future_gain_final              score_only source   True 5840  0.120766  0.323145  0.350528       8
future_gain_final              score_only  phase   True 5840  0.120766  0.323145  0.350528       8
future_gain_final              score_only      H   True 5840  0.116766  0.318808  0.349939       3
future_gain_final               time_only   seed   True 5840  0.285668  0.626492  0.534749       5
future_gain_final               time_only source   True 5840  0.288644  0.628571  0.537528       8
future_gain_final               time_only  phase   True 5840  0.288644  0.628571  0.537528       8
future_gain_final               time_only      H   True 5840  0.237901  0.599883  0.491825       3
future_gain_final target_shuffle_H_source   seed   True 5840  0.027394  0.083900  0.166154       5
future_gain_final         feature_shuffle   seed   True 5840 -0.005741  0.007696 -0.008984       5
```

## Archive Replay

```text
  H              selector  n_selected  mean_future_gain  median_future_gain  topk_basin_precision  mean_current_peak  coverage_buckets  base_mean_future_gain  base_basin_precision  delta_mean_future_gain_vs_score  delta_basin_precision_vs_score
128         S0_SCORE_TOPN          64          0.000234               0.000              0.000000           0.056688                53               0.000234              0.000000                         0.000000                        0.000000
128         A1_PHI_BUCKET          64          0.000219               0.000              0.000000           0.056484                64               0.000234              0.000000                        -0.000016                        0.000000
128         A2_BUCKET_PSI          64          0.020547               0.020              0.984375           0.018141                64               0.000234              0.000000                         0.020312                        0.984375
128 A3_BUCKET_PSI_NOVELTY          64          0.020438               0.020              1.000000           0.018156                64               0.000234              0.000000                         0.020203                        1.000000
256         S0_SCORE_TOPN          64          0.000203               0.000              0.015625           0.079562                56               0.000203              0.015625                         0.000000                        0.000000
256         A1_PHI_BUCKET          64          0.000297               0.000              0.015625           0.078781                64               0.000203              0.015625                         0.000094                        0.000000
256         A2_BUCKET_PSI          64          0.025594               0.025              0.906250           0.020609                64               0.000203              0.015625                         0.025391                        0.890625
256 A3_BUCKET_PSI_NOVELTY          64          0.027125               0.031              0.890625           0.020000                64               0.000203              0.015625                         0.026922                        0.875000
384         S0_SCORE_TOPN          64          0.002125               0.000              0.187500           0.077828                48               0.002125              0.187500                         0.000000                        0.000000
384         A1_PHI_BUCKET          64          0.002656               0.000              0.218750           0.076062                64               0.002125              0.187500                         0.000531                        0.031250
384         A2_BUCKET_PSI          64          0.025781               0.027              0.890625           0.014141                64               0.002125              0.187500                         0.023656                        0.703125
384 A3_BUCKET_PSI_NOVELTY          64          0.022578               0.020              0.781250           0.015641                64               0.002125              0.187500                         0.020453                        0.593750
```

## Interpretation

- `Psi` and archive replay show enough offline signal to justify D8.1 instrumentation-only logging before any live archive steering.

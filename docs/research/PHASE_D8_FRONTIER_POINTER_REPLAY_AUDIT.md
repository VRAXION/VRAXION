# Phase D8.2 Frontier Pointer Replay Audit

Verdict: **D8_POINTER_REPLAY_PASS**

## Summary

- D8.2 is offline-only: no live Rust run, no SAF change, no K change, no acceptance change.
- The primary replay uses OOF-only `psi_pred_seed_cv`; in-sample `psi_pred` fallback is not used.
- Behavior-sphere cell assignment excludes score and time features.
- D8.1.1 scan-depth knees are used only as confidence gates, not as proof that a cell is predictive.

## Coverage

```json
{
  "input": "output\\phase_d8_archive_psi_replay_20260427\\analysis\\panel_state_dataset.csv",
  "knee_summary": "output\\phase_d8_scan_depth_knee_20260427\\analysis\\summary.json",
  "rows_oof": 5840,
  "H_values": [
    128,
    256,
    384
  ],
  "anchor_counts": [
    16,
    32,
    64,
    128
  ],
  "anchor_seeds": [
    11,
    23,
    37
  ],
  "archive_size": 64,
  "time_buckets": 10,
  "random_seed": 8128,
  "features": [
    "stable_rank",
    "kernel_rank",
    "separation_sp",
    "collision_rate",
    "f_active",
    "unique_predictions",
    "edges",
    "accept_rate_window"
  ],
  "knee_by_H": {
    "128": 8,
    "256": 13,
    "384": 5
  },
  "psi_column": "psi_pred_seed_cv",
  "psi_policy": "OOF-only; drop missing predictions"
}
```

## Decision

```json
{
  "required_h": 2,
  "h_values": [
    128,
    256,
    384
  ],
  "p4_pass_baselines_h": 3,
  "p4_beats_random_h": 3,
  "p4_competitive_with_peer_h": 3,
  "p4_material_regression": false,
  "p1_pass_baselines_h": 3,
  "p1_material_regression": false,
  "best_selector_overall": "P2_PSI_CONF",
  "best_selector_mean_future_gain": 0.01690818194845561,
  "p4_gap_vs_best_selector": -0.00021093750000000105,
  "by_h": {
    "128": {
      "p4_mean_future_gain": 0.017056804719333442,
      "p4_delta_vs_score": 0.016822429719333443,
      "p4_delta_vs_time": 0.014791179719333442,
      "p4_delta_vs_random": 0.014963127537173862,
      "p4_confident_cell_rate": 0.776713989235541,
      "p4_unique_cells": 39.916666666666664
    },
    "256": {
      "p4_mean_future_gain": 0.01867471514424848,
      "p4_delta_vs_score": 0.018471590144248488,
      "p4_delta_vs_time": 0.0169247151442485,
      "p4_delta_vs_random": 0.017536754809128464,
      "p4_confident_cell_rate": 0.6740611518790781,
      "p4_unique_cells": 41.666666666666664
    },
    "384": {
      "p4_mean_future_gain": 0.014360213481784914,
      "p4_delta_vs_score": 0.01129771348178489,
      "p4_delta_vs_time": 0.0064852134817849,
      "p4_delta_vs_random": 0.010390432242797664,
      "p4_confident_cell_rate": 0.8312113177833576,
      "p4_unique_cells": 38.5
    }
  }
}
```

## Aggregate Selector Summary

```text
           selector  configs  mean_future_gain  mean_delta_vs_score  mean_delta_vs_time  mean_delta_vs_random  mean_basin_precision  mean_confident_cell_rate  mean_unique_cells
        P2_PSI_CONF       36          0.016908             0.015742            0.012945              0.014508              0.786295                  0.784100          40.027778
        P1_PSI_ONLY       36          0.016779             0.015613            0.012816              0.014379              0.779351                  0.759360          40.027778
P4_FRONTIER_POINTER       36          0.016697             0.015531            0.012734              0.014297              0.776747                  0.760662          40.027778
     P3_PSI_NOVELTY       36          0.016658             0.015492            0.012695              0.014258              0.775879                  0.754152          40.027778
S1_TIME_BUCKET_TOPN       36          0.003964             0.002797            0.000000              0.001563              0.234375                  0.951389          20.333333
     R0_RANDOM_CELL       36          0.002400             0.001234           -0.001563              0.000000              0.179680                  0.748943          40.027778
      S0_SCORE_TOPN       36          0.001167             0.000000           -0.002797             -0.001234              0.093750                  0.946181          21.250000
```

## Per-H Selector Summary

```text
  H            selector  configs  mean_future_gain  mean_delta_vs_score  mean_delta_vs_time  mean_delta_vs_random  mean_basin_precision  mean_confident_cell_rate  mean_unique_cells  mean_current_peak  mean_psi
128         P1_PSI_ONLY       12          0.017093             0.016859            0.014828              0.015000              0.892082                  0.775412          39.916667           0.023070  0.017156
128         P2_PSI_CONF       12          0.017187             0.016953            0.014921              0.015093              0.895988                  0.797547          39.916667           0.022812  0.017271
128      P3_PSI_NOVELTY       12          0.017046             0.016812            0.014781              0.014953              0.890780                  0.772808          39.916667           0.023074  0.017135
128 P4_FRONTIER_POINTER       12          0.017057             0.016822            0.014791              0.014963              0.892082                  0.776714          39.916667           0.023012  0.017165
128      R0_RANDOM_CELL       12          0.002094             0.001859           -0.000172              0.000000              0.160614                  0.764995          39.916667           0.053072 -0.000481
128       S0_SCORE_TOPN       12          0.000234             0.000000           -0.002031             -0.001859              0.000000                  0.960938          23.416667           0.056688 -0.003506
128 S1_TIME_BUCKET_TOPN       12          0.002266             0.002031            0.000000              0.000172              0.093750                  0.957031          23.083333           0.054000  0.002871
256         P1_PSI_ONLY       12          0.018817             0.018614            0.017067              0.017679              0.800092                  0.672759          41.666667           0.026065  0.020785
256         P2_PSI_CONF       12          0.019077             0.018874            0.017327              0.017939              0.815717                  0.711822          41.666667           0.026034  0.020954
256      P3_PSI_NOVELTY       12          0.018614             0.018410            0.016864              0.017476              0.796186                  0.666249          41.666667           0.026248  0.020675
256 P4_FRONTIER_POINTER       12          0.018675             0.018472            0.016925              0.017537              0.797488                  0.674061          41.666667           0.026149  0.020740
256      R0_RANDOM_CELL       12          0.001138             0.000935           -0.000612              0.000000              0.068316                  0.659738          41.666667           0.073117 -0.000605
256       S0_SCORE_TOPN       12          0.000203             0.000000           -0.001547             -0.000935              0.015625                  0.893229          25.166667           0.079562 -0.002499
256 S1_TIME_BUCKET_TOPN       12          0.001750             0.001547            0.000000              0.000612              0.125000                  0.906250          23.416667           0.075219  0.002185
384         P1_PSI_ONLY       12          0.014428             0.011365            0.006553              0.010458              0.645879                  0.829909          38.500000           0.021342  0.022903
384         P2_PSI_CONF       12          0.014460             0.011398            0.006585              0.010491              0.647181                  0.842930          38.500000           0.021135  0.022998
384      P3_PSI_NOVELTY       12          0.014315             0.011252            0.006440              0.010345              0.640671                  0.823399          38.500000           0.021622  0.022798
384 P4_FRONTIER_POINTER       12          0.014360             0.011298            0.006485              0.010390              0.640671                  0.831211          38.500000           0.021458  0.022872
384      R0_RANDOM_CELL       12          0.003970             0.000907           -0.003905              0.000000              0.310112                  0.822097          38.500000           0.066411  0.002855
384       S0_SCORE_TOPN       12          0.003063             0.000000           -0.004812             -0.000907              0.265625                  0.984375          15.166667           0.077828  0.000293
384 S1_TIME_BUCKET_TOPN       12          0.007875             0.004812            0.000000              0.003905              0.484375                  0.990885          14.500000           0.071609  0.005248
```

## Interpretation

- `P4_FRONTIER_POINTER` beats score/time/random baselines offline and is competitive with simpler Psi selectors.
- Numeric best selector is `P2_PSI_CONF`; P4 gap vs best is `-0.000211` mean future_gain. Treat novelty as optional until live-tested.
- Next step is D8.3 instrumentation-only, not live steering yet.

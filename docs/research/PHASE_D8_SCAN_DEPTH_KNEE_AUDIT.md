# Phase D8.1.1 Scan-Depth Knee Audit

Verdict: **CELL_SCAN_KNEE_FOUND**

## Summary

- D8.1.1 is offline-only and estimates how many samples per spherical cell are needed for reliable cell-level future_gain estimates.
- It uses deterministic bootstrap subsampling within each cell and never reads raw candidates or launches live Rust runs.
- A knee requires rank stability, top-cell overlap, and a large error drop from the one-sample estimate.
- This does not prove that spherical cells are predictive by themselves; it only estimates when an already-defined cell has been sampled deeply enough to trust its empirical mean.
- Knee thresholds are occupancy-conditional: cells with fewer observations than the knee remain low-confidence and should not drive a pointer alone.

## Coverage

```json
{
  "input": "output\\phase_d8_archive_psi_replay_20260427\\analysis\\panel_state_dataset.csv",
  "rows": 5840,
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
  "sample_sizes": [
    1,
    2,
    3,
    5,
    8,
    13,
    21,
    34,
    55
  ],
  "bootstrap_iters": 200,
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
  "rank_threshold": 0.7,
  "top_overlap_threshold": 0.5,
  "relative_error_threshold": 0.5,
  "min_eligible_cells": 8
}
```

## Decision

```json
{
  "required_h": 2,
  "found_h": 3,
  "weak_h": 3,
  "data_fail_h": 0,
  "by_h": {
    "128": {
      "configs": 12,
      "found_rate": 0.9166666666666666,
      "median_knee_sample_n": 8.0,
      "median_max_feasible_sample_n": 55.0,
      "best_rank_spearman": 0.9698733971724045,
      "best_top20_overlap": 0.99,
      "h_found": true,
      "h_weak": true,
      "h_data_fail": false
    },
    "256": {
      "configs": 12,
      "found_rate": 1.0,
      "median_knee_sample_n": 13.0,
      "median_max_feasible_sample_n": 55.0,
      "best_rank_spearman": 0.9504999999999998,
      "best_top20_overlap": 1.0,
      "h_found": true,
      "h_weak": true,
      "h_data_fail": false
    },
    "384": {
      "configs": 12,
      "found_rate": 1.0,
      "median_knee_sample_n": 5.0,
      "median_max_feasible_sample_n": 55.0,
      "best_rank_spearman": 0.9825342413984227,
      "best_top20_overlap": 1.0,
      "h_found": true,
      "h_weak": true,
      "h_data_fail": false
    }
  }
}
```

## Knee By Config

```text
  H  anchor_count  anchor_seed  knee_sample_n  knee_found  max_rank_spearman  max_top20_overlap  min_relative_error_vs_n1  max_feasible_sample_n
128            16           11            NaN       False           0.693576           0.595000                  0.114320                     55
128            16           23           21.0        True           0.891665           0.958333                  0.085388                     55
128            16           37           21.0        True           0.835975           0.753333                  0.103840                     55
128            32           11           21.0        True           0.875597           0.825000                  0.068635                     55
128            32           23           13.0        True           0.843475           0.943333                  0.077195                     55
128            32           37            8.0        True           0.903690           0.781667                  0.091718                     55
128            64           11           13.0        True           0.907277           0.867500                  0.090978                     34
128            64           23            8.0        True           0.944750           0.982500                  0.059990                     55
128            64           37            8.0        True           0.933333           0.805000                  0.079526                     55
128           128           11            5.0        True           0.969873           0.910000                  0.095128                     34
128           128           23            5.0        True           0.962714           0.990000                  0.112796                     34
128           128           37            5.0        True           0.833441           0.855000                  0.087922                     34
256            16           11           21.0        True           0.840545           0.826667                  0.107740                     55
256            16           23           34.0        True           0.864301           0.866667                  0.111375                     55
256            16           37           13.0        True           0.855525           0.788333                  0.114395                     55
256            32           11           13.0        True           0.905896           0.825000                  0.100538                     55
256            32           23           13.0        True           0.914769           0.910000                  0.092699                     55
256            32           37           13.0        True           0.864078           0.781250                  0.099964                     55
256            64           11           13.0        True           0.845949           0.730000                  0.108782                     55
256            64           23            8.0        True           0.918169           0.847500                  0.137158                     34
256            64           37            8.0        True           0.895833           0.825000                  0.110256                     55
256           128           11            5.0        True           0.950500           1.000000                  0.106419                     34
256           128           23            5.0        True           0.906120           1.000000                  0.154846                     34
256           128           37            5.0        True           0.933714           0.890000                  0.112694                     34
384            16           11            5.0        True           0.909643           0.822500                  0.156090                     55
384            16           23            8.0        True           0.925583           0.947500                  0.122219                     55
384            16           37            5.0        True           0.930121           0.676667                  0.109890                     55
384            32           11            5.0        True           0.979965           0.933333                  0.114601                     55
384            32           23            5.0        True           0.942610           0.845000                  0.133948                     55
384            32           37            5.0        True           0.958527           0.760000                  0.094387                     55
384            64           11            5.0        True           0.973427           1.000000                  0.151064                     55
384            64           23            5.0        True           0.976125           0.845000                  0.095164                     55
384            64           37            5.0        True           0.982534           0.848333                  0.093284                     55
384           128           11            5.0        True           0.967536           0.983750                  0.142839                     55
384           128           23            8.0        True           0.979863           0.871667                  0.120891                     55
384           128           37            5.0        True           0.977307           0.908333                  0.135791                     55
```

## Aggregate Scan Curve

```text
  H  sample_n  mean_rank  mean_error  mean_overlap  mean_eligible_cells
128         1   0.414948    0.004701      0.500271            45.833333
128         2   0.469118    0.003440      0.503314            41.583333
128         3   0.540989    0.002759      0.555419            40.000000
128         5   0.603196    0.002091      0.590998            35.250000
128         8   0.677644    0.001590      0.638579            31.000000
128        13   0.734491    0.001236      0.693712            24.916667
128        21   0.800949    0.000883      0.736951            19.916667
128        34   0.846254    0.000632      0.781458            13.416667
128        55   0.864072    0.000415      0.811212             8.083333
256         1   0.368629    0.005853      0.432618            50.500000
256         2   0.466739    0.004189      0.489653            47.250000
256         3   0.531036    0.003384      0.537331            44.416667
256         5   0.603114    0.002538      0.580578            39.916667
256         8   0.675404    0.001939      0.617342            33.666667
256        13   0.744049    0.001480      0.676951            27.250000
256        21   0.796460    0.001086      0.736681            21.250000
256        34   0.850685    0.000804      0.772986            14.750000
256        55   0.884929    0.000657      0.845764             9.000000
384         1   0.522253    0.008103      0.492262            41.666667
384         2   0.659021    0.005831      0.560673            37.250000
384         3   0.711846    0.004847      0.575911            34.250000
384         5   0.792808    0.003682      0.632728            31.250000
384         8   0.837872    0.002939      0.676007            27.333333
384        13   0.886040    0.002269      0.736412            23.416667
384        21   0.914387    0.001707      0.779556            20.250000
384        34   0.937650    0.001284      0.823611            16.583333
384        55   0.958603    0.000977      0.843472            12.166667
```

## Interpretation

- A practical scan-depth knee exists for most H regimes; use it as a confidence gate in future pointer replay, not as a live pointer proof.

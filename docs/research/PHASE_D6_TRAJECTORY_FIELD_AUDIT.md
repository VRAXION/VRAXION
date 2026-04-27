# Phase D6 Trajectory Field Audit

Verdict: **D7_FEATURE_POLICY**

## Artifact Coverage

- Roots used: ['phase_b1_horizon_ties_20260425', 'phase_b_full_20260424', 'phase_d1_activation_20260425', 'phase_d2_cross_h_activation_20260426', 'phase_d3_fine_k_20260426', 'phase_d3_klock_coarse_20260426', 'phase_d4_softness_20260426']
- Runs: 252
- Candidate rows represented by constructability summaries: 87,260,000
- Panel rows: 4940
- Operator rows: 2772

## Decision Signals

- early feature model seed-held-out: R2=0.355, Spearman=0.467; no-score R2=0.293, no-score Spearman=0.366
- trajectory alignment: success_cos=0.678, lift_vs_shuffle=0.065
- operator lift passes H-count=3

## Early Feature Model

```json
{
  "loocv_all_features": {
    "valid": true,
    "n": 252,
    "features": [
      "edges",
      "unique_predictions",
      "collision_rate",
      "f_active",
      "stable_rank",
      "kernel_rank",
      "separation_sp",
      "accept_rate_window",
      "main_peak_acc",
      "panel_probe_acc"
    ],
    "r2_loocv": 0.4770446952344023,
    "spearman": 0.5915653219196758,
    "pearson": 0.6911157028576834
  },
  "seed_group_cv_all_features": {
    "valid": true,
    "n": 252,
    "groups": 5,
    "group_col": "seed",
    "features": [
      "edges",
      "unique_predictions",
      "collision_rate",
      "f_active",
      "stable_rank",
      "kernel_rank",
      "separation_sp",
      "accept_rate_window",
      "main_peak_acc",
      "panel_probe_acc"
    ],
    "r2_group_cv": 0.3553141000120129,
    "spearman_group_cv": 0.4671019548368057,
    "pearson_group_cv": 0.6058558071313757
  },
  "loocv_no_score_features": {
    "valid": true,
    "n": 252,
    "features": [
      "edges",
      "unique_predictions",
      "collision_rate",
      "f_active",
      "stable_rank",
      "kernel_rank",
      "separation_sp",
      "accept_rate_window"
    ],
    "r2_loocv": 0.397732855625879,
    "spearman": 0.49392783640657534,
    "pearson": 0.6309956220121415
  },
  "seed_group_cv_no_score_features": {
    "valid": true,
    "n": 252,
    "groups": 5,
    "group_col": "seed",
    "features": [
      "edges",
      "unique_predictions",
      "collision_rate",
      "f_active",
      "stable_rank",
      "kernel_rank",
      "separation_sp",
      "accept_rate_window"
    ],
    "r2_group_cv": 0.29290567455241856,
    "spearman_group_cv": 0.3662356894685443,
    "pearson_group_cv": 0.5476018194101784
  },
  "valid": true
}
```

The decision uses seed-held-out validation. The no-score variant excludes `main_peak_acc` and `panel_probe_acc`.

Top early feature correlations:

| feature | Spearman | Pearson |
|---|---:|---:|
| main_peak_acc | 0.513 | 0.542 |
| separation_sp | 0.376 | 0.454 |
| panel_probe_acc | 0.282 | 0.310 |
| kernel_rank | 0.263 | 0.315 |
| f_active | 0.208 | 0.093 |
| accept_rate_window | 0.201 | 0.193 |
| stable_rank | 0.200 | 0.479 |
| unique_predictions | 0.022 | 0.038 |

## Trajectory Alignment

```json
{
  "valid": true,
  "n_vectors": 252,
  "success_threshold_peak_pct": 4.9,
  "success_pairwise_cosine": 0.6779619359055569,
  "failure_pairwise_cosine": 0.5505775156758895,
  "shuffle_success_cosine_mean": 0.6133672876590371,
  "success_lift_vs_shuffle": 0.06459464824651973
}
```

## Operator Field

```json
{
  "valid": true,
  "n_rows": 2772,
  "lift_threshold": 2.0,
  "h_lifts": [
    {
      "H": 128,
      "top3_bottom3_lift": 2.1838826219888956,
      "passes": true
    },
    {
      "H": 256,
      "top3_bottom3_lift": 3.758745092366411,
      "passes": true
    },
    {
      "H": 384,
      "top3_bottom3_lift": 7.135986472780216,
      "passes": true
    }
  ],
  "pass_h_count": 3
}
```

Top operator usefulness by H:

| H | operator | usefulness | V_raw | M_pos | R_neg |
|---:|---|---:|---:|---:|---:|
| 128 | theta | 0.002017 | 0.3428 | 0.005922 | 0.01114 |
| 128 | rewire | 0.001952 | 0.3511 | 0.005538 | 0.007308 |
| 128 | loop3 | 0.001911 | 0.3289 | 0.005819 | 0.008214 |
| 128 | loop2 | 0.001906 | 0.3402 | 0.00558 | 0.007313 |
| 128 | reverse | 0.001828 | 0.329 | 0.005542 | 0.007273 |
| 256 | theta | 0.0006511 | 0.1511 | 0.004477 | 0.02306 |
| 256 | loop3 | 0.0005183 | 0.1325 | 0.003983 | 0.0151 |
| 256 | channel | 0.0004847 | 0.1162 | 0.004342 | 0.02065 |
| 256 | rewire | 0.0004503 | 0.1197 | 0.00387 | 0.01419 |
| 256 | loop2 | 0.0004415 | 0.1164 | 0.00386 | 0.01415 |
| 384 | theta | 0.0001111 | 0.02514 | 0.004713 | 0.04368 |
| 384 | channel | 7.587e-05 | 0.01745 | 0.004321 | 0.04322 |
| 384 | loop3 | 5.554e-05 | 0.01408 | 0.005375 | 0.03746 |
| 384 | rewire | 4.15e-05 | 0.01115 | 0.005621 | 0.0365 |
| 384 | reverse | 4.1e-05 | 0.01071 | 0.004807 | 0.03597 |

## Interpretation

- This is not a raw graph-space gradient claim.
- A feature-policy verdict means a learned proposal conditioned on panel-state is worth testing.
- An operator-bandit verdict means adaptive operator weighting is the safer D7 first step.
- A no-signal verdict means richer mutation-target instrumentation should precede adaptive search.
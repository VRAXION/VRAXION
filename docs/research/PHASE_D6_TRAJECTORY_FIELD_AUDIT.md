# Phase D6.1 Trajectory Field Falsification Audit

Verdict: **D7_OPERATOR_BANDIT**

## Artifact Coverage

- Roots used: ['phase_b1_horizon_ties_20260425', 'phase_b_full_20260424', 'phase_d1_activation_20260425', 'phase_d2_cross_h_activation_20260426', 'phase_d3_fine_k_20260426', 'phase_d3_klock_coarse_20260426', 'phase_d4_softness_20260426']
- Runs: 252
- Candidate rows represented by constructability summaries: 87,260,000
- Panel rows: 4940
- Operator rows: 2772
- Raw candidate CSV rows were not re-scanned; D6.1 uses constructability summaries that represent those rows.

## Decision Signals

- early feature model seed-held-out: R2=0.355, Spearman=0.467; no-score R2=0.293, no-score Spearman=0.366; no-score-no-accept R2=0.290, Spearman=0.351; residual no-score Spearman=0.145; negative-control max |Spearman|=0.287
- feature-policy gate: FAIL (no_score=True, no_score_no_accept=True, controls_clean=False)
- source_group_cv_all_features: R2=0.465, Spearman=0.597
- H_group_cv_all_features: R2=-0.556, Spearman=0.365
- phase_group_cv_all_features: R2=0.413, Spearman=0.565
- arm_group_cv_all_features: R2=0.472, Spearman=0.589
- trajectory alignment: success_cos=0.678, lift_vs_shuffle=0.066, lift_vs_H_source_shuffle=0.028
- operator lift passes H-count=3

## Interpretation

- D6.1 uses stricter adversarial controls than the original D6 audit: fold-local scaling/imputation, global run IDs, source/H/phase/arm holdouts, no-score/no-accept-rate features, residual targets, stratified shuffles, and weighted operator usefulness.
- The feature-state model still contains real signal, including after removing score fields and accept-rate. However, the direct feature-policy gate is not clean enough: one negative control remains above the pre-set margin, and H-held-out value prediction is weak.
- The robust live-ready signal is operator-level: weighted top-vs-bottom operator usefulness remains above the 2x threshold for all tested H values and survives leave-one-run influence checks.
- Therefore the safe next live experiment is D7.1 operator-bandit/adaptive operator weighting. A full feature-conditioned proposal remains a D7.2 candidate after the bandit baseline and stronger controls.

## Early Feature Model

```json
{
  "loocv_all_features": {
    "valid": true,
    "n": 252,
    "features": [
      "accept_rate_window",
      "collision_rate",
      "edges",
      "f_active",
      "kernel_rank",
      "main_peak_acc",
      "panel_probe_acc",
      "separation_sp",
      "stable_rank",
      "unique_predictions"
    ],
    "r2_loocv": 0.47704985576950676,
    "spearman": 0.5915651001205108,
    "pearson": 0.6911228809583907
  },
  "seed_group_cv_all_features": {
    "valid": true,
    "n": 252,
    "groups": 5,
    "group_col": "seed",
    "features": [
      "accept_rate_window",
      "collision_rate",
      "edges",
      "f_active",
      "kernel_rank",
      "main_peak_acc",
      "panel_probe_acc",
      "separation_sp",
      "stable_rank",
      "unique_predictions"
    ],
    "r2_group_cv": 0.3554930614473699,
    "spearman_group_cv": 0.46740147276122396,
    "pearson_group_cv": 0.6063178224454637
  },
  "source_group_cv_all_features": {
    "valid": true,
    "n": 252,
    "groups": 7,
    "group_col": "source",
    "features": [
      "accept_rate_window",
      "collision_rate",
      "edges",
      "f_active",
      "kernel_rank",
      "main_peak_acc",
      "panel_probe_acc",
      "separation_sp",
      "stable_rank",
      "unique_predictions"
    ],
    "r2_group_cv": 0.4648484083461475,
    "spearman_group_cv": 0.5968176411812347,
    "pearson_group_cv": 0.6829511769538784
  },
  "H_group_cv_all_features": {
    "valid": true,
    "n": 252,
    "groups": 3,
    "group_col": "H",
    "features": [
      "accept_rate_window",
      "collision_rate",
      "edges",
      "f_active",
      "kernel_rank",
      "main_peak_acc",
      "panel_probe_acc",
      "separation_sp",
      "stable_rank",
      "unique_predictions"
    ],
    "r2_group_cv": -0.5557381329527826,
    "spearman_group_cv": 0.3654095804061575,
    "pearson_group_cv": 0.32199814673211746
  },
  "phase_group_cv_all_features": {
    "valid": true,
    "n": 227,
    "groups": 6,
    "group_col": "phase",
    "features": [
      "accept_rate_window",
      "collision_rate",
      "edges",
      "f_active",
      "kernel_rank",
      "main_peak_acc",
      "panel_probe_acc",
      "separation_sp",
      "stable_rank",
      "unique_predictions"
    ],
    "r2_group_cv": 0.41312616494529153,
    "spearman_group_cv": 0.5654698140864843,
    "pearson_group_cv": 0.6467082455750164
  },
  "arm_group_cv_all_features": {
    "valid": true,
    "n": 252,
    "groups": 42,
    "group_col": "arm",
    "features": [
      "accept_rate_window",
      "collision_rate",
      "edges",
      "f_active",
      "kernel_rank",
      "main_peak_acc",
      "panel_probe_acc",
      "separation_sp",
      "stable_rank",
      "unique_predictions"
    ],
    "r2_group_cv": 0.47179306086897077,
    "spearman_group_cv": 0.5888320370647406,
    "pearson_group_cv": 0.6874476394069595
  },
  "loocv_no_score_features": {
    "valid": true,
    "n": 252,
    "features": [
      "accept_rate_window",
      "collision_rate",
      "edges",
      "f_active",
      "kernel_rank",
      "separation_sp",
      "stable_rank",
      "unique_predictions"
    ],
    "r2_loocv": 0.39774080915017296,
    "spearman": 0.4939276512150151,
    "pearson": 0.6310058940754746
  },
  "seed_group_cv_no_score_features": {
    "valid": true,
    "n": 252,
    "groups": 5,
    "group_col": "seed",
    "features": [
      "accept_rate_window",
      "collision_rate",
      "edges",
      "f_active",
      "kernel_rank",
      "separation_sp",
      "stable_rank",
      "unique_predictions"
    ],
    "r2_group_cv": 0.2933514451035062,
    "spearman_group_cv": 0.365787840371478,
    "pearson_group_cv": 0.5483090586694178
  },
  "seed_group_cv_no_score_no_accept_features": {
    "valid": true,
    "n": 252,
    "groups": 5,
    "group_col": "seed",
    "features": [
      "collision_rate",
      "edges",
      "f_active",
      "kernel_rank",
      "separation_sp",
      "stable_rank",
      "unique_predictions"
    ],
    "r2_group_cv": 0.28950725476014794,
    "spearman_group_cv": 0.3512060543214755,
    "pearson_group_cv": 0.5420596999980875
  },
  "seed_group_cv_residual_no_score_features": {
    "valid": true,
    "n": 252,
    "groups": 5,
    "group_col": "seed",
    "features": [
      "accept_rate_window",
      "collision_rate",
      "edges",
      "f_active",
      "kernel_rank",
      "separation_sp",
      "stable_rank",
      "unique_predictions"
    ],
    "r2_group_cv": 0.015847321491174515,
    "spearman_group_cv": 0.145107929177313,
    "pearson_group_cv": 0.24021530604161484
  },
  "negative_controls": {
    "target_shuffle_within_H_source": {
      "valid": true,
      "n": 252,
      "groups": 5,
      "group_col": "seed",
      "features": [
        "accept_rate_window",
        "collision_rate",
        "edges",
        "f_active",
        "kernel_rank",
        "separation_sp",
        "stable_rank",
        "unique_predictions"
      ],
      "r2_group_cv": 0.0028806464865285264,
      "spearman_group_cv": 0.20899333747496265,
      "pearson_group_cv": 0.20110400368946285
    },
    "feature_shuffle": {
      "valid": true,
      "n": 252,
      "groups": 5,
      "group_col": "seed",
      "features": [
        "accept_rate_window",
        "collision_rate",
        "edges",
        "f_active",
        "kernel_rank",
        "separation_sp",
        "stable_rank",
        "unique_predictions"
      ],
      "r2_group_cv": -0.19913722595418792,
      "spearman_group_cv": -0.28670459295844675,
      "pearson_group_cv": -0.30054987417827705
    }
  },
  "valid": true
}
```

The decision uses train-fold-only scaling/imputation. The no-score variant excludes `main_peak_acc` and `panel_probe_acc`; the stricter no-score-no-accept variant also excludes `accept_rate_window`.

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
  "shuffle_success_cosine_mean": 0.6120989168250685,
  "success_lift_vs_shuffle": 0.06586301908048842,
  "shuffle_within_H_source_cosine_mean": 0.6500314006554915,
  "success_lift_vs_H_source_shuffle": 0.02793053525006539
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
      "top3_bottom3_lift": 2.163418051189273,
      "top3_median_lift": 1.0864899823712963,
      "bootstrap_ci_low": 2.1446230296570716,
      "bootstrap_ci_high": 2.1852796218102544,
      "min_leave_one_run_lift": 2.160211012809515,
      "passes": true
    },
    {
      "H": 256,
      "top3_bottom3_lift": 3.6580623931584992,
      "top3_median_lift": 1.2803633986133218,
      "bootstrap_ci_low": 3.5681230038742,
      "bootstrap_ci_high": 3.7612121074422205,
      "min_leave_one_run_lift": 3.6377942990105883,
      "passes": true
    },
    {
      "H": 384,
      "top3_bottom3_lift": 6.980075580021538,
      "top3_median_lift": 1.9773408959215593,
      "bootstrap_ci_low": 6.665675706369378,
      "bootstrap_ci_high": 7.3340104024766015,
      "min_leave_one_run_lift": 6.899634468522624,
      "passes": true
    }
  ],
  "pass_h_count": 3
}
```

Top operator usefulness by H:

| H | operator | weighted usefulness | mean usefulness | V_raw | M_pos | R_neg |
|---:|---|---:|---:|---:|---:|---:|
| 128 | theta | 0.002156 | 0.002017 | 0.3428 | 0.005922 | 0.01114 |
| 128 | rewire | 0.002118 | 0.001952 | 0.3511 | 0.005538 | 0.007308 |
| 128 | loop3 | 0.002066 | 0.001911 | 0.3289 | 0.005819 | 0.008214 |
| 128 | loop2 | 0.00206 | 0.001906 | 0.3402 | 0.00558 | 0.007313 |
| 128 | reverse | 0.001978 | 0.001828 | 0.329 | 0.005542 | 0.007273 |
| 256 | theta | 0.0007156 | 0.0006511 | 0.1511 | 0.004477 | 0.02306 |
| 256 | loop3 | 0.0005866 | 0.0005183 | 0.1325 | 0.003983 | 0.0151 |
| 256 | channel | 0.0005431 | 0.0004847 | 0.1162 | 0.004342 | 0.02065 |
| 256 | rewire | 0.000509 | 0.0004503 | 0.1197 | 0.00387 | 0.01419 |
| 256 | loop2 | 0.0005031 | 0.0004415 | 0.1164 | 0.00386 | 0.01415 |
| 384 | theta | 0.0001305 | 0.0001111 | 0.02514 | 0.004713 | 0.04368 |
| 384 | channel | 9.298e-05 | 7.587e-05 | 0.01745 | 0.004321 | 0.04322 |
| 384 | loop3 | 6.637e-05 | 5.554e-05 | 0.01408 | 0.005375 | 0.03746 |
| 384 | rewire | 5.02e-05 | 4.15e-05 | 0.01115 | 0.005621 | 0.0365 |
| 384 | loop2 | 4.939e-05 | 4.002e-05 | 0.01066 | 0.003926 | 0.03725 |

## Interpretation

- This is not a raw graph-space gradient claim.
- A feature-policy verdict means a learned proposal conditioned on panel-state is worth testing.
- An operator-bandit verdict means adaptive operator weighting is the safer D7 first step.
- A no-signal verdict means richer mutation-target instrumentation should precede adaptive search.
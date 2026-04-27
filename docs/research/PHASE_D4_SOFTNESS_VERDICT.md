# Phase D4 Softness Verdict

Input root: `output\phase_d4_softness_20260426`

Verdict: **SAF_STRICT_LOCK**

## Artifact Integrity

- Runs: 45
- Candidate rows: 21,600,000
- Checkpoints: 45
- Panel summaries: 45
- Panel timeseries files: 45

## Decision Notes

- No zero_p arm beats strict by >= 0.50pp on any H.

## H-Level Winners

| H | strict_peak_mean_pct | best_policy | best_delta_vs_strict_pp |
|---|---|---|---|
| 128 | 4.62 | strict | 0 |
| 256 | 6.1 | strict | 0 |
| 384 | 5.5 | strict | 0 |

## Arm Statistics

| H | jackpot | policy | n | peak_acc_pct_mean | peak_acc_pct_std | final_acc_pct_mean | peak_final_gap_pp_mean | accept_rate_pct_mean | alive_frac_mean_mean | C_K_window_ratio_mean | collapse_count |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 9 | strict | 5 | 4.62 | 0.9834 | 3.38 | 1.24 | 78.41 | 0.7156 | 0.0001339 | 0 |
| 128 | 9 | zero_p_0.3 | 5 | 4.02 | 1.119 | 2.84 | 1.18 | 82.88 | 0.7332 | 9.055e-05 | 0 |
| 128 | 9 | zero_p_1.0 | 5 | 4.44 | 1.053 | 3.26 | 1.18 | 95.88 | 0.7052 | 0.0001265 | 0 |
| 256 | 18 | strict | 5 | 6.1 | 0.728 | 3.74 | 2.36 | 62.76 | 0.4504 | 2.103e-05 | 0 |
| 256 | 18 | zero_p_0.3 | 5 | 5.76 | 1.596 | 4 | 1.76 | 77.13 | 0.4856 | 2.025e-05 | 0 |
| 256 | 18 | zero_p_1.0 | 5 | 5.46 | 1.963 | 2.86 | 2.6 | 99.95 | 0.4928 | 1.624e-05 | 0 |
| 384 | 9 | strict | 5 | 5.5 | 1.475 | 3.68 | 1.82 | 17.85 | 0.4619 | 2.686e-06 | 0 |
| 384 | 9 | zero_p_0.3 | 5 | 4 | 2.649 | 2.8 | 1.2 | 36.54 | 0.4378 | 1.455e-06 | 1 |
| 384 | 9 | zero_p_1.0 | 5 | 5.3 | 0.9874 | 3.22 | 2.08 | 99.67 | 0.5333 | 2.544e-06 | 0 |

## Interpretation

- `SAF_STRICT_LOCK` means SAF v1 can remain `SAF(K(H), tau=0, s=0)` for this substrate.
- `SAF_S_H_TABLE` means softness is H-dependent and SAF v1 needs an `s(H)` table.
- `SAF_UNSTABLE` means softness may improve mean peak but is not deployable without variance/collapse controls.
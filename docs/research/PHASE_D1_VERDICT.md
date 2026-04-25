# Phase D1 Verdict: Search-Activation Sweep

D1 tests the aperture as a two-axis search activation: jackpot pool size `K` and neutral-step valve `zero_p`.

## Integrity

- Runs: `45`
- Candidate rows: `7800000`
- Checkpoints: `45`
- Panel summaries: `45`
- Panel timeseries files: `45`
- Expected seeds per arm: `5`

## Winner

- Best mean peak: `K=9`, `strict` = `5.50%`
- Mean final: `3.68%`
- Mean accept rate: `17.85%`

## Decision Notes

- Measured winner by mean `peak_acc`: `K=9`, `strict`.
- Decision rule: neutral drift is harmful or unnecessary in the winning aperture regime.
- K=1 is weaker than the best grid point, so some jackpot discovery remains useful.

## Arm Stats

| K | policy | n | peak mean | peak std | final mean | accept mean | alive mean | C_K mean |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | strict | 5 | 1.92 | 1.28 | 1.44 | 0.85 | 0.243 | 3.616e-07 |
| 1 | zero_p_0.3 | 5 | 3.44 | 2.26 | 2.40 | 25.30 | 0.310 | 3.322e-07 |
| 1 | zero_p_1.0 | 5 | 4.02 | 2.52 | 2.70 | 69.92 | 0.405 | 3.291e-07 |
| 3 | strict | 5 | 4.24 | 2.75 | 2.68 | 4.00 | 0.466 | 1.328e-06 |
| 3 | zero_p_0.3 | 5 | 3.12 | 2.24 | 2.08 | 30.30 | 0.469 | 5.612e-07 |
| 3 | zero_p_1.0 | 5 | 3.78 | 1.70 | 2.86 | 91.08 | 0.504 | 9.904e-07 |
| 9 | strict | 5 | 5.50 | 1.47 | 3.68 | 17.85 | 0.462 | 2.260e-06 |
| 9 | zero_p_0.3 | 5 | 4.00 | 2.65 | 2.80 | 36.54 | 0.438 | 1.236e-06 |
| 9 | zero_p_1.0 | 5 | 5.30 | 0.99 | 3.22 | 99.67 | 0.533 | 2.096e-06 |

## Comparisons

Welch tests are diagnostic at n=5 and not used alone as the decision criterion.

| axis | comparison | delta pp | p | d |
|---|---|---:|---:|---:|
| policy_within_K | K1:zero_p_0.3_vs_strict | 1.52 | 0.2361 | 0.83 |
| policy_within_K | K1:zero_p_1.0_vs_strict | 2.10 | 0.1478 | 1.05 |
| policy_within_K | K1:zero_p_1.0_vs_zero_p_0.3 | 0.58 | 0.7113 | 0.24 |
| policy_within_K | K3:zero_p_0.3_vs_strict | -1.12 | 0.5014 | -0.45 |
| policy_within_K | K3:zero_p_1.0_vs_strict | -0.46 | 0.7601 | -0.20 |
| policy_within_K | K3:zero_p_1.0_vs_zero_p_0.3 | 0.66 | 0.6152 | 0.33 |
| policy_within_K | K9:zero_p_0.3_vs_strict | -1.50 | 0.3092 | -0.70 |
| policy_within_K | K9:zero_p_1.0_vs_strict | -0.20 | 0.8083 | -0.16 |
| policy_within_K | K9:zero_p_1.0_vs_zero_p_0.3 | 1.30 | 0.3501 | 0.65 |
| K_within_policy | strict:K3_vs_K1 | 2.32 | 0.1413 | 1.08 |
| K_within_policy | strict:K9_vs_K3 | 1.26 | 0.4008 | 0.57 |
| K_within_policy | strict:K9_vs_K1 | 3.58 | 0.003605 | 2.59 |
| K_within_policy | zero_p_0.3:K3_vs_K1 | -0.32 | 0.8278 | -0.14 |
| K_within_policy | zero_p_0.3:K9_vs_K3 | 0.88 | 0.5868 | 0.36 |
| K_within_policy | zero_p_0.3:K9_vs_K1 | 0.56 | 0.7285 | 0.23 |
| K_within_policy | zero_p_1.0:K3_vs_K1 | -0.24 | 0.8646 | -0.11 |
| K_within_policy | zero_p_1.0:K9_vs_K3 | 1.52 | 0.1309 | 1.09 |
| K_within_policy | zero_p_1.0:K9_vs_K1 | 1.28 | 0.3362 | 0.67 |

## Outputs

- Machine summary: `output\phase_d1_activation_20260425\analysis\phase_d1_verdict.json`
- Group stats: `output\phase_d1_activation_20260425\analysis\phase_d1_group_stats.csv`
- Comparisons: `output\phase_d1_activation_20260425\analysis\phase_d1_comparisons.csv`
- Figures: `output\phase_d1_activation_20260425\analysis\figures`
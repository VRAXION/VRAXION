# Phase D2 Verdict: Cross-H Search Activation

D2 tests whether the D1 H=384 activation result generalizes to smaller H.

## Integrity

- Runs: `60`
- Candidate rows: `10400000`
- Checkpoints: `60`
- Panel summaries: `60`
- Panel timeseries files: `60`

## Winners

| H | winner K | winner policy | peak mean | peak std |
|---:|---:|---|---:|---:|
| 128 | 9 | strict | 4.62 | 0.98 |
| 256 | 9 | strict | 5.28 | 1.79 |

## Arm Stats

| H | K | policy | n | peak mean | peak std | final mean | accept mean | alive mean | C_K mean |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | strict | 5 | 4.00 | 1.12 | 2.72 | 23.83 | 0.705 | 8.681e-05 |
| 128 | 1 | zero_p_1.0 | 5 | 3.74 | 1.08 | 2.30 | 42.52 | 0.677 | 7.151e-05 |
| 128 | 3 | strict | 5 | 4.02 | 1.13 | 2.72 | 48.25 | 0.734 | 7.880e-05 |
| 128 | 3 | zero_p_1.0 | 5 | 4.34 | 1.13 | 2.70 | 73.41 | 0.704 | 1.409e-04 |
| 128 | 9 | strict | 5 | 4.62 | 0.98 | 3.38 | 78.41 | 0.716 | 1.354e-04 |
| 128 | 9 | zero_p_1.0 | 5 | 4.44 | 1.05 | 3.26 | 95.88 | 0.705 | 1.320e-04 |
| 256 | 1 | strict | 5 | 3.48 | 1.57 | 2.74 | 6.13 | 0.442 | 4.314e-06 |
| 256 | 1 | zero_p_1.0 | 5 | 3.72 | 1.61 | 2.14 | 59.45 | 0.441 | 6.657e-06 |
| 256 | 3 | strict | 5 | 4.98 | 1.99 | 3.56 | 20.71 | 0.499 | 1.308e-05 |
| 256 | 3 | zero_p_1.0 | 5 | 4.96 | 0.72 | 4.12 | 86.46 | 0.394 | 1.761e-05 |
| 256 | 9 | strict | 5 | 5.28 | 1.79 | 3.36 | 49.62 | 0.472 | 2.131e-05 |
| 256 | 9 | zero_p_1.0 | 5 | 4.54 | 1.40 | 3.08 | 98.95 | 0.493 | 2.270e-05 |

## Comparisons

Welch tests are diagnostic at n=5 and not used alone as the decision criterion.

| axis | comparison | delta pp | p | d |
|---|---|---:|---:|---:|
| policy_within_H_K | H128_K1:zero_p_1.0_vs_strict | -0.26 | 0.7185 | -0.24 |
| policy_within_H_K | H128_K3:zero_p_1.0_vs_strict | 0.32 | 0.666 | 0.28 |
| policy_within_H_K | H128_K9:zero_p_1.0_vs_strict | -0.18 | 0.787 | -0.18 |
| K_within_H_policy | H128_strict:K3_vs_K1 | 0.02 | 0.9782 | 0.02 |
| K_within_H_policy | H128_strict:K9_vs_K3 | 0.60 | 0.3961 | 0.57 |
| K_within_H_policy | H128_strict:K9_vs_K1 | 0.62 | 0.38 | 0.59 |
| K_within_H_policy | H128_zero_p_1.0:K3_vs_K1 | 0.60 | 0.4164 | 0.54 |
| K_within_H_policy | H128_zero_p_1.0:K9_vs_K3 | 0.10 | 0.8886 | 0.09 |
| K_within_H_policy | H128_zero_p_1.0:K9_vs_K1 | 0.70 | 0.3299 | 0.66 |
| policy_within_H_K | H256_K1:zero_p_1.0_vs_strict | 0.24 | 0.8172 | 0.15 |
| policy_within_H_K | H256_K3:zero_p_1.0_vs_strict | -0.02 | 0.984 | -0.01 |
| policy_within_H_K | H256_K9:zero_p_1.0_vs_strict | -0.74 | 0.4885 | -0.46 |
| K_within_H_policy | H256_strict:K3_vs_K1 | 1.50 | 0.2247 | 0.84 |
| K_within_H_policy | H256_strict:K9_vs_K3 | 0.30 | 0.8086 | 0.16 |
| K_within_H_policy | H256_strict:K9_vs_K1 | 1.80 | 0.13 | 1.07 |
| K_within_H_policy | H256_zero_p_1.0:K3_vs_K1 | 1.24 | 0.1703 | 1.00 |
| K_within_H_policy | H256_zero_p_1.0:K9_vs_K3 | -0.42 | 0.5732 | -0.38 |
| K_within_H_policy | H256_zero_p_1.0:K9_vs_K1 | 0.82 | 0.415 | 0.54 |

## Outputs

- Machine summary: `output\phase_d2_cross_h_activation_20260426\analysis\phase_d2_cross_h_verdict.json`
- Group stats: `output\phase_d2_cross_h_activation_20260426\analysis\phase_d2_group_stats.csv`
- Comparisons: `output\phase_d2_cross_h_activation_20260426\analysis\phase_d2_comparisons.csv`
- Figures: `output\phase_d2_cross_h_activation_20260426\analysis\figures`
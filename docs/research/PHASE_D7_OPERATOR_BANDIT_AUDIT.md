# Phase D7.1 Operator Bandit Audit

Verdict: **D7_NEEDS_ARCHIVE_OR_FEATURE_POLICY**

## Summary

- D7.1 tests only operator sampling weights over locked SAF v1.
- Fixed: mutual_inhibition, strict gate, K(H), horizon, seeds, candidate budget.
- Treatments are compared by paired H/seed deltas against D7_BASELINE.

## Arm Stats

```text
  H             arm  n  peak_mean_pct  peak_median_pct  peak_std_pct  final_mean_pct  accept_mean_pct  rows_to_peak_median  wall_mean_s
128     D7_BASELINE  5           4.62              4.8      0.983362            3.38          78.4060             306000.0    1654.2866
128   D7_PRIOR_EWMA  5           4.80              5.1      1.009950            3.28          77.8870             324000.0    1633.0584
128 D7_STATIC_PRIOR  5           4.92              5.6      0.993479            3.02          79.8710             216000.0    1644.1744
256     D7_BASELINE  5           6.10              6.2      0.728011            3.74          62.7640             432000.0    5874.9826
256   D7_PRIOR_EWMA  5           5.44              4.8      1.266096            3.78          70.6595             396000.0    5945.9556
256 D7_STATIC_PRIOR  5           5.26              5.6      0.554977            3.54          72.4120             576000.0    6013.7282
384     D7_BASELINE  5           5.50              6.1      1.474788            3.68          17.8540             288000.0    5338.2160
384   D7_PRIOR_EWMA  5           4.26              5.7      2.748272            3.12          20.5445             198000.0    5376.0672
384 D7_STATIC_PRIOR  5           5.34              5.8      1.262141            3.78          28.1050             342000.0    5418.1954
```

## Paired Delta Stats

```text
            arm  n  median_delta_peak_pct  mean_delta_peak_pct  median_delta_rows_to_peak
  D7_PRIOR_EWMA 15                    0.0            -0.573333                   -36000.0
D7_STATIC_PRIOR 15                   -0.3            -0.233333                        0.0
```

## Decision

```json
{
  "D7_STATIC_PRIOR": {
    "median_delta_peak_pct": -0.29999999999999893,
    "mean_delta_peak_pct": -0.23333333333333334,
    "h_mean_delta_peak_pct": {
      "128": 0.30000000000000016,
      "256": -0.8400000000000002,
      "384": -0.15999999999999998
    },
    "h_improved_count": 1,
    "material_regression": true,
    "median_delta_rows_to_peak": 0.0,
    "median_delta_accept_rate_pct": 7.842500000000001,
    "entropy_ok": true,
    "entropy_notes": [
      {
        "H": 128,
        "entropy_median": 0.9339213801565914,
        "entropy_ratio_vs_baseline": 0.9888755891502058
      },
      {
        "H": 256,
        "entropy_median": 0.9400277809092024,
        "entropy_ratio_vs_baseline": 0.9953413053979834
      },
      {
        "H": 384,
        "entropy_median": 0.926455054496482,
        "entropy_ratio_vs_baseline": 0.9809699267006642
      }
    ],
    "accept_only_fail": true,
    "min_leave_one_seed_median_delta": -0.40000000000000036,
    "lock": false
  },
  "D7_PRIOR_EWMA": {
    "median_delta_peak_pct": 0.0,
    "mean_delta_peak_pct": -0.5733333333333337,
    "h_mean_delta_peak_pct": {
      "128": 0.18,
      "256": -0.6600000000000005,
      "384": -1.2400000000000002
    },
    "h_improved_count": 1,
    "material_regression": true,
    "median_delta_rows_to_peak": -36000.0,
    "median_delta_accept_rate_pct": 4.767500000000005,
    "entropy_ok": true,
    "entropy_notes": [
      {
        "H": 128,
        "entropy_median": 0.953600651666779,
        "entropy_ratio_vs_baseline": 1.0097128369337618
      },
      {
        "H": 256,
        "entropy_median": 0.9520874911460088,
        "entropy_ratio_vs_baseline": 1.0081106383619671
      },
      {
        "H": 384,
        "entropy_median": 0.9250838453915557,
        "entropy_ratio_vs_baseline": 0.9795180323118081
      }
    ],
    "accept_only_fail": true,
    "min_leave_one_seed_median_delta": -0.24999999999999978,
    "lock": false
  }
}
```

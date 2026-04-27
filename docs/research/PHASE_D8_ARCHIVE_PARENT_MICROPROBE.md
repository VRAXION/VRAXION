# Phase D8.4a Archive-Parent Microprobe

Verdict: **D8_ARCHIVE_PARENT_REGRESSION**

## Scope

- Live search microprobe over SAF v1.
- K(H), strict gate, operator schedule, horizon, and fixture are unchanged.
- Only the parent source changes: current-best, random archive parent, or score archive parent.
- This is not the full Ψ live controller; P2_PSI_CONF remains offline until model export/import is instrumented.

## Decision

```json
{
  "root": "output\\phase_d8_archive_parent_microprobe",
  "arm_stats": [
    {
      "arm": "D8A_RANDOM_ARCHIVE_PARENT",
      "overall_median_peak_delta_pp": -0.5000000000000001,
      "positive_h_count": 1,
      "worst_h_peak_delta_pp": -1.4
    },
    {
      "arm": "D8A_SCORE_ARCHIVE_PARENT",
      "overall_median_peak_delta_pp": -0.10000000000000009,
      "positive_h_count": 0,
      "worst_h_peak_delta_pp": -1.4
    }
  ]
}
```

## Per-H Arm Summary

```text
  H                       arm  n  peak_mean_pct  peak_median_pct  final_mean_pct  accept_mean_pct  wall_mean_s
128          D8A_CURRENT_BEST  1            1.6              1.6             1.6             69.2       39.811
128 D8A_RANDOM_ARCHIVE_PARENT  1            1.1              1.1             1.1             66.0       34.858
128  D8A_SCORE_ARCHIVE_PARENT  1            1.5              1.5             1.5             68.0       38.830
256          D8A_CURRENT_BEST  1            2.0              2.0             2.0             25.9      127.939
256 D8A_RANDOM_ARCHIVE_PARENT  1            5.4              5.4             5.4             22.0      126.319
256  D8A_SCORE_ARCHIVE_PARENT  1            2.0              2.0             2.0             21.0      117.423
384          D8A_CURRENT_BEST  1            1.8              1.8             1.8              6.0      117.561
384 D8A_RANDOM_ARCHIVE_PARENT  1            0.4              0.4             0.4              6.2      104.994
384  D8A_SCORE_ARCHIVE_PARENT  1            0.4              0.4             0.4              7.0      103.997
```

## Paired Peak Deltas vs Current-Best

```text
  H  seed                       arm  peak_delta_pp  final_delta_pp  accept_delta_pp
128    42 D8A_RANDOM_ARCHIVE_PARENT           -0.5            -0.5             -3.2
256    42 D8A_RANDOM_ARCHIVE_PARENT            3.4             3.4             -3.9
384    42 D8A_RANDOM_ARCHIVE_PARENT           -1.4            -1.4              0.2
128    42  D8A_SCORE_ARCHIVE_PARENT           -0.1            -0.1             -1.2
256    42  D8A_SCORE_ARCHIVE_PARENT            0.0             0.0             -4.9
384    42  D8A_SCORE_ARCHIVE_PARENT           -1.4            -1.4              1.0
```

## Infrastructure Audit

```text
                                                                          run_dir                       arm   H  seed archive_parent_policy  candidate_rows  expected_candidate_rows  state_log_exists  archive_parent_log_exists  panel_exists  state_rows  parent_rows  panel_rows  restored_count  selected_ids_valid  parent_links_valid  pass
         output\phase_d8_archive_parent_microprobe\H_128\D8A_CURRENT_BEST\seed_42          D8A_CURRENT_BEST 128    42          current-best            9000                     9000              True                       True          True          10           10          10               0                True                True  True
         output\phase_d8_archive_parent_microprobe\H_256\D8A_CURRENT_BEST\seed_42          D8A_CURRENT_BEST 256    42          current-best           18000                    18000              True                       True          True          10           10          10               0                True                True  True
         output\phase_d8_archive_parent_microprobe\H_384\D8A_CURRENT_BEST\seed_42          D8A_CURRENT_BEST 384    42          current-best            9000                     9000              True                       True          True          10           10          10               0                True                True  True
output\phase_d8_archive_parent_microprobe\H_128\D8A_RANDOM_ARCHIVE_PARENT\seed_42 D8A_RANDOM_ARCHIVE_PARENT 128    42        random-archive            9000                     9000              True                       True          True          10           10          10               9                True                True  True
output\phase_d8_archive_parent_microprobe\H_256\D8A_RANDOM_ARCHIVE_PARENT\seed_42 D8A_RANDOM_ARCHIVE_PARENT 256    42        random-archive           18000                    18000              True                       True          True          10           10          10               9                True                True  True
output\phase_d8_archive_parent_microprobe\H_384\D8A_RANDOM_ARCHIVE_PARENT\seed_42 D8A_RANDOM_ARCHIVE_PARENT 384    42        random-archive            9000                     9000              True                       True          True          10           10          10               9                True                True  True
 output\phase_d8_archive_parent_microprobe\H_128\D8A_SCORE_ARCHIVE_PARENT\seed_42  D8A_SCORE_ARCHIVE_PARENT 128    42         score-archive            9000                     9000              True                       True          True          10           10          10               9                True                True  True
 output\phase_d8_archive_parent_microprobe\H_256\D8A_SCORE_ARCHIVE_PARENT\seed_42  D8A_SCORE_ARCHIVE_PARENT 256    42         score-archive           18000                    18000              True                       True          True          10           10          10               9                True                True  True
 output\phase_d8_archive_parent_microprobe\H_384\D8A_SCORE_ARCHIVE_PARENT\seed_42  D8A_SCORE_ARCHIVE_PARENT 384    42         score-archive            9000                     9000              True                       True          True          10           10          10               9                True                True  True
```

## Interpretation

- Archive parent switching regressed at least one H materially; do not expand without changing the selector.

# Phase D8.4 Archive-Parent Microprobe

Verdict: **D8_ARCHIVE_PARENT_REGRESSION**

## Scope

- Live search microprobe over SAF v1.
- K(H), strict gate, operator schedule, horizon, and fixture are unchanged.
- Only the parent source changes; acceptance and mutation semantics remain fixed.
- This run tests live `P2_PSI_CONF = psi_pred * scan_depth_confidence` parent selection.
- The P2 model is exported from historical D8 panel-state data and used only for parent selection, not acceptance.

## Decision

```json
{
  "root": "output\\phase_d8_p2_microprobe2",
  "arm_stats": [
    {
      "arm": "D8B_P2_PSI_CONF_LOW_DUTY",
      "overall_median_peak_delta_pp": 0.0,
      "positive_h_count": 0,
      "worst_h_peak_delta_pp": -1.4
    },
    {
      "arm": "D8B_P2_PSI_CONF_MED_DUTY",
      "overall_median_peak_delta_pp": 0.0,
      "positive_h_count": 1,
      "worst_h_peak_delta_pp": -0.7999999999999998
    }
  ]
}
```

## Per-H Arm Summary

```text
  H                      arm  n  peak_mean_pct  peak_median_pct  final_mean_pct  accept_mean_pct  wall_mean_s
128         D8B_CURRENT_BEST  1            1.6              1.6             1.6             69.2       35.686
128 D8B_P2_PSI_CONF_LOW_DUTY  1            1.6              1.6             1.6             69.2       36.081
128 D8B_P2_PSI_CONF_MED_DUTY  1            2.4              2.4             2.4             69.7       34.593
256         D8B_CURRENT_BEST  1            2.0              2.0             2.0             25.9      106.489
256 D8B_P2_PSI_CONF_LOW_DUTY  1            2.0              2.0             2.0             25.9      122.468
256 D8B_P2_PSI_CONF_MED_DUTY  1            2.0              2.0             2.0             24.3      114.518
384         D8B_CURRENT_BEST  1            1.8              1.8             1.8              6.0      109.851
384 D8B_P2_PSI_CONF_LOW_DUTY  1            0.4              0.4             0.4              6.9      112.664
384 D8B_P2_PSI_CONF_MED_DUTY  1            1.0              1.0             1.0              5.3      109.194
```

## Paired Peak Deltas vs Current-Best

```text
  H  seed                      arm  peak_delta_pp  final_delta_pp  accept_delta_pp
128    42 D8B_P2_PSI_CONF_LOW_DUTY            0.0             0.0              0.0
256    42 D8B_P2_PSI_CONF_LOW_DUTY            0.0             0.0              0.0
384    42 D8B_P2_PSI_CONF_LOW_DUTY           -1.4            -1.4              0.9
128    42 D8B_P2_PSI_CONF_MED_DUTY            0.8             0.8              0.5
256    42 D8B_P2_PSI_CONF_MED_DUTY            0.0             0.0             -1.6
384    42 D8B_P2_PSI_CONF_MED_DUTY           -0.8            -0.8             -0.7
```

## Infrastructure Audit

```text
                                                              run_dir                      arm   H  seed archive_parent_policy  candidate_rows  expected_candidate_rows  state_log_exists  archive_parent_log_exists  panel_exists  state_rows  parent_rows  panel_rows  restored_count  selected_ids_valid  parent_links_valid  pass
        output\phase_d8_p2_microprobe2\H_128\D8B_CURRENT_BEST\seed_42         D8B_CURRENT_BEST 128    42          current-best            9000                     9000              True                       True          True          10           10          10               0                True                True  True
        output\phase_d8_p2_microprobe2\H_256\D8B_CURRENT_BEST\seed_42         D8B_CURRENT_BEST 256    42          current-best           18000                    18000              True                       True          True          10           10          10               0                True                True  True
        output\phase_d8_p2_microprobe2\H_384\D8B_CURRENT_BEST\seed_42         D8B_CURRENT_BEST 384    42          current-best            9000                     9000              True                       True          True          10           10          10               0                True                True  True
output\phase_d8_p2_microprobe2\H_128\D8B_P2_PSI_CONF_LOW_DUTY\seed_42 D8B_P2_PSI_CONF_LOW_DUTY 128    42           p2-psi-conf            9000                     9000              True                       True          True          10           10          10               0                True                True  True
output\phase_d8_p2_microprobe2\H_256\D8B_P2_PSI_CONF_LOW_DUTY\seed_42 D8B_P2_PSI_CONF_LOW_DUTY 256    42           p2-psi-conf           18000                    18000              True                       True          True          10           10          10               0                True                True  True
output\phase_d8_p2_microprobe2\H_384\D8B_P2_PSI_CONF_LOW_DUTY\seed_42 D8B_P2_PSI_CONF_LOW_DUTY 384    42           p2-psi-conf            9000                     9000              True                       True          True          10           10          10               2                True                True  True
output\phase_d8_p2_microprobe2\H_128\D8B_P2_PSI_CONF_MED_DUTY\seed_42 D8B_P2_PSI_CONF_MED_DUTY 128    42           p2-psi-conf            9000                     9000              True                       True          True          10           10          10               1                True                True  True
output\phase_d8_p2_microprobe2\H_256\D8B_P2_PSI_CONF_MED_DUTY\seed_42 D8B_P2_PSI_CONF_MED_DUTY 256    42           p2-psi-conf           18000                    18000              True                       True          True          10           10          10               1                True                True  True
output\phase_d8_p2_microprobe2\H_384\D8B_P2_PSI_CONF_MED_DUTY\seed_42 D8B_P2_PSI_CONF_MED_DUTY 384    42           p2-psi-conf            9000                     9000              True                       True          True          10           10          10               3                True                True  True
```

## Interpretation

- Archive parent switching regressed at least one H materially; do not expand without changing the selector.

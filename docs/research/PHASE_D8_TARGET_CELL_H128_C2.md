# Phase D8 Target Cell H128 C2

Verdict: `D8_TARGET_CELL_DETAIL_READY`

This is a focused observer-only follow-up for the atlas cell the dashboard marked as high-priority: `H=128`, frozen runtime `archive_cell_id=2`.

No live search behavior changed. SAF v1, `K(H)`, strict acceptance, and the operator schedule stayed unchanged. The run only collected more live panel states and joined them back to the frozen D8 atlas cell ID.

## Result

| group | n | seeds | confidence | mean_psi | mean_future_gain | basin_precision |
|---|---:|---:|---:|---:|---:|---:|
| baseline atlas | 3 | 2 | 0.375 | 0.0182 | 0.0340 | 1.000 |
| previous observer | 1 | 1 | 0.125 | 0.0081 | 0.0060 | 1.000 |
| H128 C2 targeted observer | 12 | 4 | 1.000 | 0.0071 | 0.0074 | 0.333 |
| combined live | 13 | 4 | 1.000 | 0.0072 | 0.0073 | 0.385 |
| total baseline + live | 16 | 5 | 1.000 | 0.0092 | 0.0123 | 0.500 |

## Interpretation

The scan succeeded at the sampling objective: C2 moved from `3/8` samples to `16/8` total samples, so it is no longer underfilled.

The quality signal did not hold. The old atlas view made C2 look strong because it had only three optimistic historical samples. The new live samples reduced the combined estimate:

- `mean_psi`: `0.0182 -> 0.0092`
- `mean_future_gain`: `0.0340 -> 0.0123`
- `basin_precision`: `1.000 -> 0.500`

Practical read: H128/C2 was worth scanning, but after enough samples it should not be promoted as a safe branch-trial target. It is now better treated as a de-risked / downgraded cell unless a later controlled branch test produces stronger evidence.

## Artifacts

- Detail HTML: `output/phase_d8_target_cell_H128_C2_20260428/target_cell_detail.html`
- Summary CSV: `output/phase_d8_target_cell_H128_C2_20260428/target_cell_summary.csv`
- Sample rows: `output/phase_d8_target_cell_H128_C2_20260428/target_cell_samples.csv`
- Updated atlas from augmented dataset: `output/phase_d8_cell_atlas_h128_c2_scan_20260428/cell_atlas.html`

Note: the updated full atlas recomputes robust scaling and cell assignment. For exact “same cell” tracking, use the target detail report, which follows the frozen runtime `archive_cell_id=2`.

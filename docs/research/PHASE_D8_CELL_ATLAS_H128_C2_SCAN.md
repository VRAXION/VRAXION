# Phase D8.6 Cell Atlas / Basin Map Dashboard

Verdict: `D8_CELL_ATLAS_READY`

D8.6 is analysis/visualization only. It does not launch a Rust run, change SAF, change K(H), switch archive parents, or alter acceptance.

## What The Atlas Shows

- `Constellation Graph`: cells are connected by k-nearest neighbors in the original high-D behavior-feature space.
- `PCA/SVD Atlas`: a deterministic 2D projection for overview only.
- `Command Grid`: a priority dashboard, not a spatial map.

The 2D atlas is a visualization/projection, not exact high-D geometry.

## Geometry Contract

Cell geometry uses only behavior-core features:

`stable_rank`, `kernel_rank`, `separation_sp`, `collision_rate`, `f_active`, `unique_predictions`, `edges`, `accept_rate_window`.

Score/time fields are displayed as metrics but are not used for behavior-space cell assignment.

## Outputs

- HTML dashboard: `output\phase_d8_cell_atlas_h128_c2_scan_20260428\cell_atlas.html`
- Cell table: `output\phase_d8_cell_atlas_h128_c2_scan_20260428\analysis\cell_table.csv`
- Cell neighbors: `output\phase_d8_cell_atlas_h128_c2_scan_20260428\analysis\cell_neighbors.csv`
- Sample-more candidates: `output\phase_d8_cell_atlas_h128_c2_scan_20260428\analysis\sample_more_candidates.csv`
- Split candidates: `output\phase_d8_cell_atlas_h128_c2_scan_20260428\analysis\split_candidates.csv`
- Branch-trial candidates: `output\phase_d8_cell_atlas_h128_c2_scan_20260428\analysis\branch_trial_candidates.csv`
- Retire candidates: `output\phase_d8_cell_atlas_h128_c2_scan_20260428\analysis\retire_candidates.csv`

## Coverage

- Input rows: `5920`
- Atlas cells: `158`
- Neighbor rows: `948`
- H values: `[128, 256, 384]`
- Anchor config: `64` anchors, seed `11`
- Knees: `{128: 8, 256: 13, 384: 5}`

## Candidate Counts

- `sample_more_candidates`: `17` rows, by H `{128: 6, 256: 9, 384: 2}`
- `split_candidates`: `14` rows, by H `{128: 5, 256: 1, 384: 8}`
- `branch_trial_candidates`: `60` rows, by H `{128: 20, 256: 20, 384: 20}`
- `retire_candidates`: `11` rows, by H `{128: 3, 256: 4, 384: 4}`

## Interpretation

Use this dashboard as an operator-facing atlas: choose cells to sample more, split, retire, or branch-test. It does not prove live improvement by itself; it makes the next live tests less blind.

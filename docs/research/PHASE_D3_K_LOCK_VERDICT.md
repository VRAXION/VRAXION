# Phase D3 Verdict: Search Aperture Function K-Lock

D3 tests the K axis of the Search Aperture Function under strict acceptance (`tau=0`, `s=0`).

## Integrity

- D3 runs: `27`
- D3 candidate rows: `12960000`
- D3 checkpoints: `27`
- D3 panel summaries: `27`
- D3 panel timeseries files: `27`

## Verdict

- Result: **K(H) TABLE**
- Lock margin: `0.50pp` mean peak over K=9
- The best K appears H-dependent under the lock margin; use a provisional K(H) table and fine sweep the affected region.
- H=256: K=18 beats K=9 by 1.07pp.

## Seed-Matched K Grid

Primary comparison uses seeds `42,1042,2042` for every K, because D3 new K values are n=3.

| H | K | n | peak mean | peak std | final mean | accept mean | alive mean | C_K mean | wall/candidate ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 3 | 4.50 | 1.25 | 3.27 | 23.25 | 0.711 | 8.031e-05 | 12.263 |
| 128 | 3 | 3 | 4.60 | 1.10 | 3.13 | 47.84 | 0.762 | 8.232e-05 | 6.627 |
| 128 | 5 | 3 | 4.67 | 1.17 | 3.37 | 63.01 | 0.736 | 1.315e-04 | 5.315 |
| 128 | 9 | 3 | 5.27 | 0.40 | 3.70 | 78.08 | 0.737 | 1.504e-04 | 4.686 |
| 128 | 13 | 3 | 5.20 | 0.44 | 3.37 | 85.03 | 0.728 | 1.325e-04 | 4.363 |
| 128 | 18 | 3 | 5.23 | 0.32 | 3.50 | 90.03 | 0.797 | 1.193e-04 | 4.292 |
| 256 | 1 | 3 | 3.83 | 2.11 | 2.90 | 5.54 | 0.418 | 3.575e-06 | 22.148 |
| 256 | 3 | 3 | 5.53 | 1.47 | 4.40 | 18.30 | 0.527 | 1.158e-05 | 12.552 |
| 256 | 5 | 3 | 3.67 | 0.74 | 2.27 | 34.29 | 0.510 | 1.447e-05 | 10.533 |
| 256 | 9 | 3 | 5.53 | 1.71 | 3.67 | 48.75 | 0.497 | 1.576e-05 | 8.717 |
| 256 | 13 | 3 | 5.67 | 1.57 | 3.07 | 58.00 | 0.511 | 1.616e-05 | 9.021 |
| 256 | 18 | 3 | 6.60 | 0.35 | 4.07 | 62.59 | 0.462 | 2.078e-05 | 8.812 |
| 384 | 1 | 3 | 1.40 | 1.21 | 0.77 | 0.65 | 0.207 | 2.916e-07 | 34.943 |
| 384 | 3 | 3 | 3.67 | 1.91 | 2.40 | 3.35 | 0.511 | 1.079e-06 | 19.898 |
| 384 | 5 | 3 | 2.03 | 2.41 | 1.63 | 3.13 | 0.386 | 4.490e-07 | 16.215 |
| 384 | 9 | 3 | 5.93 | 0.76 | 4.27 | 16.53 | 0.460 | 2.288e-06 | 18.612 |
| 384 | 13 | 3 | 3.50 | 2.72 | 2.20 | 15.08 | 0.309 | 1.954e-06 | 13.959 |
| 384 | 18 | 3 | 4.50 | 4.19 | 3.70 | 15.48 | 0.364 | 1.611e-06 | 13.255 |

## Winner Table

| H | best K | best peak mean | K9 peak mean | delta vs K9 |
|---:|---:|---:|---:|---:|
| 128 | 9 | 5.27 | 5.27 | 0.00 |
| 256 | 18 | 6.60 | 5.53 | 1.07 |
| 384 | 9 | 5.93 | 5.93 | 0.00 |

## Full Anchor Context

Existing K={1,3,9} anchors are also retained at full n=5 where available.

| H | K | n | peak mean | peak std |
|---:|---:|---:|---:|---:|
| 128 | 1 | 5 | 4.00 | 1.12 |
| 128 | 3 | 5 | 4.02 | 1.13 |
| 128 | 9 | 5 | 4.62 | 0.98 |
| 256 | 1 | 5 | 3.48 | 1.57 |
| 256 | 3 | 5 | 4.98 | 1.99 |
| 256 | 9 | 5 | 5.28 | 1.79 |
| 384 | 1 | 5 | 1.92 | 1.28 |
| 384 | 3 | 5 | 4.24 | 2.75 |
| 384 | 9 | 5 | 5.50 | 1.47 |

## Outputs

- Machine summary: `output\phase_d3_klock_coarse_20260426\analysis\phase_d3_klock_verdict.json`
- Seed-matched stats: `output\phase_d3_klock_coarse_20260426\analysis\phase_d3_seed_matched_stats.csv`
- Full anchor stats: `output\phase_d3_klock_coarse_20260426\analysis\phase_d3_full_anchor_stats.csv`
- Figure: `output\phase_d3_klock_coarse_20260426\analysis\figures\phase_d3_k_curve.png`
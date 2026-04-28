# Phase D8.7 Cell Scan Delta

Verdict: `D8_CELL_SCAN_DELTA_READY`

D8.7 is an observer-only spin: SAF v1, K(H), strict acceptance, and operator schedule stay unchanged. The P2 model is used only to log archive cell IDs and Psi values.

This report reflects the targeted H128 follow-up scan used to fill the previously under-sampled `H=128`, frozen runtime `archive_cell_id=2` region. See `PHASE_D8_TARGET_CELL_H128_C2.md` for the exact before/after target-cell detail.

## Coverage

- Runs: `5`
- Live panel states: `80`
- Visited cells: `28`
- Opened new cells: `8`
- Reinforced cells: `10`
- Cooled cells: `0`

## Top Reinforced / Opened Cells

| H | cell | new_n | base_n | conf_after | new_psi | new_gain | opened | reinforced | cooled |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| 128 | 45 | 1 | 1 | 0.250 | 0.0221 | 0.0410 | False | True | False |
| 128 | 51 | 12 | 56 | 1.000 | 0.0216 | 0.0309 | False | True | False |
| 128 | 18 | 1 | 11 | 1.000 | 0.0142 | 0.0200 | False | True | False |
| 128 | 22 | 3 | 72 | 1.000 | 0.0195 | 0.0177 | False | True | False |
| 128 | 38 | 3 | 15 | 1.000 | 0.0118 | 0.0147 | False | True | False |
| 128 | 5 | 1 | 36 | 1.000 | 0.0073 | 0.0140 | False | True | False |
| 128 | 6 | 2 | 21 | 1.000 | 0.0181 | 0.0105 | False | True | False |
| 128 | 20 | 2 | 80 | 1.000 | 0.0132 | 0.0100 | False | True | False |
| 128 | 12 | 4 | 1 | 0.625 | 0.0099 | 0.0083 | False | True | False |
| 128 | 39 | 4 | 4 | 1.000 | 0.0097 | 0.0078 | False | True | False |
| 128 | 46 | 1 | 0 | 0.125 | 0.0178 | 0.0160 | True | False | False |
| 128 | 11 | 2 | 0 | 0.250 | 0.0169 | 0.0130 | True | False | False |

## Interpretation

Opened cells are cells visited by the live spin that were absent from the baseline atlas. Reinforced cells are existing atlas cells where the new live panels still show positive future-gain and Psi support. Cooled cells are previously high-Psi cells where this spin produced little follow-on gain.

This does not prove live branch improvement. It tells us how the live trajectory populated the atlas before we choose split/sample/branch actions.

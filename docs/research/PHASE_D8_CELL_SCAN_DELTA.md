# Phase D8.7 Cell Scan Delta

Verdict: `D8_CELL_SCAN_DELTA_READY`

D8.7 is an observer-only spin: SAF v1, K(H), strict acceptance, and operator schedule stay unchanged. The P2 model is used only to log archive cell IDs and Psi values.

## Coverage

- Runs: `3`
- Live panel states: `24`
- Visited cells: `11`
- Opened new cells: `4`
- Reinforced cells: `5`
- Cooled cells: `0`

## Top Reinforced / Opened Cells

| H | cell | new_n | base_n | conf_after | new_psi | new_gain | opened | reinforced | cooled |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| 128 | 51 | 2 | 56 | 1.000 | 0.0232 | 0.0270 | False | True | False |
| 128 | 45 | 1 | 1 | 0.250 | 0.0200 | 0.0270 | False | True | False |
| 256 | 8 | 8 | 36 | 1.000 | 0.0209 | 0.0124 | False | True | False |
| 128 | 18 | 1 | 11 | 1.000 | 0.0114 | 0.0060 | False | True | False |
| 128 | 12 | 1 | 1 | 0.250 | 0.0098 | 0.0060 | False | True | False |
| 128 | 8 | 1 | 0 | 0.125 | 0.0071 | 0.0060 | True | False | False |
| 384 | 2 | 6 | 0 | 1.000 | 0.0181 | 0.0055 | True | False | False |
| 384 | 29 | 1 | 0 | 0.200 | 0.0162 | 0.0030 | True | False | False |
| 128 | 0 | 1 | 0 | 0.125 | 0.0037 | 0.0000 | True | False | False |
| 128 | 2 | 1 | 3 | 0.500 | 0.0081 | 0.0060 | False | False | False |
| 384 | 20 | 1 | 230 | 1.000 | 0.0178 | 0.0030 | False | False | False |

## Interpretation

Opened cells are cells visited by the live spin that were absent from the baseline atlas. Reinforced cells are existing atlas cells where the new live panels still show positive future-gain and Psi support. Cooled cells are previously high-Psi cells where this spin produced little follow-on gain.

This does not prove live branch improvement. It tells us how the live trajectory populated the atlas before we choose split/sample/branch actions.

# Phase D3.1 Verdict: H=256 Fine K-Lock

D3.1 resolves the H=256 K-axis region opened by D3 under strict SAF (`tau=0`, `s=0`).

## Integrity

- D3.1 runs: `20`
- D3.1 candidate rows: `15600000`
- D3.1 checkpoints: `20`
- D3.1 panel summaries: `20`
- D3.1 panel timeseries files: `20`

## Verdict

- Result: **H256_K18_LOCK**
- Lock margin: `0.50pp` mean peak over K=18
- No D3.1 K beats K=18 by the lock margin (0.50pp).

## Fine K Grid

| K | n | peak mean | peak std | final mean | accept mean | collapse | C_K mean | wall/candidate ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15 | 5 | 5.40 | 0.94 | 2.80 | 64.47 | 0 | 1.461e-05 | 11.220 |
| 18 | 5 | 6.10 | 0.73 | 3.74 | 62.76 | 0 | 1.721e-05 | 10.458 |
| 21 | 5 | 5.98 | 1.23 | 4.30 | 71.65 | 0 | 1.804e-05 | 8.796 |
| 24 | 5 | 5.78 | 0.90 | 4.20 | 68.03 | 0 | 1.552e-05 | 7.633 |

## H=256 Context

| source | K | n | peak mean | peak std | C_K mean |
|---|---:|---:|---:|---:|---:|
| D2 | 1 | 5 | 3.48 | 1.57 | 4.314e-06 |
| D2 | 3 | 5 | 4.98 | 1.99 | 1.308e-05 |
| D2 | 9 | 5 | 5.28 | 1.79 | 2.131e-05 |
| D3 | 5 | 3 | 3.67 | 0.74 | 1.447e-05 |
| D3 | 13 | 3 | 5.67 | 1.57 | 1.616e-05 |
| D3 | 18 | 3 | 6.60 | 0.35 | 2.078e-05 |
| D3.1 | 15 | 5 | 5.40 | 0.94 | 1.461e-05 |
| D3.1 | 18 | 5 | 6.10 | 0.73 | 1.721e-05 |
| D3.1 | 21 | 5 | 5.98 | 1.23 | 1.804e-05 |
| D3.1 | 24 | 5 | 5.78 | 0.90 | 1.552e-05 |

## Outputs

- Machine summary: `output\phase_d3_fine_k_20260426\analysis\phase_d3_fine_k_verdict.json`
- Fine stats: `output\phase_d3_fine_k_20260426\analysis\phase_d3_fine_k_stats.csv`
- Formula stats: `output\phase_d3_fine_k_20260426\analysis\saf_k_formula_stats.csv`
- Figure: `output\phase_d3_fine_k_20260426\analysis\figures\phase_d3_fine_k_h256_curve.png`
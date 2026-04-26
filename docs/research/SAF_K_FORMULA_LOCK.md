# SAF K Formula Lock

This report keeps `tau=0` and `s=0` fixed and summarizes the empirical sampling aperture rule.

The tested null model is:

```text
P_hit(K,H) = 1 - (1 - p_pos(H))^K
```

`P_hit` is diagnostic only. The lock rule also uses peak accuracy, C_K, variance, collapse count, and cost.

## Provisional K(H)

| H | K_lock | status |
|---:|---:|---|
| 128 | 9 | provisional_lock |
| 256 | 18 | locked |
| 384 | 9 | provisional_lock |

The lock table uses the seed-matched D3 verdict for H=128/H=384 and the D3.1 fine verdict for H=256. The diagnostics table below merges broader context and is not used by itself as a winner table.

## Formula Diagnostics

| H | K | peak mean | C_K mean | V_raw | P_hit model | near best | collapse |
|---:|---:|---:|---:|---:|---:|---|---:|
| 128 | 1 | 4.00 | 8.681e-05 | 0.2987 | 0.299 | false | 0 |
| 128 | 3 | 4.02 | 7.880e-05 | 0.2962 | 0.655 | false | 0 |
| 128 | 5 | 4.67 | 1.315e-04 | 0.3033 | 0.830 | false | 0 |
| 128 | 9 | 4.62 | 1.354e-04 | 0.3026 | 0.959 | false | 0 |
| 128 | 13 | 5.20 | 1.325e-04 | 0.3004 | 0.990 | true | 0 |
| 128 | 18 | 5.23 | 1.193e-04 | 0.2910 | 0.998 | true | 0 |
| 256 | 1 | 3.48 | 4.314e-06 | 0.0347 | 0.035 | false | 0 |
| 256 | 3 | 4.98 | 1.308e-05 | 0.0756 | 0.100 | false | 0 |
| 256 | 5 | 3.67 | 1.447e-05 | 0.0930 | 0.162 | false | 0 |
| 256 | 9 | 5.28 | 2.131e-05 | 0.1069 | 0.272 | false | 0 |
| 256 | 13 | 5.67 | 1.616e-05 | 0.1016 | 0.368 | true | 0 |
| 256 | 15 | 5.40 | 1.461e-05 | 0.1134 | 0.411 | false | 0 |
| 256 | 18 | 6.10 | 1.721e-05 | 0.0889 | 0.470 | true | 0 |
| 256 | 21 | 5.98 | 1.804e-05 | 0.1126 | 0.523 | true | 0 |
| 256 | 24 | 5.78 | 1.552e-05 | 0.0821 | 0.571 | true | 0 |
| 384 | 1 | 1.92 | 3.616e-07 | 0.0040 | 0.004 | false | 1 |
| 384 | 3 | 4.24 | 1.328e-06 | 0.0067 | 0.012 | false | 0 |
| 384 | 5 | 2.03 | 4.490e-07 | 0.0033 | 0.020 | false | 1 |
| 384 | 9 | 5.50 | 2.260e-06 | 0.0153 | 0.035 | true | 0 |
| 384 | 13 | 3.50 | 1.954e-06 | 0.0106 | 0.050 | false | 1 |
| 384 | 18 | 4.50 | 1.611e-06 | 0.0078 | 0.069 | false | 1 |

## Interpretation

- `K` is not a universal constant in the current grower substrate.
- `P_hit` explains the sampling funnel pressure, but it is not sufficient by itself: H=384 shows that larger K can increase variance/collapse risk.
- The practical SAF lock remains empirical: choose the smallest near-best K after penalizing instability and cost.
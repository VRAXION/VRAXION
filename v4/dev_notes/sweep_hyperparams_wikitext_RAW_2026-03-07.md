# Hyperparameter Permutation Sweep — WikiText-2 Raw Results

> **WARNING: RAW SWEEP OUTPUT — FOR PROCESSING, NOT FINAL.**
> **Needs consolidation + review before any conclusions are drawn.**
> **Generated: 2026-03-07**

## Sweep Design

**Goal**: Push BPC lower toward perplexity ~100 by systematically varying under-explored factors.

**Data**: WikiText-2-raw-v1 (10.5 MB train, byte-level)
**Steps**: 500 per config | **Batch**: 32 x 256 | **Device**: CPU (16 cores)

### 8 Configurations

| Cfg | Label | H | R | SD | M | Embed | LR | What it tests |
|-----|-------|---|---|-----|-----|---------|------|---------------|
| A | base_small | 256 | 1 | 64 | 512 | learned | 1e-3 | Baseline small |
| B | small_R2 | 256 | 2 | 64 | 512 | learned | 1e-3 | R effect (small model) |
| C | wide | 512 | 1 | 64 | 512 | learned | 1e-3 | H effect |
| D | wide_R2 | 512 | 2 | 64 | 512 | learned | 1e-3 | H+R combo |
| E | wide_R2_bit | 512 | 2 | 64 | 512 | bitlift | 1e-3 | Embed comparison |
| F | wide_R2_slow | 512 | 2 | 64 | 512 | learned | 5e-4 | LR effect |
| G | wide_R3 | 512 | 3 | 64 | 512 | learned | 1e-3 | More R |
| H | small_R2_slow | 256 | 2 | 64 | 512 | learned | 5e-4 | Small+slow |

### Factor Analysis Design

Orthogonal comparisons:
- **R effect**: A vs B (H=256), C vs D (H=512), D vs G (R=2 vs R=3)
- **H effect**: A vs C (R=1), B vs D (R=2)
- **Embed effect**: D vs E (learned vs bitlift)
- **LR effect**: D vs F (H=512,R=2), B vs H (H=256,R=2)

---

## Raw Results

### Summary Table (ranked by Final BPC)

| # | Cfg | Label | Params | Final Loss | Final BPC | Best BPC | Final Acc | Best Acc | GNorm | Time |
|---|-----|-------|--------|------------|-----------|----------|-----------|----------|-------|------|
| 1 | D | wide_R2 | 284,196 | 2.0531 | 2.962 | 2.819 | 0.427 | 0.454 | 0.3 | 929s |
| 2 | C | wide | 284,196 | 2.0533 | 2.962 | 2.822 | 0.427 | 0.452 | 0.3 | 917s |
| 3 | G | wide_R3 | 284,196 | 2.0546 | 2.964 | 2.823 | 0.427 | 0.453 | 0.3 | 942s |
| 4 | A | base_small | 158,756 | 2.1560 | 3.110 | 2.970 | 0.398 | 0.424 | 0.2 | 1346s |
| 5 | B | small_R2 | 158,756 | 2.1576 | 3.113 | 2.973 | 0.397 | 0.422 | 0.2 | 1386s |
| 6 | F | wide_R2_slow | 284,196 | 2.2550 | 3.253 | 3.091 | 0.383 | 0.410 | 0.3 | 927s |
| 7 | H | small_R2_slow | 158,756 | 2.3687 | 3.417 | 3.256 | 0.352 | 0.378 | 0.2 | 1358s |
| 8 | E | wide_R2_bit | 157,732 | 2.5297 | 3.650 | 3.489 | 0.308 | 0.337 | 3.5 | 1104s |

### Factor Analysis

| Factor | Comparison | BPC Delta | Winner | Magnitude |
|--------|------------|-----------|--------|-----------|
| **H (hidden_dim)** | H=256→512 at R=1 | +0.148 | **H=512** | STRONG |
| **H (hidden_dim)** | H=256→512 at R=2 | +0.151 | **H=512** | STRONG |
| **Embed encoding** | learned vs bitlift | +0.688 | **learned** | VERY STRONG |
| **LR** | 1e-3 vs 5e-4 at H=512 | -0.291 | **1e-3** | STRONG |
| **LR** | 1e-3 vs 5e-4 at H=256 | -0.304 | **1e-3** | STRONG |
| R (attn radius) | R=1→R=2 at H=256 | -0.002 | R=1 | NEGLIGIBLE |
| R (attn radius) | R=1→R=2 at H=512 | +0.000 | tie | NEGLIGIBLE |
| R (attn radius) | R=2→R=3 at H=512 | -0.002 | R=2 | NEGLIGIBLE |

### Key Findings

1. **hidden_dim is the dominant lever**: H=256→512 gives ~0.15 BPC improvement consistently
2. **R (attention radius) does NOT matter on WikiText**: R=1, R=2, R=3 all within noise (0.002 BPC)
   - This contradicts synthetic task results where R=2 was a clear winner
   - Likely because WikiText has local correlations that don't benefit from wider ring attention
3. **learned embedding massively outperforms bitlift**: +0.688 BPC gap, 3.5x higher gradient norms
   - YAML should be updated from `bitlift` to `learned`
4. **Higher LR (1e-3) is better than 5e-4**: ~0.3 BPC gap at 500 steps
   - The model is still in rapid learning phase at 500 steps
   - Lower LR may catch up with more steps but wastes compute

### Raw Per-Config Training Curves

**Config A (H=256, R=1, learned, lr=1e-3) — 158,756 params**
```
Step   Loss    BPC    Acc    GNorm  Speed
  1    5.6398  8.137  0.002  1.8    0.3 step/s
 50    4.7152  6.803  0.133  1.6    0.4 step/s
100    3.9000  5.626  0.179  1.0    0.4 step/s
150    2.8971  4.180  0.258  0.4    0.4 step/s
200    2.6062  3.760  0.308  0.3    0.4 step/s
250    2.4386  3.518  0.338  0.2    0.4 step/s
300    2.3360  3.370  0.358  0.2    0.4 step/s
350    2.2628  3.264  0.374  0.2    0.4 step/s
400    2.1993  3.173  0.388  0.2    0.4 step/s
450    2.1499  3.102  0.399  0.2    0.4 step/s
500    2.1127  3.048  0.407  0.2    0.4 step/s
FINAL: loss=2.1560  bpc=3.110  acc=0.398  best_bpc=2.970  (1346s)
```

**Config B (H=256, R=2, learned, lr=1e-3) — 158,756 params**
```
Step   Loss    BPC    Acc    GNorm  Speed
  1    5.6398  8.137  0.002  1.8    0.4 step/s
 50    4.7151  6.802  0.133  1.6    0.4 step/s
100    3.9002  5.627  0.178  1.0    0.4 step/s
150    2.8975  4.180  0.257  0.4    0.4 step/s
200    2.6058  3.759  0.308  0.3    0.4 step/s
250    2.4379  3.517  0.338  0.2    0.4 step/s
300    2.3359  3.370  0.358  0.2    0.4 step/s
350    2.2638  3.266  0.374  0.2    0.4 step/s
400    2.2007  3.175  0.388  0.2    0.4 step/s
450    2.1514  3.104  0.398  0.2    0.4 step/s
500    2.1146  3.051  0.407  0.2    0.4 step/s
FINAL: loss=2.1576  bpc=3.113  acc=0.397  best_bpc=2.973  (1386s)
```

**Config C (H=512, R=1, learned, lr=1e-3) — 284,196 params**
```
Step   Loss    BPC    Acc    GNorm  Speed
  1    5.5089  7.948  0.008  1.8    0.6 step/s
 50    4.6007  6.637  0.151  1.7    0.6 step/s
100    3.8193  5.510  0.198  1.1    0.5 step/s
150    2.8200  4.068  0.282  0.4    0.5 step/s
200    2.4956  3.600  0.339  0.3    0.5 step/s
250    2.3279  3.358  0.369  0.3    0.5 step/s
300    2.2293  3.216  0.388  0.3    0.5 step/s
350    2.1579  3.113  0.403  0.3    0.5 step/s
400    2.0952  3.023  0.418  0.3    0.5 step/s
450    2.0468  2.953  0.428  0.3    0.5 step/s
500    2.0114  2.902  0.437  0.3    0.5 step/s
FINAL: loss=2.0533  bpc=2.962  acc=0.427  best_bpc=2.822  (917s)
```

**Config D (H=512, R=2, learned, lr=1e-3) — 284,196 params (WINNER)**
```
Step   Loss    BPC    Acc    GNorm  Speed
  1    5.5085  7.947  0.008  1.8    0.6 step/s
 50    4.6004  6.637  0.152  1.7    0.5 step/s
100    3.8183  5.509  0.199  1.1    0.5 step/s
150    2.8180  4.066  0.283  0.4    0.5 step/s
200    2.4941  3.598  0.339  0.3    0.5 step/s
250    2.3275  3.358  0.369  0.3    0.5 step/s
300    2.2294  3.216  0.387  0.3    0.5 step/s
350    2.1581  3.114  0.403  0.3    0.5 step/s
400    2.0951  3.023  0.417  0.3    0.5 step/s
450    2.0464  2.952  0.428  0.3    0.5 step/s
500    2.0111  2.901  0.437  0.3    0.5 step/s
FINAL: loss=2.0531  bpc=2.962  acc=0.427  best_bpc=2.819  (929s)
```

**Config E (H=512, R=2, bitlift, lr=1e-3) — 157,732 params**
```
Step   Loss    BPC    Acc    GNorm  Speed
  1    5.6230  8.112  0.000  3.2    0.5 step/s
 50    4.7659  6.876  0.163  2.3    0.5 step/s
100    4.0702  5.872  0.176  1.6    0.5 step/s
150    3.2696  4.717  0.190  0.8    0.5 step/s
200    3.0889  4.456  0.201  1.1    0.5 step/s
250    2.9288  4.225  0.235  1.4    0.5 step/s
300    2.7837  4.016  0.268  1.4    0.5 step/s
350    2.6719  3.855  0.284  1.5    0.5 step/s
400    2.5846  3.729  0.297  2.2    0.5 step/s
450    2.5220  3.638  0.309  4.6    0.5 step/s
500    2.4748  3.570  0.319  4.7    0.5 step/s
FINAL: loss=2.5297  bpc=3.650  acc=0.308  best_bpc=3.489  (1104s)
NOTE: gnorm diverging (1.8→4.7) — bitlift encoding is unstable
```

**Config F (H=512, R=2, learned, lr=5e-4) — 284,196 params**
```
Step   Loss    BPC    Acc    GNorm  Speed
  1    5.5085  7.947  0.008  1.8    0.6 step/s
 50    4.8891  7.053  0.132  1.8    0.5 step/s
100    4.2019  6.062  0.171  1.4    0.5 step/s
150    3.2379  4.671  0.238  0.7    0.5 step/s
200    2.8331  4.087  0.285  0.4    0.5 step/s
250    2.6221  3.783  0.318  0.3    0.5 step/s
300    2.4865  3.587  0.341  0.3    0.5 step/s
350    2.3893  3.447  0.358  0.3    0.5 step/s
400    2.3090  3.331  0.372  0.3    0.5 step/s
450    2.2477  3.243  0.384  0.3    0.5 step/s
500    2.2009  3.175  0.394  0.3    0.5 step/s
FINAL: loss=2.2550  bpc=3.253  acc=0.383  best_bpc=3.091  (927s)
```

**Config G (H=512, R=3, learned, lr=1e-3) — 284,196 params**
```
Step   Loss    BPC    Acc    GNorm  Speed
  1    5.5086  7.947  0.008  1.8    0.6 step/s
 50    4.6004  6.637  0.151  1.7    0.5 step/s
100    3.8181  5.508  0.199  1.1    0.5 step/s
150    2.8176  4.065  0.283  0.4    0.5 step/s
200    2.4942  3.598  0.339  0.3    0.5 step/s
250    2.3287  3.360  0.369  0.3    0.5 step/s
300    2.2312  3.219  0.387  0.3    0.5 step/s
350    2.1599  3.116  0.403  0.3    0.5 step/s
400    2.0968  3.025  0.417  0.3    0.5 step/s
450    2.0480  2.955  0.428  0.3    0.5 step/s
500    2.0124  2.903  0.436  0.3    0.5 step/s
FINAL: loss=2.0546  bpc=2.964  acc=0.427  best_bpc=2.823  (942s)
```

**Config H (H=256, R=2, learned, lr=5e-4) — 158,756 params**
```
Step   Loss    BPC    Acc    GNorm  Speed
  1    5.6398  8.137  0.002  1.8    0.4 step/s
 50    5.0498  7.285  0.109  1.7    0.4 step/s
100    4.2775  6.171  0.150  1.3    0.4 step/s
150    3.2714  4.720  0.214  0.7    0.4 step/s
200    2.9268  4.222  0.258  0.4    0.4 step/s
250    2.7343  3.945  0.290  0.3    0.4 step/s
300    2.6007  3.752  0.309  0.3    0.4 step/s
350    2.5040  3.612  0.326  0.2    0.4 step/s
400    2.4240  3.497  0.342  0.2    0.4 step/s
450    2.3618  3.407  0.353  0.2    0.4 step/s
500    2.3133  3.337  0.361  0.2    0.4 step/s
FINAL: loss=2.3687  bpc=3.417  acc=0.352  best_bpc=3.256  (1358s)
```

---

### Full results.json path
`v4/sweep_results/hyperparam_sweep_20260307_160856/results.json`

### CSV per-step logs
`v4/sweep_results/hyperparam_sweep_20260307_160856/*_seed42.csv`


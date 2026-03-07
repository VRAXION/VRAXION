# Hyperparameter Sweep — 20260307_160229

**Data**: WikiText-2 (11.1 MB) | **Steps**: 50 | **Batch**: 32x256 | **Device**: cpu

## Results

| # | Cfg | Label | H | R | Embed | LR | Params | Final BPC | Best BPC | Final Acc | Best Acc | GNorm | Time |
|---|-----|-------|---|---|-------|----|--------|----------|----------|-----------|----------|-------|------|
| 1 | B | small_R2 | 256 | 2 | learned | 1e-03 | 158,756 | 6.802 | 5.106 | 0.133 | 0.197 | 1.6 | 141s |
| 2 | A | base_small | 256 | 1 | learned | 1e-03 | 158,756 | 6.803 | 5.107 | 0.133 | 0.197 | 1.6 | 136s |

## Factor Analysis

| Factor | Comparison | BPC Delta | Winner |
|--------|------------|-----------|--------|
| R (H=256) | R=1→R=2 | +0.000 | R=2 |

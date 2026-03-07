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

*(Will be filled after sweep completes)*


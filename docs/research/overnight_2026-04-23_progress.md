# Overnight SCT Empirical Research — Progress Log

**Start**: 2026-04-24 01:42 CEDT (night of 2026-04-23)
**Branch**: `research/overnight-sct-empirical-20260423`
**Anchor goal**: validate or refute the Structured Chaos Theory formula (`L = Ψ · σ_μ / D`) using data that already exists in the repo, before writing new theory.

---

## Iteration 1 — Ground-truth data scan

**Timestamp**: 2026-04-24 01:42 CEDT
**Type**: Data mining (ground-truth feasibility scan)

### What was run

- Inventoried `target/` (**162 subdirectories** of experimental bundles) and `output/` (5 champion/sweep dirs)
- Inspected `target/grower-regression/` (3 bundles from 2026-04-12: 131312Z, 131653Z, 151618Z)
- Inspected `target/byte-opcode-acceptance/` (3 bundles from 2026-04-12)
- Deep-read `output/byte_unit_latent_dim_sweep_gpu_probe/summary.json` (generated 2026-04-23 22:59, just before the loop started)

### What was observed (concrete numbers)

The **GPU probe** is a 60-configuration sweep: 5 latent_dims × 3 hiddens × 4 activations × 1 codebook (binary), **1 seed per cell**.

Aggregate statistics for `final_lossless`:

| Grouping | min | mean | max |
|---|---|---|---|
| activation=identity (n=15) | 9.38 | 30.29 | 44.92 |
| activation=tanh (n=15) | 21.09 | 36.07 | 71.09 |
| activation=relu (n=15) | 11.33 | 32.68 | 56.64 |
| **activation=c19 (n=15)** | **20.31** | **49.40** | **78.91** |
| hidden=8 (n=20) | 9.38 | 29.88 | 57.03 |
| hidden=12 (n=20) | 10.55 | 35.57 | 67.97 |
| **hidden=16 (n=20)** | **24.61** | **45.88** | **78.91** |
| latent_dim=8 (n=12) | 15.23 | 28.55 | 40.62 |
| latent_dim=10 (n=12) | 9.38 | 32.16 | 75.78 |
| latent_dim=12 (n=12) | 19.92 | 38.41 | 76.95 |
| latent_dim=16 (n=12) | 21.09 | 37.86 | 56.25 |
| **latent_dim=24 (n=12)** | **11.33** | **48.57** | **78.91** |

- **Best cell**: H=16, LD=24, c19 → 78.91 % final lossless
- **Worst cell**: H=8, LD=10, identity → 9.38 % final lossless

### Implication for SCT — CONTRADICT (tentative)

The SCT formula predicts:
> `L = Ψ · σ_μ / D`. Higher D (search dimensionality) → lower L.

The probe shows the OPPOSITE for both axes that plausibly measure D:
- Higher hidden dimension (`8 → 12 → 16`) gives **monotonically higher** mean learning.
- Higher latent_dim (`8 → 10 → 12 → 24`) gives higher mean learning (LD=16 is the non-monotonic outlier, but the trend is still upward).

If hidden × latent_dim is what the formula means by D, the data contradicts the formula's naive form. The rescue would be: "Ψ grows with capacity" — but that makes Ψ an unmeasured free parameter absorbing any deviation. Unfalsifiable.

The activation effect (c19 >> identity by +19 pp mean) IS consistent with "architectural prior matters", which is the spirit of Ψ, but that's a categorical comparison, not a scalar scaling law.

### Adversarial question

**Single seed per cell.** The 78.91 vs 9.38 spread could be primarily seed noise, not architecture. If I re-ran just the H=16/LD=24/c19 cell with 5 seeds, would the variance dwarf the cross-config differences? Until we know the seed-variance floor, no architectural claim can be separated from luck.

### Next iteration choice — adversarial probe of this finding

The most informative next step is NOT more mining. It is: **pick the best cell and the worst cell from this sweep, and run each with 3-5 seeds**. If the best cell stays >>50% and the worst stays <<25% across seeds, architecture is real. If they collapse toward the mean, we were looking at noise.

Since I cannot launch GPU training loops without the original script + approval (and `diag_byte_unit_latent_dim_sweep.py` or equivalent is the relevant tool, which I need to locate), **next iteration** will:

1. Locate the source script that generated this sweep (`grep` for `latent_dim_sweep`)
2. Check whether it supports a `--seeds` flag or can be parametrized
3. If yes → plan the multi-seed rerun for iteration 3 (CPU if feasible, otherwise flag for morning)
4. If no → pivot to iteration type 1 (more mining) on a DIFFERENT data source — specifically the 3 grower-regression bundles, whose `metrics.json` may have per-seed per-step data I haven't looked at yet

---

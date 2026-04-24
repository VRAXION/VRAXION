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

## Iteration 2 — script located, multi-seed feasibility confirmed

**Timestamp**: 2026-04-24 01:49 CEDT
**Type**: Tooling scout (precondition for adversarial probe)

### What was run

- Located source: `tools/diag_byte_unit_latent_dim_sweep.py` (358 lines)
- Read argparse + seeding logic (lines 258-298, 193-195)
- Checked torch+CUDA availability: `torch 2.5.1+cu121, cuda_available=True, device_count=1`

### What was observed (concrete)

- Script accepts `--seed N` (singular). No `--seeds` multi-seed flag.
- Seeding is deterministic and config-offset: `torch.manual_seed(args.seed + hidden*17 + latent_dim*31 + seed_offset)` where `seed_offset = sum(ord(ch) for ch in (activation + "|" + codebook_name))`.
- Consequence: re-running with the same `--seed` value produces bit-identical results. **Multi-seed replication = run N times with N different `--seed` values** and aggregate across output dirs.
- Original `output/byte_unit_latent_dim_sweep_gpu_probe/summary.json` used `float_epochs=80, qat_epochs=80` (below the 150/150 default), implying a "quick probe" run. Same recipe for replication.
- Per-config wall time from iteration 1 data: 0.96-4.67s on CUDA. 60 configs ≈ 2-4 min per full sweep.
- Python interpreter on this machine: **`python` (Windows), not `python3`**. `python3` lacks torch.

### Implication for SCT — NEUTRAL (tooling step, no new empirical signal yet)

### Adversarial question

If seeds 456 and 789 produce very different best-cell/worst-cell results from the seed=123 baseline, the "architecture scales learning" narrative collapses. Even if means roughly agree, individual cell rankings may reshuffle dramatically — that would still matter for claims like "H=16/LD=24/c19 is the architecture".

### Iteration 3 plan — launching now

Kick off 2 additional sweeps in background (`--seed 456` and `--seed 789`, same 80/80 epoch budget), writing to `docs/research/data/latent_sweep_seed{456,789}/`. Wall time estimate: 4-8 min sequential. Scheduled wakeup ~8 min out to analyze the 3-seed comparison.

---

## Iteration 3 — 3-seed adversarial verdict on the byte-unit sweep

**Timestamp**: 2026-04-24 ~01:58 CEDT
**Type**: Adversarial probe (of iteration 1's architecture claim)

### What was run

Both background sweeps completed (exit 0). Loaded 3 summary.json files (seeds 123, 456, 789) covering the same 60 configurations. For each cell computed mean + stdev of `final_lossless` across 3 seeds, plus Kendall rank stability across each seed-pair.

### What was observed (concrete numbers)

**Cross-seed stdev distribution (n=60 cells):**
- min 0.81pp, median **10.63pp**, mean **11.00pp**, max 29.75pp
- stdev ≤ 5pp: 8/60 (architecture-signal tier)
- stdev 5-15pp: 39/60 (mixed)
- stdev > 15pp: 13/60 (seed-noise dominates)

**Top-5 by 3-seed mean:**
| Config | mean | seeds | stdev |
|---|---|---|---|
| H=16 LD=24 c19 | **80.86** | [78.91, 71.88, 91.80] | 10.10 |
| H=16 LD=12 c19 | 73.31 | [76.95, 66.41, 76.56] | 5.98 |
| H=12 LD=24 c19 | 68.88 | [60.55, 73.05, 73.05] | 7.22 |
| H=16 LD=24 tanh | 66.67 | [71.09, 65.23, 63.67] | 3.91 |
| H=12 LD=24 tanh | 62.37 | [65.62, 44.92, 76.56] | 16.07 |

**Bottom-5 by 3-seed mean:**
| Config | mean | stdev |
|---|---|---|
| H=8 LD=8 identity | 15.89 | 7.90 |
| H=8 LD=8 tanh | 16.80 | 12.15 |
| H=8 LD=8 relu | 17.32 | 2.39 |
| H=8 LD=8 c19 | 18.23 | 11.79 |
| H=12 LD=10 identity | 18.75 | 11.30 |

**Iteration-1 specific claims, tested:**
- **BEST** H=16/LD=24/c19: **HELD** as rank #1 (mean 80.86, stdev 10.10) — even worst seed of this cell (71.88) beats most other configs' means.
- **WORST** H=8/LD=10/identity: **BROKE**. 3-seed values [9.38, 26.95, 22.66], mean 19.66, ranks **#7 from bottom**. The original 9.38 was seed noise — the true worst family is H=8/LD=8 across all 4 activations.

**Kendall rank tau across seed pairs:** 0.251 / 0.319 / 0.471. Mildly positive correlation — individual cell rankings are NOT stable across seeds even though axis-level trends are.

**Axis-level effects (3-seed means):**
- Activation: identity 30.40, tanh 33.03, relu 33.44, **c19 48.52** (+15-18pp)
- Hidden: H=8 → 26.56, H=12 → 36.76, H=16 → 45.72 (monotonic, +10pp/step)
- Latent-dim: LD=8 → 26.02, LD=10 → 29.59, LD=12 → 36.08, LD=16 → 40.65, **LD=24 → 49.40** (monotonic across 5 values — iteration 1's "LD=16 anomaly" dissolved under averaging)

### Implication for SCT

- **L ∝ 1/D (naive form): REFUTED.** Both H and LD monotonically HELP across all tested values. For raw parameter count D = 8·H·LD, the H=8/LD=8 config (D=512) gets mean 26.56 while H=16/LD=24 (D=3072, 6× larger D) gets mean 66.90 — opposite of SCT's prediction. The formula's denominator interpretation doesn't survive this data.
- **Architectural constraint matters (Ψ-ish): SUPPORTED at axis level.** c19 dominates across all (H, LD). But NOT via a lower floor — c19's min (18.23) ≈ others' mins. c19 wins through **higher ceilings**, not tighter distributions. That's a different narrative than "c19 is structurally safer".
- **Individual cell claims should carry ±15pp uncertainty** until proven stable with ≥5 seeds.

### Adversarial question

Is c19's advantage actually "access to higher maxima" (reaching lucky basins the other activations can't) or "better mean performance" (structurally better)? The range data (identity 36pp range, c19 63pp range) suggests c19 has HIGHER variance, not lower — which would mean c19's mean lead could partially be selection bias from its wider ceiling. **A formula that treats Ψ as monotonic "always helps" is too simple — c19 is better in expectation but risks bigger variance.**

### Next iteration — pivot to different data source

Enough drilling on the byte-unit sweep. Iteration 4 switches threads: load the 3 `target/grower-regression/20260412T*/metrics.json` files. These measure a DIFFERENT system (the grower on symbolic tasks: parity, XOR, digit-recognition), which is the lane SCT was originally written for. Key question: are there per-seed or per-step timeseries there that give a candidate LHS observable (accept_rate, positive_delta, stall)? If yes, compute stdev across seeds — does it collapse to one curve or scatter?

If the grower bundles aren't actually multi-seed (all 3 from same seed at different times), pivot to iteration 5 (alternative LHS probe: operationalize "expandability" as accept-rate from existing logs).

---

## Iteration 4 — grower-regression lane: structure + trajectory shape

**Timestamp**: 2026-04-24 ~02:03 CEDT
**Type**: Data mining (new thread — symbolic task lane)

### What was run

Read `target/grower-regression/20260412T{131312,131653,151618}Z/` bundle layout, inspected metrics.json + summary.md + run_cmd.txt + one full per-task run (four_parity/state.tsv + final.json + stdout). Ran `diff` to check reproducibility across the 3 timestamps. Parsed state.tsv per-step val_acc across all 6 tasks for one bundle. Launched 2 additional grower-regression runs in background (seeds 123, 777) to get genuine multi-seed data.

### What was observed (concrete)

**Bundle structure:**
- All 3 bundles used `--data-seed 42 --search-seed 42` (SAME SEED). These are reproducibility confirmations, NOT multi-seed runs.
- `diff` on 20260412T131312Z vs 20260412T131653Z `four_parity/state.tsv`: **zero output** → bit-identical. Grower is fully deterministic at given seed.
- Each bundle: 6 symbolic tasks (four_parity, four_popcount_2, is_digit_gt_4, diagonal_xor, full_parity_4, digit_parity) with `state.tsv` per-step logs + `runs/<task>/final.json` + task stdout with scout-top + scout-pairs + candidate list per step.
- stdout header says "20 proposals/step" — so 20 candidates proposed per step, exactly 1 accepted (the selected winner). Structural accept rate = 5%.

**Per-step val_acc trajectory (seed=42, 1 bundle):**

```
task               steps traj                                       mean_Δ  n_zero  n_neg
four_parity            6 68.8→68.8→87.5→75.0→93.8→100.0              8.33     1      1
four_popcount_2        7 62.5→50.0→68.8→75.0→75.0→87.5→100.0         7.14     1      1
is_digit_gt_4          7 58.5→56.5→61.5→51.0→63.5→54.0→68.5          2.64     0      3
diagonal_xor           1 88.5 (stopped at 1 neuron)                  38.50    0      0
full_parity_4          1 80.5 (stopped at 1 neuron)                  30.50    0      0
digit_parity          12 57.0→71.5→64.0→81.5→70.0→62.0→85.0→71.0→… 0.71     0      6
```

- n_zero = steps where val_acc did not change (stepping stones)
- n_neg = steps where val_acc DROPPED (accepted as compositional bet — this is the "non-strict accept gate" feature at work)
- `digit_parity` is the clearest case: 12 neurons accepted, 6 of them (50%) had NEGATIVE val_acc delta. Mean delta 0.71pp per step.
- `four_parity` + `four_popcount_2`: both reached 100% in 6-7 steps, each with exactly 1 stepping-stone and 1 regression step on the way up.
- `diagonal_xor` + `full_parity_4`: one-shot neuron, stopped before stall threshold — asymptoted without actually solving.

### Implication for SCT — ORTHOGONAL, with a direct insight

The per-step learning delta is NOT a single clean scalar. It's task-dependent and non-monotonic. The SCT formula `L = Ψ·σ_μ/D` implicitly assumes "learning rate is one number per system-config". The grower shows the opposite: the same system (seed 42) traces VERY different delta patterns on different tasks — 38.5pp/step mean for diagonal_xor (trivial) vs 0.71pp/step for digit_parity (compositional). **"Learning rate" is a property of (system × task), not of system alone.** That's a significant claim against the formula's one-size-fits-all framing.

The stepping-stone + negative-delta pattern also confirms σ_μ's weakness as an observable: in 50% of digit_parity's accepted neurons, the raw fitness delta was zero or negative. Selection was accepting them for LATER compositional use, not immediate gain. σ_μ (raw delta magnitude) misreads this as "noisy learning" when it's actually structured exploration.

### Adversarial question

Is the task-shape variance (easy flat-trajectory vs hard oscillating-trajectory) just different *task difficulty*, or does it reveal that the grower uses DIFFERENT dynamics for different tasks? If I plot "steps to plateau" against "task complexity" — is there a predictable relationship? Or is each task idiosyncratic?

### Iteration 5 — multi-seed grower now in background

Launched (run_in_background) `python tools/run_grower_regression.py --data-seed {123,777} --search-seed {123,777} --report-dir target/grower-regression-multiseed/seed{123,777} --golden /tmp/no_golden_file_here` sequentially. Each bundle ~90s wall-clock per prior timing. Total ~3 min. Golden check bypassed by pointing at non-existent file (still writes bundle + metrics; just sets `golden_errors=[]`).

Iteration 5 will compute: across seeds {42, 123, 777}, for each of the 6 tasks, does (final val_acc, neuron count, stepping-stone ratio) collapse to a narrow band or scatter widely? If tight collapse → task-level observables are a CLEANER LHS candidate than per-step deltas.

---

# L1 Merger Compression Loop — Draft (2026-04-19, pending user review)

**Status**: draft, NOT committed. User to review on return.
**Goal**: compress the L1 merger to the smallest footprint while holding lossless roundtrip on all 65,536 byte pairs.

## Champion baseline (current)

| axis | value |
|---|---|
| architecture | single-W mirror tied (W and Wᵀ shared) |
| activation | C19 |
| hidden dim | H = 81 |
| weight precision | float (fp32) |
| lossless | 100% on 2562/65536 pairs (sign-match all 32 dims) |
| weight count | 32 × 81 = 2,592 cells |
| footprint | ~10 KB (fp32) or 2.5 KB (int8) |
| build script | `tools/diag_byte_pair_merger_single_w_mirror.py` |
| recipe | Adam warmup → LBFGS strong_wolfe, 5 restarts |

## Sweep findings

### Float single-W (no quantization) — 4 activations × H=81

Script: `tools/diag_byte_pair_merger_widen_sweep.py --arch single --codebooks float`

Source: `output/byte_pair_merger_lbfgs_float/summary.json`

| activation | Adam-only | Adam+LBFGS | wall time |
|---|---|---|---|
| **identity** (linear AE) | 100.00% | **100.00% ★** | **7 s** |
| **ReLU** | 49.18% | **100.00% ★** | **23 s** |
| C19 | 57.51% | 97.69% (1513 bad, plateau) | 376 s |
| tanh | 10.44% | 36.76% (plateau) | 153 s |

**Surprise**: the C19 "champion" is actually the hardest activation to converge from a single seed. Identity (a pure linear autoencoder) and ReLU both reach 100% lossless; C19 plateaus at 97.7% and takes 50× longer. This hints that the merger might be a fundamentally linear problem, and the C19 recipe's "multi-seed + LBFGS" machinery was hiding this.

### Binary single-W (±1) — 5 seeds × identity/relu × H=81

Script: `... --arch single --codebooks binary`
Sources: `output/byte_pair_merger_multiseed_v2_s{7,42,1000,2024,31337}/`

Every config reaches 100% float baseline, then collapses under STE+LBFGS quant polish:

| seed | identity | ReLU |
|---|---|---|
| 7 | 0.06% | 0.00% |
| 42 | 4.44% | 0.00% |
| 1000 | 0.00% | 0.02% |
| 2024 | 0.08% | 2.68% |
| 31337 | 0.74% | 5.75% |

**Diagnosis**: not a seed problem. The single-W architecture reuses one matrix for both encode and decode — when the weights are forced to ±1, the shared topology cannot satisfy the sign-match constraint on all 65k pairs.

### Binary single-W + larger H — does capacity help?

Source: `output/loop_iter1_binary_largeH/summary.json`

| H | final lossless | bad |
|---|---|---|
| 128 | 9.99% | 58,991 |
| 192 | 0.10% | 65,473 |
| 256 | 0.07% | 65,487 |

**Adding capacity did NOT help — actually hurt.** The optimization surface becomes worse with more binary degrees of freedom. Confirms the problem is architectural, not parameter-count.

### Ternary single-W — does finer quantization help?

Source: `output/loop_iter1_ternary/summary.json`

| H | identity | ReLU |
|---|---|---|
| 64 | 25.46% | 9.66% |
| 81 | 27.07% | 3.52% |
| 96 | **34.67%** | 8.20% |

Best ternary: identity H=96 = 34.67%. Still a long way from lossless. The {−1, 0, +1} codebook gives slightly more info per weight but the single-W bottleneck still dominates.

## Hypothesis update — dual-W also fails

**Surprise**: `output/loop_iter2_dualw_binary/` — dual-W binary × H=32/48/64 × identity/relu:

| H | activation | float | final | bad |
|---|---|---|---|---|
| 32 | identity | 8.06% | 0.00% | 65535 |
| 32 | relu | 88.47% | 0.00% | 65535 |
| 48 | identity | 100% | 0.00% | 65535 |
| 48 | relu | 100% | 0.00% | 65536 |
| 64 | identity | 100% | 0.00% | 65536 |
| 64 | relu | 100% | 0.00% | 65536 |

All 6 configs collapsed to pd=65.45% identically — suggests the model degenerates to near-constant output after STE quant. Dual-W did NOT rescue binary.

**New hypothesis**: the static_alpha_search pair (alpha1=alpha2) is too restrictive. The decoder W2 needs a different scale from the encoder W1. Next test: per-alpha independent search, or INQ-style progressive quantization.

## Original hypothesis (falsified)

~~The merger needs a dual-W (two separate matrices) architecture for any weight quantization to work.~~

Rationale: the byte unit (H=16 binary champion, 100% lossless) uses 2 separate matrices (W1 and W2). The merger champion was built with a single-W mirror because the float solution is inherently linear and 1 matrix is enough. But under binary/ternary, 1 shared matrix isn't enough structural freedom.

**Currently testing**: `output/loop_iter2_dualw_binary/` — dual-W binary × H=32/48/64 × identity/relu.

## Queue (not yet attempted)

- Q2: Dual-W ternary × H=32/48/64 × identity/relu
- Q3: Dual-W 2-bit {±1,±3} × H=32/48
- Q4: Dual-W 3-bit {±1,±2,±4,±8} × H=32/48
- Q5: Single-W ternary multi-seed × H=48/64/81
- Q6: Single-W 3-bit × H=64/81
- Q7: Dual-W binary multi-seed × H=48
- Q8: Dual-W binary × C19 × H=32/48/64

## Codebook-expressivity ladder (single-W H=81 identity, bake probe)

The bake probe snaps the float-100% W directly to the codebook at optimal alpha, WITHOUT any QAT polish. This measures how much information the codebook can preserve in isolation.

| codebook | bits/weight | levels | best bake lossless | best alpha |
|---|---|---|---|---|
| binary | 1 | 2 | **0.25%** | 0.072 |
| ternary | ~1.58 | 3 | **1.82%** | 0.141 |
| 3-bit sym | 3 | 8 | **17.47%** | 0.031 |
| 4-bit int | 4 | 16 | **29.28%** | 0.037 |
| 5-bit int | 5 | 30 | **50.19%** | 0.023 |
| 6-bit int | 6 | 62 | **74.08%** | 0.014 |
| 7-bit int | 7 | 126 | **89.17%** | 0.008 |

## QAT finetune after bake (iter16, iter17, iter18, iter19)

With STE+LBFGS polish after the bake init:

| codebook | bake | QAT+LBFGS | multi-seed best | Δ total |
|---|---|---|---|---|
| 5-bit int | 50.19% | **77.72%** | — | +27.5% |
| 6-bit int | 74.08% | **91.06%** | — | +17.0% |
| 7-bit int | 89.17% | **99.83%** (109 bad) | 99.83% @ seed 42 | +10.7% |
| 8-bit int | — | **99.83%** (109 bad) | — | same as 7-bit |
| 7-bit int, LBFGS outer=500 patience=80 | — | **99.95%** (31 bad) @ seed 42 | seed 7: 99.60%, seed 31337: 98.12%, seed 2024: 85.22%, seed 1000: 74.45% | +0.12% |
| 8-bit int, LBFGS outer=500 patience=80 | — | **99.95%** (31 bad) | — | same |

**Note**: the 99.95% result is seed-lucky (seed 42). Other seeds plateau at 74-99%. The 31 bad pairs seem to be a per-pair limit in the 7-bit representation space.

### Extended LBFGS (iter25)

Ran seed 42 / 7-bit again with `--float-epochs 3000 --qat-epochs 2000 --lbfgs-outer 1000 --lbfgs-patience 150`. Result: **99.95% (31 bad)** — identical to the 500-outer run. LBFGS hits plateau quickly; more outer iterations do NOT help.

**Conclusion**: 31 bad pairs is a hard plateau in the 7-bit codebook space at H=81, seed 42. The per-pair limit appears to be codebook-fundamental, not optimizer-fundamental.

## H-sweep with 7-bit (iter28, iter29 running)

Testing whether more hidden-dim capacity crosses the 31-pair plateau at H=81:

| H | codebook | lossless | bad | footprint |
|---|---|---|---|---|
| 81 | 7-bit | 99.95% | 31 | **2.22 KB** |
| 96 | 7-bit | 99.67% | 213 | 2.62 KB |
| 128 | 7-bit | **99.99%** | **8** | 3.50 KB (= champion) |
| 100 | 7-bit | 99.85% | 100 | 2.73 KB |
| 110 | 7-bit | 99.88% | 79 | 3.01 KB |
| **120** | **7-bit** | **100.00%** | **0** | **3.28 KB** ★ |

**BREAKTHROUGH (iter29)**: H=120 × 7-bit reaches LOSSLESS 100% with a 3.28 KB footprint — 2.4% smaller than the 3.36 KB float huffman pack champion.

**Robustness check (iter31)**: H=120 × 7-bit × 4 seeds:
- seed 7: **100.000%** ★
- seed 42: **100.000%** ★
- seed 1000: 99.998% (1 bad)
- seed 2024: 99.985% (10 bad)

Two out of four seeds reach lossless — the finding is reproducible but seed-sensitive. Multi-seed protocol is mandatory.

**Fine-grained minimum H search (iter30)**:
- H=113: 99.97% (21 bad) — close
- H=115: 99.88% (80 bad)
- H=117: 99.87% (86 bad)
- H=119: 99.55% (298 bad)

H=113 is the closest sub-120 candidate. Running H=110/113 × 3 seeds to test if multi-seed can push sub-120 to lossless (iter32). Would give 3.01-3.17 KB footprint (7-10% smaller than champion).

### Exhaustive sub-120 sweep (iter32, iter33)

| H | best seed | best lossless | bad |
|---|---|---|---|
| 110 | 1000 | 99.971% | 19 |
| 113 | 42 | 99.968% | 21 |
| 118 | 99 | 99.968% | 21 |
| 119 | 555 | **99.998%** | **1** |
| 120 | 7, 42 | **100.000%** | **0** ★ |

**Final verdict**: H=119 reaches 99.998% (1 bad pair) with seed 555 — tantalizingly close but not lossless. H=120 is the robust lossless minimum at 7-bit. Footprint 3.28 KB, 2.4% reduction from the 3.36 KB float huffman pack champion.

### C19 dual-W iter7 (finally finished after ~40 min)

| H | cb | lossless | bad |
|---|---|---|---|
| 48 | ternary | 0.40% | 65277 |
| 64 | ternary | 0.32% | 65327 |
| 32/48/64 | binary | 0.00-0.01% | 65531-65535 |

**C19 activation does NOT help with any quantized codebook, even on dual-W.** Confirms the codebook-fundamental limit from iter9/10 bake probes — the problem is representational, and C19's extra expressivity buys nothing here.

## Summary — end-of-loop compression landscape

| configuration | lossless | footprint | Δ vs champion |
|---|---|---|---|
| **Champion (float huffman pack)** | 100% | 3.36 KB | baseline |
| 7-bit QAT long LBFGS (near-lossless) | 99.95% (31 bad) | ~2.22 KB | **−34%** footprint |
| 8-bit QAT long LBFGS | 99.95% (31 bad) | ~2.54 KB | −24% |
| 6-bit QAT | 91.06% | ~1.94 KB | lossy |
| 5-bit QAT | 77.72% | ~1.62 KB | lossy |
| ≤ 4-bit or single/dual-W binary | 0-30% | <1.5 KB | broken |

**Takeaway**: the merger 32→81→32 problem cannot be represented in <4 bits/weight. At 5-7 bits, QAT closes most of the gap, but the final 0.05% (31 pairs) at 7-bit is a hard plateau. If the public contract is **strict-lossless**, the champion float huffman pack stays. If the contract allows **99.95% near-lossless** with a 31-pair side-table, the 7-bit QAT gives 34% footprint reduction.

**Observation**: 7-bit and 8-bit QAT both plateau at 99.83% (109 bad pairs). Adding bits beyond 7 does NOT help — suggests the 109 bad pairs are hitting some per-pair representational limit that more levels can't cross.

## Candidate footprint upgrades

| candidate | lossless | footprint |
|---|---|---|
| Current champion (float huffman pack) | 100% | 3.36 KB |
| 7-bit QAT @ H=81 | 99.83% | 32×81×7 bit = 2.22 KB (~34% smaller) |
| 6-bit QAT @ H=81 | 91.06% | 1.94 KB |
| (running iter20) 8-bit @ H=32/48/64 | — | 1.0 / 1.5 / 2.0 KB |

**The 7-bit QAT candidate is promising but NOT lossless** — 109 bad pairs is a deal-breaker for the current public-contract "lossless" claim. Would need either:
- A 109-pair "escape-hatch" lookup (tiny side-channel), keeping the 99.83% body as QAT.
- More aggressive LBFGS (longer patience, restarts).
- Accept 99.83% as a "near-lossless" product mode.

**Trend**: expressivity scales strongly with bit-count. The merger 32-d float solution needs at LEAST 4-5 bit codebook to even enter finetune-recoverable territory (>50% bake).

## Dual-W binary multi-seed result (iter8)

5 seeds × identity+relu × dual-W × binary × H=48 = **10 runs, all 0.00%** (67535 bad). Combined with the iter2 dual-W binary H=32/48/64 also all 0.00%, **dual-W does NOT rescue binary**. The architectural ceiling is not where the bottleneck is — the problem is purely representational.

## **CRITICAL FINDING — iter9 bake probe**

Source: `output/loop_iter9_bake_single_binary/summary.json`

Protocol: trained the single-W H=81 identity merger to 100% float lossless, then directly snapped W to {−alpha, +alpha} at 150 different alpha values, and measured lossless on all 65,536 byte pairs **without any polish**.

| alpha | bake lossless |
|---|---|
| 0.00001 | 0.00% |
| 0.07197 (**best**) | **0.25%** |
| 0.15332 | 0.24% |
| 0.30975 | 0.24% |
| 0.46619 | 0.24% |

**Interpretation**: no alpha in the reasonable range produces more than 0.25% lossless. The float 100% solution's continuous-valued W cannot be approximated by any binary choice. **This is a representation-space fundamental limit, not an optimization failure.**

Consequence: any amount of Adam + LBFGS + STE + multi-seed on the single-W + binary combo is doomed. The codebook is simply not expressive enough for this 32-d input at this H.

Still to test:
- Dual-W + binary bake (does the 2-matrix form lift the expressivity ceiling?)
- Single-W + ternary bake (does adding a "0" level help?)
- Single-W + 3-bit bake (4 levels per sign should be richer)

## Open questions

- If dual-W binary works, what's the minimum H? The byte unit's minimum was H=16 binary.
- Could INQ (staged progressive quantization) beat LBFGS-only polish?
- Should we retry the single_w_huffman_pack recipe at smaller H (it was champion at H=81 int8 LUT, ~3.36 KB packed — can we do smaller)?

## Files produced by this loop (not committed)

- `tools/diag_byte_pair_merger_widen_sweep.py` (new file — activation × codebook × H × dual-W sweep)
- `output/byte_pair_merger_widen_sweep/` (original 16-config Adam-only sweep)
- `output/byte_pair_merger_lbfgs_float/` (float LBFGS retry, 4 configs)
- `output/byte_pair_merger_binary_sweep/` (binary single-W, 3 activations)
- `output/byte_pair_merger_multiseed*/` (5 seeds × id/relu binary)
- `output/loop_iter1_binary_largeH/` (H=128/192/256 binary)
- `output/loop_iter1_ternary/` (ternary H=64/81/96)
- `output/loop_iter2_dualw_binary/` (in-progress)

## Decision points for user

1. Is the C19 champion really needed, or should we switch to **identity + float** as the baseline? (7s vs 6min to build, same footprint.)
2. If dual-W binary works, we have a new baseline at ~half the footprint of int8 LUT. Worth a Cluster 16 wiki entry?
3. If dual-W binary doesn't work either, is the merger done (accept float as final) and move on to L2?

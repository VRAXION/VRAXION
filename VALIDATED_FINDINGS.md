# VRAXION Validated Findings

Canonical evidence summary. Repo-tracked docs are canonical; the GitHub wiki is a mirrored secondary surface.

## Current State (v5.0.0-beta.1)

The primary implementation surface is now **Rust** (`instnct-core`). The Python reference line (`instnct/model/graph.py`) remains in-repo for developers.

### Rust lane: proven results

| Finding | Result | Status |
|---|---|---|
| **Smooth cosine-bigram fitness** | 21.7% peak with 1+1 ES (+2.6pp over stepwise argmax) | **Current mainline** — default in `evolve_language.rs` |
| **1+9 jackpot selection** | **25.8% peak** (prefill seed=123, 30K steps). Previous: 24.6%. | **Current mainline** — `evolution_step_jackpot()` in library |
| **SDR density 20% optimal** | 7-density sweep (5–80%): 20% = 22.3% mean / 24.6% peak. 40% second. Very sparse and dense both worse. | **Validated finding** — default confirmed |
| **Ticks=6 optimal** | Tick sweep v2 (4/6/8/12/18): ticks=6 = 24.6% peak. ticks=12 = 23.0%. ticks=4 = 20.1%. | **Validated finding** — default confirmed |
| **Prefill > empty for language** | Prefill: 25.8% peak (167 edges avg). Empty: 23.2% peak. Complex tasks need initial capacity. | **Validated finding** |
| **Addition learning** | 80% on 0-4 + 0-4 from empty network (83 edges). Freq baseline 20%. | **Validated finding** |
| **Empty start >> prefilled (addition)** | 80% with 83 edges vs 64% with 3400 prefilled edges on addition task | **Validated finding** |
| **10×10 addition doesn't scale** | 5×5 = 78% mean. 10×10 ≈ 10% (freq baseline). 100 input combinations too many for sequential processing. | **Validated finding** |
| **W mutation nearly useless** | Adaptive ops test: 0% accept rate for projection mutations across all seeds | **Validated finding** — W reduced to 5% in schedule |
| **Loop mutations** | `mutate_add_loop(len=2/3)` added to library. Critical for sparse/empty network bootstrap. | **Current mainline** — 10% of mutation schedule |
| **Charge pattern similarity** | Trained addition networks: same-sum charge cosine ~0.91, cross-sum ~0.92. Topology is under-differentiated. | **Validated finding** |
| CSR skip-inactive | 8.7x at H=256, 19x at H=512 | **Current mainline** |
| Learnable int8 readout | `Int8Projection` with `raw_scores()` for smooth fitness | **Current mainline** |
| Theta floor / zero-theta collapse | Zero-theta networks collapse into indistinguishable activation patterns | **Validated finding** |
| Chain-50 init | Raises worst-seed floor from 6.5% to 16.1% at H=256 | **Current mainline** for H<512 |

### Python lane: historical peak results

| Finding | Result | Notes |
|---|---|---|
| Breed + crystallize | 24.4% | Consensus structure + pruning (2026-03-29). Rust now exceeds this (25.8%). |
| Learnable channel (C19 Wave Gating) | 23.8% | Cos-shaped LUT, replaces sin/phase/rho |
| Voltage medium leak schedule | 22.11% peak / 21.46% plateau | Fixed schedule, not promoted to defaults |
| Word-pair log-likelihood eval | 23.8% | Task-memory evaluation, not canonical mainline |

### Key architectural findings (cross-lane)

| Finding | Evidence |
|---|---|
| Binary masks sufficient | Binary {0,1} matches ternary accuracy (86.5%). Multiply-free forward pass. |
| Topology > edge precision | Binary edges match float at all tested scales |
| Hub-inhibitor architecture | 10% inhibitory neurons with 2x fan-out (matches FlyWire biological data) |
| Sparse evolution > dense prefill | Evolution with few edges produces targeted circuits; dense prefill = noise |
| Fitness function shape matters | Smooth (continuous) > discrete (step). The #1 bottleneck was not architecture but fitness signal quality. |

## How To Read This Page

- **Current mainline**: shipped in code on `main`, part of `v5.0.0-beta.1`.
- **Validated finding**: experimentally supported, not yet promoted into canonical code path.
- **Experimental branch**: active target or design direction, not yet validated as default.

If code and docs disagree, **code wins for "Current mainline."**

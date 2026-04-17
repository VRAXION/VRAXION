# VRAXION Validated Findings

_Last updated: 2026-04-17_

Canonical evidence summary. Repo-tracked docs are canonical; the GitHub wiki is a mirrored secondary surface.

## Current State

The repo is in a transition state:

- **Released public tag:** `v5.0.0-beta.1` — Rust language-evolution beta
- **Current mainline on `main`:** Rust grower (`instnct-core/examples/neuron_grower.rs`)
- **Reference/support lane:** Python `instnct/model/graph.py`

### Grower lane: proven on `main`

| Finding | Result | Status |
|---|---|---|
| **Bias-free threshold neuron** | Persistent grower now stores/evaluates neurons directly as `dot >= threshold`; redundant bias removed from search and state. | **Current mainline** |
| **Scout oracle** | Mainline parent ranking now uses single-signal scores, connect-all probe weights, and pair-lift shortlist before ternary search. | **Current mainline** |
| **Non-strict accept gate** | Compositional stepping-stones are no longer rejected on equal-val intermediate steps; `four_parity` now reaches 100%. | **Current mainline** |
| **Persistent grower state** | Authoritative resume file is bias-free `state.tsv` with per-step JSON checkpoints. | **Current mainline** |
| **4-bit curriculum entry** | Small exhaustive four-bit pattern tasks solve quickly and act as a compositional smoke floor for the grower. | **Validated finding** |
| **Exact byte readout via latent LUT** | On `1 byte + 4 opcode -> 1 byte`, direct bitbank stalls at `75%`, but the frozen hidden binary latent supports a collision-free LUT translator at `100%` exact over the full `1024`-sample domain. | **Validated finding** |

### Released beta.1 Rust lane: historical peak results

| Finding | Result | Status |
|---|---|---|
| **Smooth cosine-bigram fitness** | 21.7% peak with 1+1 ES (+2.6pp over stepwise argmax) | **Current mainline** — default in `evolve_language.rs` |
| **1+9 jackpot selection** | 24.6% peak (+3.4pp over 1+1 ES). Python parity achieved. | **Current mainline** — `evolution_step_jackpot()` in library |
| **Addition learning** | 80% on 0-4 + 0-4 from empty network (83 edges). Freq baseline 20%. | **Validated finding** |
| **Empty start >> prefilled** | 80% with 83 edges vs 64% with 3400 prefilled edges on addition task | **Validated finding** |
| **W mutation nearly useless** | Adaptive ops test: 0% accept rate for projection mutations across all seeds | **Validated finding** — W reduced to 5% in schedule |
| **Loop mutations** | `mutate_add_loop(len=2/3)` added to library. Critical for sparse/empty network bootstrap. | **Current mainline** — 10% of mutation schedule |
| **Charge pattern similarity** | Trained addition networks: same-sum charge cosine ~0.91, cross-sum ~0.92. Topology is under-differentiated. | **Validated finding** |
| CSR skip-inactive | 8.7x at H=256, 19x at H=512 | **Current mainline** |
| Learnable int8 readout | `Int8Projection` with `raw_scores()` for smooth fitness | **Current mainline** |
| Theta floor / zero-theta collapse | Zero-theta networks collapse into indistinguishable activation patterns | **Validated finding** |
| Chain-50 init | Raises worst-seed floor from 6.5% to 16.1% at H=256 | **Current mainline** for H<512 |

### Python lane: historical reference results

| Finding | Result | Notes |
|---|---|---|
| Breed + crystallize | 24.4% | Consensus structure + pruning (2026-03-29) |
| Learnable channel (C19 Wave Gating) | 23.8% | Cos-shaped LUT, replaces sin/phase/rho |
| Voltage medium leak schedule | 22.11% peak / 21.46% plateau | Fixed schedule, not promoted to defaults |
| Word-pair log-likelihood eval | 23.8% | Task-memory evaluation, not canonical mainline |

### Abstract-core pipeline: named-layer architecture (2026-04-13/14/15)

| Finding | Result | Status |
|---|---|---|
| **L0 Byte Interpreter (LOCKED)** | Flat 8->4 neurons, binary {-1,+1} weights, 36 bits total, 100% round-trip. Pure integer deployment: POPCOUNT -> int32 sum -> int8 output. NO C19 needed. NO float needed. NO multiply (binary = pass or negate). Exhaustive search guaranteed optimal (0.05s). | **CANONICAL / Frozen** |
| **L0 bitwidth sweep** | 1-bit {-1,+1}: 4 neurons, 36 bits, fastest (0.05s); Ternary {-1,0,+1}: 3 neurons, 43 bits, fewest ops; 2-bit {-2..+2}: 2 neurons, 42 bits, fewest neurons (194s exhaustive, 0.7s STE). Winner for deployment: 1-bit (simplest HW, POPCOUNT native). | **CANONICAL / Frozen** |
| **Multi-layer encoder WORSE than flat** | 8->wide->narrow architectures tested: bottleneck effect makes them worse. 8->6->3 = 75 params vs flat 8->4 = 36 params. Multi-layer useful for complex tasks (L1+), counterproductive for byte encoding. | **Validated finding** |
| **Bitflip-only (no sum)** | Without summation, needs 5 output neurons (trivial: wire lower 5 bits). Sum is essential: aggregates multiple bits into richer codes, enabling 4-neuron encoding. | **Validated finding** |
| **Backprop STE matches exhaustive** | Straight-Through Estimator finds identical 100% solutions. 0.7s vs 194s for 2-bit (278x faster). Multiple valid optima exist. Critical for L1+ where exhaustive is impossible. For byte encoder: exhaustive beats STE (0.05s vs 9s) because search space is tiny. | **Validated finding** |
| **Pure integer pipeline** | No C19, no float: POPCOUNT -> integer sum -> int8 output. Binary weight = pass or negate. int32 accumulator -> int8 output. Standard HW support everywhere. | **CANONICAL / Frozen** |
| **Ternary training -> binary deployment** | Train with ternary {-1,0,+1} for sparsity, deploy as binary via add_list + sub_list format (zero-weight edges pruned). | **Validated finding** |
| **L1 Input Merger** | Linear projection 112->96: exact 100% reconstruction at 86% compression. Sigmoid removed (was 99.98% ceiling). Next design target. | **Validated finding** |
| **L2 Feature Extractor** | Conv1D(k=3,f=64)+MLP: 96.6% train (+33pp, beats one-hot 93.8%). Overfits: test 48.5% with 474K params. Needs work. | **Validated finding** |
| **L3 Brain** | INSTNCT sparse spiking network. Not yet built. | **Experimental direction** |
| **Int8 quantization lossless** | +/-0.3% across all layers (L0, L1, L2). Extends preprocessor-only finding to full pipeline. | **Validated finding** |
| **Binary weights: encode yes, predict no** | Binary {-1,+1} achieves 100% on encoding but collapses for prediction. Int8 needed for prediction. | **Validated finding** |
| **ReLU beats C19 in deep networks** | 70% vs 42%. Reversal of shallow-network finding. C19 still superior for shallow/single-layer. | **Validated finding** |
| **C19-mixed best preprocessor activation** | 48.8% accuracy. Softplus 2nd at 45.6%. | **Validated finding** |
| **Minimum model beats INSTNCT** | 1 hidden neuron (487 params) exceeds INSTNCT 24.6% baseline. | **Validated finding** |
| **MLP backprop ~18x more parameter-efficient** | Fair A/B vs INSTNCT evolution at matched param count. | **Validated finding** |
| **ctx=16 optimal** | Diminishing returns beyond ctx=16 on 100KB corpus. | **Validated finding** |
| **Merger scrambles spatial structure** | Conv accuracy drops -7pp through merger. Conv should bypass merger for spatial input. | **Validated finding** |
| **Ternary sparse list unifies with INSTNCT** | Ternary -> sparse list format (neuron_id, input_id, weight) maps directly to ConnectionGraph. | **Validated finding** |
| **Topology carries intelligence** | In sparse networks, which bits connect to which neurons determines separability, not weight magnitude. INSTNCT philosophy validated at L0. | **Validated finding** |

### Key architectural findings (cross-lane)

| Finding | Evidence |
|---|---|
| Binary masks sufficient | Binary {0,1} matches ternary accuracy (86.5%). Multiply-free forward pass. |
| Topology > edge precision | Binary edges match float at all tested scales |
| Hub-inhibitor architecture | 10% inhibitory neurons with 2x fan-out (matches FlyWire biological data) |
| Sparse evolution > dense prefill | Evolution with few edges produces targeted circuits; dense prefill = noise |
| Fitness function shape matters | Smooth (continuous) > discrete (step). The #1 bottleneck was not architecture but fitness signal quality. |

### Beukers gate + quantization (2026-04-16/17)

Char-LM task (FineWeb 30MB). Neuron activations and weight representations explored systematically against the B0 Beukers baseline.

| Finding | Result | Status |
|---|---|---|
| **B0 Beukers gate = optimal char-LM activation** | `output = ab / (1 + |ab|)` 2-projection gate. Baseline 74.03 +/- 1.05 on FineWeb 30MB, nf=128, 3 seeds. Every one of 15+ variants tested LOST to B0. Multiplication detects correlation (bigram patterns) which is the core char-LM signal — B0 is near-optimal here. | **Validated finding** |
| **Beukers variant sweep: B0 wins** | branch-budget (ab vs c) 73.07 +/- 0.17; 3-tower sum-of-products 72.13 +/- 0.29; alpha-modulated Beukers 72.10 +/- 0.22; delta group-norm / alpha+delta combos 71.60-73.47; margin-gate (sort-based) 73.57 +/- 0.29 (deterministic probe showed outlier-detector behavior, not general); range/hybrid gates 69-73; sharpen variants (m-space + raw) 67-70; residual Beukers (bounded + unbounded + learnable) 73.17-73.77. | **Validated finding** |
| **Staged INQ int4 BEATS float32** | nf=64 B0 Beukers on FineWeb 30MB, seed=42. float32 baseline 66.50%; one-shot PTQ int4 64.30% (-1.80pp); one-shot QAT int4 62.80% (-3.30pp); **staged INQ int4 at 100% quantized = 67.90% (+1.40pp)** — CHAMPION. Round-by-round 67-69% (stable regularization effect). Mechanism: acts as weight regularization during retraining. Zhou et al. 2017 INQ (gradual easiest-to-grid-first). | **Validated finding** |
| **Int4 = 8x smaller, matching-or-better accuracy** | 16k params: 64 KB (float32) -> 8 KB (int4), with staged INQ matching or beating full-precision on char-LM. | **Validated finding** |
| **Ternary/binary staged INQ: floor at small scale** | nf=64, 100% quantized: ternary 41.20% (-25.30pp from float32), binary 46.70% (-19.80pp). Sweet spot: ternary at 50% frozen = 68.70% (best of all tested configs). At nf=64 char-LM, ternary/binary sit below a ~47% floor. BitNet b1.58 (Microsoft 2024) suggests this floor disappears at 3B+ params; on our small model the floor is real. Mixed-precision (some ternary, some int4/8) is the practical win. | **Validated finding** |

## How To Read This Page

- **Current mainline**: shipped in code on `main` right now.
- **Validated finding**: experimentally supported, not yet promoted into canonical code path.
- **Experimental branch**: active target or design direction, not yet validated as default.

If code and docs disagree, **code wins for "Current mainline."**

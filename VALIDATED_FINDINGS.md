# VRAXION Validated Findings

_Last updated: 2026-04-30_

Canonical evidence summary. Repo-tracked docs are canonical; the GitHub wiki is a mirrored secondary surface.

## Current State

The repo is in a transition state:

- **Released public tag:** `v5.0.0-beta.9` — Phase D10u state-anchored wiring search, `top_01` release-candidate research checkpoint, 30/30 pass adversarial 16k gate (`v5.0.0-beta.8` is the prior Phase D9.2 multi-objective confirmation + `seed2042_improved_generalist_v1` validated H=384 research checkpoint; `v5.0.0-beta.7` is the prior Phase D9 smooth+accuracy specialist checkpoint; `v5.0.0-beta.6` is the prior Phase D6/D7/D8 research-line checkpoint + wiki/HTML doc-drift correction patch; `v5.0.0-beta.5` is the prior Phase D3/D3.1/D4 SAF lock + 56-example archive cleanup release; `v5.0.0-beta.1` remains as the original language-evolution beta, historical reference only)
- **Current mainline on `main`:** Rust grower (`instnct-core/examples/neuron_grower.rs`)
- **Python deploy SDK:** `Python/` — Block A + B, pure numpy (256/256 + 65536/65536 lossless)
- **Historical Python research lane:** frozen at tag `archives/python-research-20260420` (was `instnct/`, migrated to Rust `instnct-core/` on 2026-04-13)
- **Active research line:** Byte-level lexical-to-neural pipeline (L0 + L1 frozen; Tokenizer V2 champion; Embedder + Nano Brain scaffolds awaiting training)

### Grower lane: proven on `main`

| Finding | Result | Status |
|---|---|---|
| **Bias-free threshold neuron** | Persistent grower now stores/evaluates neurons directly as `dot >= threshold`; redundant bias removed from search and state. | **Current mainline** |
| **Scout oracle** | Mainline parent ranking now uses single-signal scores, connect-all probe weights, and pair-lift shortlist before ternary search. | **Current mainline** |
| **Non-strict accept gate** | Compositional stepping-stones are no longer rejected on equal-val intermediate steps; `four_parity` now reaches 100%. | **Current mainline** |
| **Persistent grower state** | Authoritative resume file is bias-free `state.tsv` with per-step JSON checkpoints. | **Current mainline** |
| **4-bit curriculum entry** | Small exhaustive four-bit pattern tasks solve quickly and act as a compositional smoke floor for the grower. | **Validated finding** |
| **Exact byte readout via latent LUT** | On `1 byte + 4 opcode -> 1 byte`, direct bitbank stalls at `75%`, but the frozen hidden binary latent supports a collision-free LUT translator at `100%` exact over the full `1024`-sample domain. | **Validated finding** |

### Phase D9.2 direct-genome H=384 research checkpoint (2026-04-29)

| Finding | Result | Status |
|---|---|---|
| **D9.2 multi-objective generalist confirmation** | `seed2042_improved_generalist_v1` retains the D9 smooth/accuracy gains, remains echo-safe, and flips unigram from a regression into a positive baseline-relative delta on fresh seeds. D9.2b top_01 gates at eval_len 16000: smooth lower95 +0.016916652, accuracy lower95 +0.003755206, echo lower95/upper95 near zero, unigram lower95 +0.005150928. | **Validated finding (H=384 research checkpoint)** |
| **D9.2 no-overclaim boundary** | The checkpoint is a validated direct-genome H=384 research artifact, not a replacement for the current public mainline grower. Stable local artifact path: `output/releases/v5.0.0-beta.8/seed2042_improved_generalist_v1.ckpt`; SHA256 `D63200504C5B4A6EA2134FD26E3E3D7CB75FF05884236DE2CC6E206BB4BA8D54`. | **Validated finding (scope boundary)** |

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
| **Staged INQ int4 BEATS float32** ⚠️ **SUPERSEDED 2026-04-18 — see revised section below** | ~~nf=64 B0 Beukers on FineWeb 30MB, seed=42. float32 baseline 66.50%; one-shot PTQ int4 64.30% (-1.80pp); one-shot QAT int4 62.80% (-3.30pp); **staged INQ int4 at 100% quantized = 67.90% (+1.40pp)** — CHAMPION.~~ **Revised 2026-04-18**: this +1.40pp "win" is a protocol artifact. Staged INQ gives 200 extra training epochs to quantized runs vs float baseline. Matched-epoch float on same task reaches equal or higher accuracy. See "Quantization championship (2026-04-17/18)" section below for full control experiments. | **Superseded — protocol artifact** |
| **Int4 = 8x smaller, matching-or-better accuracy** | 16k params: 64 KB (float32) -> 8 KB (int4), with staged INQ matching or beating full-precision on char-LM. | **Validated finding** |
| **Ternary/binary staged INQ: floor at small scale** ⚠️ **PARTIALLY SUPERSEDED 2026-04-18** | ~~At nf=64 char-LM, ternary/binary sit below a ~47% floor.~~ **Revised 2026-04-18**: the floor was a protocol + capacity combination, not fundamental. (1) QAT STE ternary at same scale lifts ternary to match binary (+16.5pp fix). (2) Binary at nf=1024 reaches 70.70-71.50% — BitNet b1.58's "scale solves it" reconfirmed. See "Quantization championship" section below. | **Superseded — protocol + capacity** |

## Quantization championship (2026-04-17/18) — revised final story

Comprehensive quantization sweep on FineWeb char-LM (B0 Beukers gate), RTX 4070 Ti Super. 50+ runs total, ~2.5h wallclock across 3 major sweeps (CPU 18-run nf=32/64/96/128, GPU 8-run nf=1024 across 4 modes, GPU mid 6-run int5/int8/fp16) plus 4 control experiments. Earlier "int4 +1.4pp beats float" claim was shown to be a protocol artifact (extra training epochs given to quantized runs); full control experiments below supersede it.

| Finding | Result | Status |
|---|---|---|
| **QAT int8 = new absolute champion (nf=1024)** | FineWeb char-LM, nf=1024: QAT int8 (STE) = **86.40% eval** — beats pure float_long 86.20% (400 ep, no quantization). Staged INQ int8 = 85.20%. QAT int8 is now the recommended cloud/server default (essentially lossless at 4x compression). | **Validated finding** |
| **Staged INQ int4 = Pareto sweet spot** | nf=1024 FineWeb: staged INQ int4 = **84.75%**, just -1.65pp from pure float (86.20%) at **8x compression**. QAT int4 = 84.50%. Mobile/edge deployment recommendation. | **Validated finding** |
| **QAT binary/ternary tied at nf=1024** | QAT STE binary = 71.50%, QAT STE ternary = 71.50% (tied). Staged binary = 70.70%. At nf=1024 binary reaches ~71% — this is a **capacity ceiling, not an info-ceiling**. Confirms BitNet b1.58 literature: binary is viable at scale. IoT/FPGA deployment: QAT binary + Beukers LUT = 32x compression, -14.9pp, native bit-ops. | **Validated finding** |
| **Earlier "int4 +1.4pp win over float" = PROTOCOL ARTIFACT** | The 2026-04-17 staged-INQ int4 +1.40pp result was caused by the staged protocol giving quantized runs extra training epochs. Control: float trained for 400 ep (no quantization) = 86.20%; same-epoch float_staged = 84.50%. The "win" vanishes under matched compute. | **Validated (revised) — supersedes prior "+1.4pp int4 beats float32" row above** |
| **"Staged ternary is fundamentally bad" = PROTOCOL BUG, not fundamental** | Staged ternary at nf=1024 = 55.00%, but QAT STE ternary on same nf=1024 = 71.50% (+16.5pp). The staged protocol's scale/2 threshold over-prunes ternary weights. Ternary is NOT fundamentally worse than binary — the earlier protocol was broken. | **Validated (revised)** |
| **"Binary info-ceiling at 49% FineWeb" = capacity, not information** | Earlier small-scale result (nf=64 binary = 46.70%) was capacity-bound. At nf=1024, binary reaches 70.70% (staged) / 71.50% (QAT). Confirms BitNet b1.58: binary viability scales with width. | **Validated (revised)** |
| **Progressive growing + per-neuron int4 quant = FAILS** | FineWeb nf=128, progressive per-neuron int4 grow-and-quant = 63.35% vs batch_float nf=128 = 78.20% (**-14.85pp**). Greedy lock-in of quantized neurons dominates the loss landscape. Dense training + post-hoc quantization wins. | **Validated finding (negative)** |
| **Random-rotation sparse training = DOMINATED** | Best rotation variant (50% hot) = 82.75% FineWeb vs QAT int4 = 84.50%. Sparse-rotation training loses on accuracy at matched memory. | **Validated finding (negative)** |
| **Generational cluster growth = DOMINATED** | Gen 1 (256) -> Gen 2 (+256) -> Gen 3 (+256) sequential cluster growth = 79.50% FineWeb vs single-shot nf=768 = 84.70% (**-5.20pp**). Sequential grow-then-freeze underperforms joint training. | **Validated finding (negative)** |
| **Stacked exhaustive clusters (D=14 random subsets, ternary)** | 50 clusters = 26.55% vs float+PTQ same-D = 32.20%. Dominated in accuracy AND memory AND time. Exhaustive cluster stacking does not scale. | **Validated finding (negative)** |
| **True ternary exhaustive D=16 = mathematical optimum, limited capacity** | True ternary exhaustive D=16 = 21.25% vs float same-D = 30.25% (-9pp gap). Mathematically optimal over the D=16 search space but capacity-limited. Useful only for micro-components (D <= 20, byte-encoder scale) where optimality guarantee matters. | **Validated finding** |
| **Big Beukers-cluster joint exhaustive = stalls** | 15 clusters x 32 sec each -> 22.40%. Same best config found 5 times (search stalls early). Exhaustive joint over Beukers-cluster space does not scale past micro-components. | **Validated finding (negative)** |
| **Revised per-precision deploy recommendations** | **Cloud/server**: QAT int8 = 86.40%, 4x compression, essentially lossless. **Mobile/edge**: staged INQ int4 = 84.75%, 8x compression, -1.65pp. **IoT/FPGA**: QAT binary + Beukers LUT = 71.50%, 32x compression, -14.9pp, native bit-ops. **Micro-components only** (D <= 20): true ternary exhaustive — mathematical optimum guarantee, capacity-limited. | **CANONICAL / Frozen** |
| **Meta-lesson: gradient + post-hoc quantization is Pareto-dominant** | Gradient-based dense training + post-training quantization (QAT STE for int8/int4, or staged INQ for int4) remains the Pareto-dominant approach at all scales tested. Exhaustive search is valuable only for micro-components (D <= 20) where mathematical optimality matters AND the limited capacity is acceptable. Clustered / progressive / rotated sparse approaches were dominated in every metric. The earlier project finding "MLP backprop ~18x more parameter-efficient than INSTNCT evolution" is now reconfirmed on modern quantization techniques: **gradient wins**. Final verdict visualization: `docs/playground/quant_final_verdict.html`. | **Validated finding / meta** |

## Byte-level lexical-to-neural pipeline (2026-04-17/18/19)

Parallel research line to the Rust grower. Builds a compact byte→text→tokens→embedding→brain stack with each layer independently validated. Frozen champion artifacts are committed; each step is reproducible from `tools/diag_*`.

| Finding | Result | Status |
|---|---|---|
| **L0 Byte Unit — LOCKED** | C19 `8 → 24 → 16` tied-mirror autoencoder, int4 precision, 100% lossless on all 256 bytes. Deploy form: 256-entry LUT at `tools/byte_embedder_lut.h` (4.1 KB). Input: 1 byte (8 bits). Output: 16-dim embedding. | **Validated finding / Frozen** |
| **L0 alternative champion — binary + C19 + H=16 (2026-04-19)** | Exhaustive activation-precision sweep confirmed binary weights with C19 activation reach 100% exact lossless at hidden width H=16 — the smallest H across all tested (precision, activation) pairs. Artifacts: `output/byte_unit_champion_binary_c19_h16/` (6.5 KB weights JSON, 4 KB raw int8 LUT, 30 KB C header). Sweep matrix surfaces: tanh + 2-bit @ H=12 (smallest for 2-bit); tanh + ternary @ H=32; C19 + binary @ H=16 (smallest overall). Weight-reload and LUT-based round-trip both 256/256. | **Validated finding / Alternative champion** |
| **L1 Byte-Pair Merger — CHAMPION** | Single-W mirror-tied autoencoder (one 32×81 matrix, 2,592 weight cells). 100% lossless on all 65,536 byte pairs. Huffman-packed deploy: **3,440 B (3.36 KB)** at `output/merger_single_w_huffman_pack/packed_model.bin`. Progression: fp32 11.20 KB → fp16 5.60 KB → Huffman-packed 3.36 KB. Standard compressors (lzma/bz2/gzip on raw fp16) all beaten by the custom structured encoding. Shannon floor: 2,422 B (~42% gap remains — next target). | **Validated finding / Frozen champion** |
| **L1 Byte-Pair Merger — native 7-bit candidate (2026-04-19)** | Identity (linear) autoencoder at H=120 with 7-bit integer weights reaches 100% exact lossless. Native footprint: `3,840 × 7 bit (W) + 152 × 3 bit (b1+b2) + 4 B (fp32 alpha) = 3,421 B (3.34 KB)` — ~0.55% smaller than the Huffman champion and **native** (no Huffman decode step required). Lossless at seeds 7 and 42; robust across 4 seeds tested. Confirmed negative results (autonomous compression loop, 2026-04-19): binary / ternary / 2-bit / 3-bit single-W and dual-W **cannot represent the merger solution** (bake probe ceiling 0.25–29% — representational, not optimization); alpha scaling **cannot be eliminated** (seed 42 → 65,480 bad without alpha); C19 aux params (c, rho, biases) **cannot be post-hoc quantized at int8** even on the float exact model. Scripts: `tools/diag_byte_pair_merger_widen_sweep.py`, `tools/diag_byte_pair_merger_bake_probe.py`, `tools/diag_byte_pair_merger_aux_quant_probe.py`, `tools/diag_byte_pair_merger_alpha_ablation.py`. Full findings: `docs/wiki/COMPRESSION_LOOP.md`. | **Validated finding / Native alternative** |
| **L1 research-side "73% hard ceiling" was a single-seed artifact** | Cluster 10's "tied-mirror cannot exceed 73% byte recovery" ceiling was overturned when single-W mirror tied at H=81 reached 100% lossless on the same 65,536 byte-pair set. The ceiling was a protocol/seed artifact, not fundamental. Overrides the earlier Cluster 10 finding in the archived timeline. | **Validated (revised)** |
| **Word Tokenizer V2 hybrid champion** | Whole-word + subword + byte-fallback, `whole_ratio=0.9375`, 32,294 vocab. 10 MB FineWeb-EDU: **30.43% real Huffman compression** of raw (beats gzip-9 37.62% by 7.19pp; 0.46pp above bzip2-9 29.97%; 1.82pp above lzma-9e 28.61%). Byte-fallback only 1.26% of input bytes; LEARNED coverage 95.90%. Shannon floor 30.34% (Huffman 0.29% above — near-optimal). 100% lossless round-trip on 10 MB, 0/2000 unreachable tokens, 14/14 adversarial edge cases pass. Matches SuperBPE τ=0.9 ([arXiv:2503.13423](https://arxiv.org/abs/2503.13423)). Frozen public artifact: `output/word_tokenizer_champion/`. | **Validated finding / Frozen champion** |
| **Research-swarm pipeline verdict** | Parallel deep-research against 2024-2026 tokenizer literature confirms our pipeline (scan → pre-segment word/punct/ws → DP per word → byte fallback) is a legitimate 2025-frontier hybrid: matches tiktoken pre-segmentation + SentencePiece Unigram DP + SentencePiece byte_fallback. SuperBPE τ=0.9 aligns with our `whole_ratio=0.9375`. Not a published "tokenize + Huffman/rANS only" pipeline — potential contribution gap identified. | **Validated (research-anchored)** |
| **Word Embedder V1 — SCAFFOLD** | 32,294 × 64 Xavier-init lookup table, 2,066,816 params (8.27 MB f32 / 2.07 MB int8). Symmetric per-tensor int8 quant ~0.0012 mean dequant error. Dim 64 chosen to match tiny-champion philosophy (L0=16, L1=81). Forward-pass verified; training not started. | **Scaffold — untrained** |
| **Nano Brain V1 — SCAFFOLD** | 2-layer causal transformer, 64 dim, 4 heads, FFN 64→256→64 GELU, tied embedder/output head. Total 2,182,144 params (94.7% in embedder). End-to-end forward pass `text → IDs → [N, 64] → logits [N, 32294]` verified (81 ms CPU on 10 tokens). Init cross-entropy 11.15 vs random-uniform 10.38 baseline (expected noise). | **Scaffold — untrained** |
| **Block C byte-pair embedder — CHAMPION (2026-04-21)** | E=32 per-pair lookup trained by full-softmax next-pair CE on 100 MB FineWeb-EDU (L4 GPU, Modal). Three-seed fine-tune from warm-start (ep3 of earlier LR=0.1 diverging run) + LR=0.03 + cosine decay → **acc@1 34.06 ± 0.82%** over 10 epochs, converged without divergence. Intelligent quantization (per-channel int4 α=0.5 on 3,386 hot rows + shared OOV for 62,150 cold rows) packs the full table to **62,528 B (61 KB) — 134× compression** vs float32 (8.39 MB). Preserves 74.4% top-5 cluster overlap with float reference among hot pairs. Deploy champion: `output/block_c_bytepair_champion/packed.bin` + meta.json. Python deploy SDK: `Python/block_c_embedder/embedder.py`, zero ML deps. Chain test A+B+C (10 invariants) passes. Empirically learned clusters: word-start equivalence (`' t'` = `'\nt'` = `'(t'` = `'-t'`), case-invariance (`th`↔`Th`, `he`↔`He`, `in`↔`In`, `on`↔`On`, `an`↔`An`), sentence-terminator (`'. '`/`'! '`/`'? '`/`'.\n'`), clause-punct (`', '`/`'; '`/`': '`), function-word groups (`in`/`by`/`on`). Total Modal cost: $9.11 (PoC v1 $4.07 learning + PoC v2 $5.04 champion). | **Validated finding / Frozen champion** |
| **L2 reconstruction merger — DEPRIORITIZED** | Attempted to compress 16-byte windows (8 × 81-dim L1 hiddens = 648-dim) with a second mirror-tied autoencoder. Phase-0 PCA geometry probe: at D=128 only 97.6% per-dim sign-match / 2.6% exact-window. Neural tied-mirror ablation under-fit even the linear PCA baseline. Geometry of L1 hidden space is anisotropic on natural text; linear reconstruction does not scale within current capacity. Pivoted to word-tokenizer line (Cluster 16). | **Validated finding (negative) — direction closed** |

### ABC-to-Brain integration + fitness function sweep + ablation (2026-04-21/22)

Block C byte-pair champion wired into the INSTNCT brain for the first time. Fitness function sweep, edge weight experiments, multi-channel experiments, crystallize port, and ablation study.

| Finding | Result | Status |
|---|---|---|
| **ABC-to-Brain char-level integration** | C-embedding matches SdrTable at ~25% accuracy (multi-seed validated). First end-to-end wiring of the abstract-core character embedding into the INSTNCT brain. Confirms embedding quality is not the bottleneck — brain topology is. | **Validated finding** |
| **Byte-pair prediction (397 class)** | 7.1% peak with smooth cosine fitness (frequency baseline 4.2%). On a much harder 397-class byte-pair prediction task, the brain exceeds frequency baseline by +2.9pp. Signal is real but weak — brain topology needs structural improvement. | **Validated finding** |
| **Fitness function sweep: smooth linear cosine champion** | 10 fitness function variants tested. Smooth linear cosine is the champion — provides the best gradient signal for mutation-selection. Stepwise / argmax / pure-accuracy variants all dominated. | **Validated finding** |
| **Edge weights [1-3]: worse than binary** | Weighted edges (integer weights 1-3) degrade performance vs binary {0,1} masks. Signal-to-noise ratio degrades when edge precision increases — extra weight values create noise that mutation-selection cannot navigate. Reconfirms topology > edge precision. | **Validated finding (negative)** |
| **Multi-channel input: worse than single** | Dual-input and multi-channel injection schemes tested. All worse than single-channel input. Dimension curse: extra input channels increase the search space faster than they add useful signal for mutation-selection. | **Validated finding (negative)** |
| **Crystallize grow-prune cycles: validated** | Crystallize (iterative grow-prune-regrow cycles to find minimal topology) ported from Python to Rust and validated. Prune-regrow cycles converge to compact circuits. | **Validated finding** |
| **Ablation study: single-attractor collapse** | Systematic ablation reveals 7 dominant neurons form a bottleneck — single-attractor topology collapse. The brain converges to one dominant attractor basin instead of developing multiple competing pathways. This is the core pathology limiting current accuracy. | **Validated finding** |

## Phase D9 direct-genome landscape + smooth+accuracy specialist (2026-04-28/29)

Eight-run audit (D9.0n → D9.0o → D9.0q → D9.0r → D9.0s → D9.0v → D9.0w → D9.0x → D9.0y → D9.0z → D9.1) of the direct-genome landscape around H=384 seed2042 on the `evolve_mutual_inhibition` substrate, using paratrooper deepening, endpoint export+replay, basin-topology audit, linear-bridge falsification, perturbation-overlap probing, and a 30-seed × 3-eval-length robustness suite.

**No-overclaim scope.** The D9 saga produces one validated specialist checkpoint, `seed2042_improved_v1`, that improves the smooth metric and paired direct accuracy at H=384 seed2042 but **regresses unigram** behavior. It is NOT a general-purpose mainline replacement and does NOT supersede the current `neuron_grower.rs` mainline path on `main`.

| Finding | Result | Status |
|---|---|---|
| **D9 direct-genome landscape audit framework** | 600 rows × 1 base × radii {1, 4, 16} × {channel, edge, polarity, threshold} × 50 samples/cell. Type-split verdict: `channel`/`polarity` cliffy, `edge`/`threshold` rugged. Visual atlas (`direct_landscape_atlas.html`, `sphere_landscape.html`, `sphere_tiles.csv`) + per-radius heatmaps. D9.0b freezes projection and mutates only the persisted core genome. Doc: `docs/research/PHASE_D9_DIRECT_GENOME_LANDSCAPE_AUDIT.md`. | **Validated finding** |
| **D9.0n/o/q paratrooper climb at H=384 seed2042** | D9.0n 100-step `BASIN_CONFIRMED` → D9.0o 200-step `MOUNTAIN_CONFIRMED` → D9.0q 300-step still climbing past 200 steps. Mountain-renderer states (`BASIN_CONFIRMED`, `DEEP_BASIN_CONFIRMED`, `MOUNTAIN_CONFIRMED`) drive the progressive planet visualization. | **Validated finding** |
| **D9.0r endpoint export + replay** | 8 endpoints exported across 4 source tiles. **7/8** retain ≥70% of the climb-time gain on fresh evaluation; **8/8** are positive vs the seed2042 baseline. Endpoints `11_16_endpoint_01`, `12_29_endpoint_04`, `9_26_endpoint_03` carried into D9.0x. | **Validated finding** |
| **D9.0s basin topology** | Pairwise genome-distance ratio between the three positive-island tiles measured at **1.019** — essentially equidistant in genome space, not nested. | **Validated finding** |
| **D9.0v linear-bridge falsification** | 100 interior-point evaluations along the straight genome-space line between two positive endpoints: **0/100** interior points positive. The basin is **not** a connected ridge; the straight-bridge hypothesis is falsified. | **Validated finding (negative)** |
| **D9.0w 3 independent islands at H=384 seed2042** | Per-tile perturbation clouds do not touch in genome space. Three independent local optima are confirmed; they are separated islands, not a connected basin. | **Validated finding** |
| **D9.0x endpoint robustness suite** | 30 fresh seeds × 3 eval lengths (1000/4000/16000), paired vs seed2042 baseline. Verdict: `D9_ENDPOINT_ROBUSTNESS_STRICT_PASS`. Top endpoint `11_16_endpoint_01` overall lower95 = **+0.01666** (n=90, mean +0.01699, positive_rate 100.0%, min_lower95_across_eval_lens +0.01581). | **Validated finding** |
| **`seed2042_improved_v1` smooth specialist (D9.0y)** | Exported H=384 candidate from `11_16_endpoint_01`, re-evaluates +0.01733 mean (lower95 +0.01714) over n=60 fresh-seed paired runs vs the seed2042 baseline. Doc: `docs/research/PHASE_D9_PRODUCTION_CANDIDATE.md`. Smooth-metric scope only at this verdict. | **Validated finding (specialist — smooth)** |
| **Multi-task limit (D9.0z generalization gate)** | Verdict `D9_SMOOTH_ONLY_WIN`. Per-metric paired deltas (n=60 each): smooth +0.01733, accuracy +0.00455 (positive_rate 95.0%), echo −0.00009 (neutral), unigram **−0.00882** (positive_rate 0.0%, regression). Decision: keep as smooth/accuracy specialist; do NOT promote as broad generalization checkpoint. Doc: `docs/research/PHASE_D9_GENERALIZATION_GATE.md`. | **Validated finding (scope-limit)** |
| **Local repair limit (D9.1)** | Local repair microprobe + confirm produce only **weak** repair and cannot bridge the unigram task gap from this candidate. The smooth+accuracy gain does not generalize to unigram by local search. Docs: `docs/research/PHASE_D9_1_REPAIR_MICROPROBE.md`, `docs/research/PHASE_D9_1_REPAIR_CONFIRM.md`. | **Validated finding (negative)** |
| **D9.0d progressive planet renderer (5-tier basin hierarchy)** | Visualization tooling under `tools/d9_0d_progressive_planet/` with a 5-tier basin/mountain hierarchy and the mountain-renderer state machine used by D9.0n/o/q. Renderer is owned by a parallel research line; the artifact landed at the same release boundary as the D9.0r-D9.0z verdicts. | **Validated finding (tooling)** |

## Phase D10u state-anchored wiring search — release-candidate research checkpoint (2026-04-30)

D10u pivots from "can we promote beta.8?" to "search for candidates that pass the D10r-v8 state-identity artifact gate from the start." The causal mechanism is `edge_threshold_coadaptation`, first confirmed at D9.4b. The top_01 candidate from `seed_2042` under the `edge_threshold_coadapted` arm is the first checkpoint to pass all artifact controls at promotion-grade eval.

**No-overclaim scope.** `top_01` is a release-candidate research checkpoint at H=384 only, on the `seed_2042` baseline only. It is NOT a public mainline grower replacement, does NOT constitute cross-architecture or cross-seed replication, and does NOT unlock H512 or universality claims. CPU/Rust cross-check is the recommended next gate before final ship.

| Finding | Result | Status |
|---|---|---|
| **D10u state-anchored search machinery** | Short scout (eval_len=128, 3 seeds, 3 checkpoints × 3 arms): verdict `D10S_NO_TRUSTED_SIGNAL` at scout budget. Seed2042 `edge_threshold_coadapted` arm produced the best weak-state-anchored signal. Ladder follow-up (eval_len=128, 3 rounds, 6 proposals/round) promotes a `STRICT_TRUSTED` scout candidate from seed_2042. Doc: `docs/research/PHASE_D10U_STATE_ANCHORED_WIRING_PRIOR.md`. | **Validated finding (tooling + scout)** |
| **D10u bounded D10r-v8 confirm (eval_len=1000)** | `top_01_seed_2042_edge_threshold_coadapted.ckpt` vs seed_2042 D7 H=384 baseline, 4 eval seeds, verdict `D10R_V8_STATE_IDENTITY_PASS`. real MO delta CI low +0.184054; trusted MO CI low +0.170111. All state-identity diagnostics clean (zero projection-consistent drift, 0 duplicate/similar rows, all shuffle bounds pass). Reopens the release-candidate path. | **Validated finding** |
| **D10u longer D10r-v8 confirm (eval_len=4000)** | 4 eval seeds, verdict `D10R_V8_STATE_IDENTITY_PASS`. real MO delta CI low +0.185742; trusted MO CI low +0.131453. `state_shuffle_shared` — the main beta.8 blocker — does not block this candidate. | **Validated finding** |
| **D10u promotion-grade 16k / 30-seed confirm** | `top_01` vs seed_2042 baseline, eval_len=16000, eval seeds 970101..970130, sharded fail-stop runner. Verdict: `D10U_TOP01_16K_SHARDED_PASS`. 30/30 shards complete, 0 fail, no blocking control families. Minimum trusted MO CI low: **+0.084493**. Minimum real MO delta CI low: **+0.178087**. Weakest shard: shard_24 / seed 970125 (trusted_mo_ci_low +0.084493). Decision: `D10U_TOP01_RELEASE_CANDIDATE_RESEARCH_CHECKPOINT`. | **Validated finding (release-candidate research checkpoint)** |
| **D10u all 4 tasks pass** | At promotion-grade eval, all four task gates pass: smooth, accuracy, echo, unigram. The unigram regression that scoped beta.7 to "specialist" is resolved in `top_01`. | **Validated finding** |
| **D10u causal mechanism** | Causal explanation is `edge_threshold_coadaptation`, same mechanism confirmed at D9.4b on the beta.8 generalist. Projection, channel, and polarity are unchanged; the improvement is a co-adapted edge-wiring plus threshold-timing package. | **Validated finding** |
| **D10u no-overclaim boundary** | `top_01` is a validated H=384 research checkpoint on `seed_2042` only. Limitations: H=384 only, seed_2042 baseline only, replication across seeds/H not yet established. Pending next gate: CPU/Rust cross-check before any final ship decision. | **Validated finding (scope boundary)** |

## How To Read This Page

- **Current mainline**: shipped in code on `main` right now.
- **Validated finding**: experimentally supported, not yet promoted into canonical code path.
- **Experimental branch**: active target or design direction, not yet validated as default.

If code and docs disagree, **code wins for "Current mainline."**

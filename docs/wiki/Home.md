# Vraxion

> **Vraxion** /vræk.ʃən/ ("VRAK-shun") · **INSTNCT** /ˈɪnstɪŋkt/ ("instinct")

<p align="center">
  <img src="https://raw.githubusercontent.com/VRAXION/VRAXION/main/docs/assets/vraxion-home-hero.jpg" alt="Vraxion front-door illustration" width="740">
  <br>
  <em>The engineering of the "I"</em>
</p>

Vraxion is building **INSTNCT**: a gradient-free self-wiring architecture that learns by changing its own directed graph instead of training a fixed topology with backpropagation.

> **Core thesis**
>
> **Inference emerges as the fixed point of destructive interference.**
>
> Signal enters a structured recurrent substrate, incompatible propagation paths cancel through destructive interference, and the surviving pattern — the fixed point — is read out as inference. This generalizes the older loop-era framing without claiming that every result on `main` has already proved the full thesis.

Vraxion exists to advance machine consciousness as an engineering reality. In public technical terms, that means building systems that can be instrumented, checked, refined, and argued about without collapsing theory, implementation, and evidence into one blur.

This page is the mission-first technical front door. Use [Pages](https://vraxion.github.io/VRAXION/) for the polished front door — the Blocks ladder ([A Byte Unit](https://vraxion.github.io/VRAXION/blocks/a-byte-unit.html) → [E Brain](https://vraxion.github.io/VRAXION/blocks/e-brain.html)) plus a Legacy detail view. Use the repo [README.md](https://github.com/VRAXION/VRAXION/blob/main/README.md) for the code-facing entry, and [Research Process & Archive](Timeline-Archive) for protocol, chronology, and proof trail.

## Current D10 Finding

As of 2026-04-30, the HDS basin mapping side quest is closed. The H384 D9/D10
landscape contains real local signal, but not a broad universal release-ready
basin. The beta.8 `seed2042_improved_generalist_v1` checkpoint remains a real
research finding explained by edge + threshold co-adaptation, but D10r-v8
blocked its release path because `state_shuffle_shared` can beat the real
signal.

The current path is **D10u state-anchored candidate search**: keep the useful
edge+threshold signal, but require candidates to beat artifact controls during
search. A short D10u scout found weak state-anchored signal on seed4042, but no
near/strict trusted release candidate yet. See [D10 HDS Basin & State Identity
Finding](D10-HDS-Basin-State-Identity-Finding) and [D10 Release Readiness
Gate](D10-Release-Readiness-Gate).

## At a Glance

- **Architecture line:** `INSTNCT`
- **Stable public release:** [`v5.0.0-beta.6`](https://github.com/VRAXION/VRAXION/releases/tag/v5.0.0-beta.6) is the current Rust public beta tag (Phase D6/D7/D8 research-line checkpoint + doc-drift correction patch on top of the beta.5 SAF lock); see the [release list](https://github.com/VRAXION/VRAXION/releases) for the full beta.1 → beta.6 chain. The legacy `v4.x` Python lane is preserved for context only.
- **Public research lane:** English remains the first-class public lane; task-memory and GPU stay secondary validation surfaces.
- **Implementation momentum:** Rust is the primary implementation surface, and the current mainline path on `main` is the bias-free threshold grower (`instnct-core/examples/neuron_grower.rs`). The released language-evolution beta still records its **24.6% peak** smooth-fitness + 1+9 jackpot result as the historical Rust language line, but the active grower lane has moved past it: see [`docs/PUBLIC_BETA_TRAINING.md`](https://github.com/VRAXION/VRAXION/blob/main/docs/PUBLIC_BETA_TRAINING.md) for the canonical training runbook and [`docs/GROWER_RUN_CONTRACT.md`](https://github.com/VRAXION/VRAXION/blob/main/docs/GROWER_RUN_CONTRACT.md) for the frozen B0 engine contract.
- **Current frontier:** The **byte-level pipeline** is the active project line as of 2026-04-19. L0 Byte Unit and L1 Byte-Pair Merger are both 100% lossless with shipped deploy artifacts. The Brain-side lexical-to-neural bridge (Cluster 16, 2026-04-19) is now scaffolded end-to-end: **Word Tokenizer V2 hybrid champion** (PR #130, 30.43% real Huffman compression on 10 MB FineWeb-EDU — 0.46pp above bzip2, 7.19pp below gzip), **Word Embedder V1** (PR #131, 32,294 × 64 random-init lookup table), and **Nano Brain V1** (PR #132, 2-layer causal transformer with tied embedder/head, 2.18M params total, forward-pass verified). All three are random-init scaffolds; the training loop is the open next step. The earlier **Tokenizer V1** (Cluster 15, 52.22% whole-word compression) is superseded by V2 hybrid. The **L2 reconstruction merger** investigation is deprioritized — geometry probe showed linear PCA cannot losslessly fit 16-byte windows at D≤128, and neural ablation confirmed the direction doesn't scale within current capacity. The earlier character-level abstract-core track (Beukers gate, 83.6% masked char prediction) is a **validated but archived prior track** — see "Earlier exploration" below.

  **Pipeline architecture (current — byte-level):**
  - **L0 Byte Unit** (LOCKED, shipped): C19 architecture, `8 → 24 → 16` tied mirror autoencoder. Input: 1 raw byte (8 bits). Output: 16-dim embedding. Int4 precision, 100% lossless on all 256 bytes. Artifact: `tools/byte_unit_winner_int4.json`. Deploy form: `tools/byte_embedder_lut.h` (256-entry LUT, 4.1 KB).

    **Alternative champion (Cluster 17, 2026-04-19):** An exhaustive (precision × activation × H) sweep found that **binary + C19 + H=16** also reaches 100% lossless on all 256 bytes, with a smaller hidden width (16 vs 24) and a lighter weight JSON (6.5 KB vs 8.9 KB, 26% smaller). The baked int8 LUT size is unchanged at 4 KB raw — it is determined by the 256-entry × 16-dim output shape, not by internal precision. This alternative champion is retained at `output/byte_unit_champion_binary_c19_h16/` alongside the int4 champion. The int4 C19 H=24 artifact (`tools/byte_unit_winner_int4.json`, `tools/byte_embedder_lut.h`) remains the proven production artifact; the binary champion is the candidate for the next deploy surface pending downstream SDK migration. See [Timeline Archive](Timeline-Archive) Cluster 17 for the full activation-precision pairing matrix and reproduction instructions.
  - **L1 Byte-Pair Merger** (CHAMPION, shipped): single-W mirror-tied autoencoder (`C19(x @ W + b1) @ W.T + b2`, one 32×81 matrix, 2592 weight cells). Input: 2 × L0 outputs = 32-dim. Output: 32-dim merged. 100% lossless on all 65,536 byte pairs. Deploy champion: **3440 B (3.36 KB) Huffman-packed** (`output/merger_single_w_huffman_pack/packed_model.bin`, commit `f0ab75a`). Progression: fp32 11.20 KB → fp16 5.60 KB (Cluster 12) → generator+Huffman **3.36 KB (Cluster 13)**. Standard compressors (lzma/bz2/gzip on raw fp16) all beaten by the custom structured encoding. Shannon floor: 2422 B (~42% gap remains).
  - **Brain / higher layers** (NEXT): INSTNCT gradient-free self-wiring on top of frozen L0+L1 byte features.

  **Earlier exploration (character-level abstract-core, archived 2026-04-18):**
  - **L0 Char Embedding** (VALIDATED): 16-dimensional character lookup table. 100% lossless round-trip verified.
  - **L1 Conv with Beukers gate** (VALIDATED): activation `xy/(1+|xy|)`, k=7, nf=128 filters. 83.6% masked character prediction — project record on the char-level task. Record progression: 77.4% → 80.1% → 82.1% → 83.6%. Single layer strictly beats deep; k=7 = 14 chars = 2-3 words receptive field. Novel discovery: the Beukers gate (`f(x,y) = xy/(1+|xy|)`) emerges from zeta-function series manipulation. Brain-on-top: frozen Beukers features + brain layer = 81.8%, +1.4% over end-to-end. The earlier L0 binary byte encoder (flat 8→4 neurons, binary {-1,+1}, 36 bits, POPCOUNT) remains a validated finding for pure-integer deployment paths.
  - This track is **preserved as research history** and is not the active frontier. See [Timeline Archive](Timeline-Archive) Clusters 9-13 for the byte-level continuation and the 2026-04-15/16 session for the Beukers gate full record. See [`docs/wiki/pipeline-architecture.svg`](pipeline-architecture.svg) for the visual pipeline diagram.

## What Vraxion Is

Vraxion is both a company and a research program organized around one architecture line: fixed I/O surfaces, a self-wiring signed hidden graph, persistent internal state across ticks, and mutation-selection training instead of backpropagation through the graph.

The current implementation-facing architecture page is [INSTNCT Architecture](INSTNCT-Architecture).

INSTNCT core anatomy at a glance:

<p align="center">
  <img src="https://raw.githubusercontent.com/VRAXION/VRAXION/main/docs/assets/instnct-at-a-glance-core.png" alt="INSTNCT core anatomy at a glance" width="740">
</p>

## Mission and Method

Vraxion exists to advance machine consciousness as an engineering reality, and to build forms of intelligence and consciousness that can endure beyond any single institution, deployment, or era.

That ambition is not a claim of achieved sentience. The active public standard is stricter: theory, implementation, and proof stay separate on purpose; architecture claims must match shipped code, evidence claims must survive reproducible protocol, and experimental work must be labeled as experimental.

In practice, that means the project is run across four distinct surfaces:

- theory is stated explicitly, not implied
- implementation is described separately from the theory
- chronology and protocol live on a dedicated research surface
- the Rust grower lane is the canonical public implementation surface, and the Python `graph.py` lane is preserved as a historical reference rather than the active default

## Read the Project

- [INSTNCT Architecture](INSTNCT-Architecture) — the current implementation line, including what is shipped, validated, and still experimental
- [Local Constructability Framework](Local-Constructability-Framework) — the current theoretical umbrella; supersedes the older Theory-of-Thought and Structured-Chaos pages. Three sub-documents: [Interference Dynamics](Interference-Dynamics) (signal level), [Mutation-Selection Dynamics](Mutation-Selection-Dynamics) (structure level, the Three Laws and *C_K*), [Constructed Computation](Constructed-Computation) (emergence level), plus [Speculative Cognitive Emergence](Cognitive-Emergence-Speculative)
- [Research Process & Archive](Timeline-Archive) — the run contract, chronology, reversals, and retained proof trail
- [Rust Implementation Surface](v5-Rust-Port-Benchmarks) — the Rust implementation lane, including validation checkpoints, design notes, and archived experiments

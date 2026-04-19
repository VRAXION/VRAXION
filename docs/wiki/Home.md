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

This page is the mission-first technical front door. Use [Pages](https://vraxion.github.io/VRAXION/) for the polished front door, including the public [INSTNCT](https://vraxion.github.io/VRAXION/instnct/), [Research](https://vraxion.github.io/VRAXION/research/), and [Rust](https://vraxion.github.io/VRAXION/rust/) surfaces; use the repo [README.md](https://github.com/VRAXION/VRAXION/blob/main/README.md) for the code-facing entry, and [Research Process & Archive](Timeline-Archive) for protocol, chronology, and proof trail.

## At a Glance

- **Architecture line:** `INSTNCT`
- **Stable public release:** [`v5.0.0-beta.2`](https://github.com/VRAXION/VRAXION/releases/tag/v5.0.0-beta.2) is the current Rust public beta tag (grower-based, with the `neuron_infer` standalone CLI and the public beta training runbook); [`v5.0.0-beta.1`](https://github.com/VRAXION/VRAXION/releases/tag/v5.0.0-beta.1) remains the prior Rust language-evolution beta as a historical reference, and the legacy `v4.x` Python lane is preserved for context only.
- **Public research lane:** English remains the first-class public lane; task-memory and GPU stay secondary validation surfaces.
- **Implementation momentum:** Rust is the primary implementation surface, and the current mainline path on `main` is the bias-free threshold grower (`instnct-core/examples/neuron_grower.rs`). The released language-evolution beta still records its **24.6% peak** smooth-fitness + 1+9 jackpot result as the historical Rust language line, but the active grower lane has moved past it: see [`docs/PUBLIC_BETA_TRAINING.md`](https://github.com/VRAXION/VRAXION/blob/main/docs/PUBLIC_BETA_TRAINING.md) for the canonical training runbook and [`docs/GROWER_RUN_CONTRACT.md`](https://github.com/VRAXION/VRAXION/blob/main/docs/GROWER_RUN_CONTRACT.md) for the frozen B0 engine contract.
- **Current frontier:** The **byte-level pipeline** is the active project line as of 2026-04-18. Both L0 and L1 layers are 100% lossless and have shipped deploy artifacts. The earlier character-level abstract-core track (Beukers gate, 83.6% masked char prediction) is a **validated but archived prior track** — see "Earlier exploration" below.

  **Pipeline architecture (current — byte-level):**
  - **L0 Byte Unit** (LOCKED, shipped): C19 architecture, `8 → 24 → 16` tied mirror autoencoder. Input: 1 raw byte (8 bits). Output: 16-dim embedding. Int4 precision, 100% lossless on all 256 bytes. Artifact: `tools/byte_unit_winner_int4.json`. Deploy form: `tools/byte_embedder_lut.h` (256-entry LUT, 4.1 KB).
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
- [Theory of Thought](Theory-of-Thought) — the theoretical framing behind destructive interference and fixed-point inference
- [Research Process & Archive](Timeline-Archive) — the run contract, chronology, reversals, and retained proof trail
- [Rust Implementation Surface](v5-Rust-Port-Benchmarks) — the Rust implementation lane, including validation checkpoints, design notes, and archived experiments

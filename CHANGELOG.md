# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added — Cluster 16: lexical-to-neural bridge (2026-04-19)

- **Word Tokenizer V2 hybrid champion** (PR #130): whole-word + subword + byte-fallback, `whole_ratio=0.9375`, 32,294 vocab. Real Huffman compression **30.43%** on 10 MB FineWeb-EDU (0.46pp above bzip2, 7.19pp below gzip). 1.26% byte-fallback, 95.90% LEARNED coverage, 0/2000 unreachable tokens, 14/14 adversarial edge cases pass. Parameter choice matches SuperBPE τ=0.9 ([arXiv:2503.13423](https://arxiv.org/abs/2503.13423)). Frozen public artifact at `output/word_tokenizer_champion/`.
- **Word Embedder V1 scaffold** (PR #131): 32,294 × 64 Xavier-init lookup table, 2.07M params (8.27 MB f32 / 2.07 MB int8). Forward-pass verified text → `[N, 64]` tensor. Untrained.
- **Nano Brain V1 scaffold** (PR #132): 2-layer causal transformer, 64 dim, 4 heads, tied embedder/output head, 2.18M total params. Forward-pass verified end-to-end (text → logits). Untrained.
- **Adversarial + sanity battery** for tokenizers (`tools/diag_word_tokenizer_adversarial.py`, `_v2.py`, `_champion_freeze.py`): round-trip on 10 MB, per-input-byte fallback rate, real Huffman compression, gzip/bzip2/lzma baselines, unreachable-token audit, edge-case battery.

### Changed

- **L2 reconstruction merger line deprioritized**: PCA geometry probe and neural ablation both under-fit on 16-byte windows; the direction does not scale within current capacity. Pivoted to the word-tokenizer pipeline (Cluster 16).

## [v5.0.0-beta.2] — 2026-04-19

Grower persistence, byte-level pipeline, and L1 merger compression championship.

### Added

- **Neuron grower — forever-network mode**: task-list, interactive, and exhaustive grow modes with crash-safe incremental trace + fsync.
- **Grower CLI flags**: `--bake-best` (pick ternary-bake winner), `--force-pick N`, `--preview-only`, `--refit-alphas` (per-task alpha refit for forever-network mode).
- **L0 FINAL**: 2-neuron int8 LUT (54 bytes) — frozen deploy path for the first byte-level layer.
- **L1 FINAL**: canonical 2-byte merger — linear int8, 729/729 lossless, 1458-byte LUT.
- **L1 merger compression championship** (Clusters 11-13): single-W fp16 champion at 5.60 KB / 100% lossless (Cluster 12); Huffman-packed at 3.36 KB (Cluster 13).
- **Exact Huffman packer** for single-W hybrid merger model.
- **Byte-level L2 merger runner**: byte-roundtrip validation harness.
- **Tokenizer V1**: word, parquet, and subword tokenizer — exact lossless, space-aware.
- **FineWeb parquet pipeline** + code corpus fixture.
- **Interactive playground visualizations**: L1 Byte-Pair Merger arch + baked visualizers.
- **New public beta landing page**.

### Changed

- Grower persistent state now correctly loads checkpoint on startup (not just saves).
- Byte-level L0+L1 pipeline is now the current documented pipeline; abstract-core docs archived.
- Quantization championship findings revised: Beukers variant diagnostics + CPU/GPU multi-size sweep harness.

### Fixed

- Grower forever-network mode stall fixed by `refit_alphas` — per-task alpha refit prevents alpha saturation.

---

## [v5.0.0-beta.1] — 2026-04-06

First public beta release. Rust achieves Python parity at **24.6% peak** next-character prediction accuracy on English text.

### Breaking changes from pre-beta

- `evolve_language.rs` now uses smooth cosine-bigram fitness and 1+9 jackpot selection by default (previously: binary argmax accuracy, 1+1 ES).
- Mutation schedule rebalanced: W projection 10% → 5%, channel 5% → 10%.

### Added

- `evolution_step_jackpot()` — multi-candidate evolution step (N mutations per step, best wins). The Python "multi-worker" pattern ported to Rust.
- `Int8Projection::raw_scores()` — returns the full score vector before argmax, enabling smooth fitness computation.
- Smooth cosine-bigram fitness in `evolve_language.rs` — continuous fitness signal replaces discrete binary accuracy.
- 8 experimental examples: A/B fitness test, fixed-W test, adaptive operator selection, jackpot test, addition learning (sequential, parallel, empty-start, diagnostic).

### Key findings

- **Smooth fitness** broke the 17-18% ceiling: 21.7% peak with 1+1 ES (+2.6pp over stepwise).
- **1+9 jackpot** broke it further: 24.6% peak (+3.4pp over 1+1 ES).
- **W mutation is nearly useless**: adaptive operator test showed 0% accept rate for projection mutations across all seeds.
- **Empty-start networks outperform prefilled**: 80% accuracy on 0-4 addition with only 83 edges (vs 64% with 3400 prefilled edges). Sparse = better gradient signal for evolution.
- **Addition learning works**: seq_5x5 reaches 53% mean, 64% peak (freq baseline 20%). First proof of real computation in the spiking network.
- **Addition from empty network**: 80% accuracy on 0-4 + 0-4 from an empty network with just 83 edges (vs 64% with 3400 prefilled edges). Sparse evolution builds targeted circuits.

### Public beta surface

Rust `instnct-core` is the main public implementation surface for INSTNCT. Curated crate-root API covers network construction, propagation, evolution, SDR input, and checkpoint persistence. 150 tests, zero unsafe, full docs.

### How to run the canonical beta

```powershell
cargo run --release --example evolve_language -- <corpus-path> `
  --steps 30000 `
  --seed-count 6 `
  --report-dir target/beta-report
```

### Known limitations

- Seed variance remains high (best seed 24.6%, worst may fall below 15%).
- Addition learning works for small digits (0-4) but does not yet scale to larger ranges.
- The Python reference line remains in-repo for developers; the stable beta contract is Rust.

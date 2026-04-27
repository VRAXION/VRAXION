# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [5.0.0-beta.5] - 2026-04-27

### Changed — 2026-04-27: Phase D3/D3.1/D4 Search Aperture Function lock + 56-example archive cleanup

Patch-level beta release that closes the Phase D K-axis sweep arc, locks the Search Aperture Function K(H) table under strict acceptance, freezes the SAF v1 form as `SAF(K(H), tau=0, s=0)` (no softness needed), and consolidates the `instnct-core/examples/` tree by moving the 2026-04-17-era archive subdir off `main` per the ARCHIVE.md "only mainline belongs on `main`" rule.

- **Repo cleanup — 56 retired examples archived**: The `instnct-core/examples/archive/2026-04/` subdir (56 retired research `.rs` files + README; addition_*, pocket_*, chip_*, circuit_*, conv_*, breed_*, connectome_*, flybrain*, mirror_*, abstract_core_v1..v4, byte_*/byte_opcode_v1, grid3_*, all_binary_mirror) was preserved at the immutable content-snapshot tag `archives/instnct-examples-2026-04-archive-20260427` (pushed to origin before deletion), then physically removed from `main`. Net diff: −21,036 lines / +85 lines. Per-file restore: `git show archives/instnct-examples-2026-04-archive-20260427:instnct-core/examples/archive/2026-04/<file>.rs`.
- **`instnct-core/examples/README.md` rewrite**: previous version named `evolve_language.rs` as the canonical public-beta runner, but `BETA.md`/`README.md` have promoted `neuron_grower.rs` to current mainline since beta.2. New layout reflects reality: current-mainline runners (`neuron_grower`, `neuron_infer`, `byte_opcode_grower`) → active research runners (Phase A/B/D fixtures: `evolve_mutual_inhibition`, `evolve_bytepair_proj`, `diag_phase_b_panel`) → reference/historical runners (`evolve_language` as the released beta.1 lane) → archive pointer.
- **`tools/README.md` Phase D analyzer table extended**: 4 new rows for the post-D1 analyzers — `analyze_phase_d2_cross_h.py` (D2 cross-H), `analyze_phase_d3_klock.py` (D3), `analyze_phase_d3_fine_k.py` (D3.1 + SAF formula), `analyze_phase_d4_softness.py` (D4) — each linked to its verdict report under `docs/research/`.
- **`ARCHIVE.md` re-enumerated**: new content snapshot `archives/instnct-examples-2026-04-archive-20260427` listed; new branch-head snapshot `archive/main-pre-cleanup-20260427` (HEAD `0a6852e`) listed alongside the existing 2026-04-25 → 2026-04-26 entries.
- **Version metadata sync**: `Cargo.toml` (`5.0.0-beta.4` → `5.0.0-beta.5`), `Cargo.lock` (workspace package row), `CITATION.cff` (`version`, `date-released: 2026-04-27`), `docs/VERSION.json` (`current_release`), `README.md` (Release Snapshot block), `BETA.md` (released-tag line) all bumped.

### Added — 2026-04-26: Phase D3 K-axis lock under strict acceptance

The Search Aperture Function K-axis was tested under strict acceptance (`tau=0`, `s=0`) and the lock margin (0.50pp mean peak over K=9) reveals an H-dependent winner table.

- **D3 coarse K verdict** (`docs/research/PHASE_D3_K_LOCK_VERDICT.md`): 27 runs × 12.96M candidate rows. Result: **K(H) TABLE** — best K is H-dependent. Seed-matched (seeds 42/1042/2042) verdict at H=128/256/384 over K ∈ {1,3,5,9,13,18}. H=256 K=18 beats K=9 by 1.07pp; H=128 and H=384 stay at K=9.
- **D3.1 fine K verdict at H=256** (`docs/research/PHASE_D3_FINE_K_VERDICT.md`): 20 runs × 15.6M candidate rows over K ∈ {15,18,21,24} resolves the H=256 K-axis region. Result: **H256_K18_LOCK** — K=18 wins by ≥0.50pp lock margin (peak 6.10% mean, 0.73 std) over the fine grid.
- **SAF K formula readout** (`docs/research/SAF_K_FORMULA_LOCK.md`): provisional K(H) table — `H=128 → K=9 (provisional)`, `H=256 → K=18 (locked)`, `H=384 → K=9 (provisional)`. The null model `P_hit(K,H) = 1 - (1 - p_pos(H))^K` is diagnostic only; the practical lock rule is "smallest near-best K after penalizing instability and cost."
- **New tooling**: `tools/analyze_phase_d3_klock.py`, `tools/analyze_phase_d3_fine_k.py`. Driver `tools/diag_dimensionality_sweep.py` extended (+85 lines) with `--phase-d3` and `--phase-d3-fine` modes feeding the existing `evolve_mutual_inhibition.rs` Phase B CLI surface.

### Added — 2026-04-26 → 2026-04-27: Phase D4 softness arm verdict — SAF v1 stays strict

D4 tested whether replacing the strict acceptance valve (`tau=0`, `s=0`) with a softness arm (`zero_p` ∈ {0.3, 1.0}) under the locked K(H) recovers any of the H=384 accept-rate collapse without sacrificing peak.

- **D4 verdict** (`docs/research/PHASE_D4_SOFTNESS_VERDICT.md`): 45 runs × 21.6M candidate rows (3 H × 3 policies × 5 seeds). Result: **SAF_STRICT_LOCK** — no `zero_p` arm beats strict by ≥0.50pp on any H. SAF v1 can remain `SAF(K(H), tau=0, s=0)` for this substrate.
- Per-H winners (n=5, K(H) locked): H=128 K=9 strict 4.62%, H=256 K=18 strict 6.10%, H=384 K=9 strict 5.50%. The `zero_p_1.0` arm at H=384 hits 99.67% accept-rate but does not improve peak; the `zero_p_0.3` arm at H=384 also has 1 collapse vs 0 collapses for strict.
- **New tooling**: `tools/analyze_phase_d4_softness.py`. Driver `tools/diag_dimensionality_sweep.py` extended (+346 lines) with `--phase-d4` softness mode.

### Verification

The full README "5-Minute Proof" canonical battery was run before tagging:

- `cargo test -p instnct-core --release` — all unit, integration, and doc tests pass (exit 0; 197 + 1 + 2 + 14 = 214 tests).
- `python tools/run_grower_regression.py` — B0 engine-freeze contract: 6-task regression matrix completes (mean_val 88.4%, max_val 100.0%, mean_neurons 5.67, 0 stalls), evidence bundle written, **Golden Check PASS**.
- `python tools/run_byte_opcode_acceptance.py` — B1 promotion gate: byte/opcode v1 LUT-translator path correct on all probe entries; direct-path negative control behaves as expected (selective MISSes).
- `python tools/check_public_surface.py` — public-surface drift check passes (re-run after each doc edit, not just at the end).
- `python -m compileall Python tools` — all Python sources compile cleanly.
- `python -m pytest Python/ -q` — 31 passed, 1 skipped.

Pre-cleanup HEAD: `0a6852e` (preserved at `archive/main-pre-cleanup-20260427`, pushed to origin). Cleanup commit: `041c4f7`.

## [5.0.0-beta.4] - 2026-04-26

### Changed — 2026-04-26: D2 cross-H verdict + repo consolidation pass

Patch-level release that consolidates the GPT-side `research/sandbox-h128-d1-cross-h-finding` sandbox branch back into `main`, repairs three dangling archive-branch references that survived prior cleanup passes, and brings doc-tracked version metadata back in sync with the actually-released tag.

- **Branch consolidation (continued)**: 1 → 0 non-main branches on `origin`. The `research/sandbox-h128-d1-cross-h-finding` branch (3 commits, GPT-side D1b H=128 cross-H pilot at n=3) was preserved at `archive/research-sandbox-h128-d1-20260426`, then its core insight (flag the H-dependence of the search activation `(K, policy)` pairing) was synthesized with main's authoritative D1 (H=384, n=5) and D2 (H={128,256}, n=5) data into a new wiki section. The branch was deleted from origin after the archive tag was confirmed online. Remote now has only `origin/main`.
- **Wiki — Phase D2 cross-H verdict integrated**: `docs/wiki/Constructed-Computation.md` gains a new "H-dependence of (K, policy) — Phase D2 cross-H verdict (2026-04-26)" section. Per-H winners (n=5): H=128 K=9 strict 4.62%, H=256 K=9 strict 5.28%, H=384 K=9 strict 5.50%. K=9 strict generalizes; K=1 ranking flips between H=128 and H≥256; K=3 has a softer H-gradient. The pilot's K=9 ties-wins reading at n=3 was a small-n artefact not reproduced at n=5. `docs/wiki/Mutation-Selection-Dynamics.md` gains a one-paragraph cross-reference. The `VRAXION.wiki` GitHub wiki was synced via `tools/sync_wiki_from_repo.py`.
- **Doc consistency — dangling archive refs purged**: `archive/diamond-code-era-20260322` and `archive/instnct-surface-freeze-20260322` (two distinct branch identifiers, occurring at five line-positions across `README.md` and `ARCHIVE.md`) exist nowhere — not on remote (`git ls-remote origin`), not in `.git/packed-refs`, not in any local `.git/refs/`. README now points to `ARCHIVE.md` for the canonical archive list. ARCHIVE.md re-enumerated against the actually-existing 12+ archive tags spanning the 2026-03-18 → 2026-04-26 cleanup eras.
- **Pre-cleanup snapshots**: `archive/main-pre-cleanup-20260426` (main HEAD `aca3b91` before this pass) and `archive/research-sandbox-h128-d1-20260426` (research HEAD `a9dc92f` before deletion) — both pushed to origin before any change, in line with the ARCHIVE.md preserve-then-delete protocol.
- **Version metadata sync**: `docs/VERSION.json` (`current_release`) and `CITATION.cff` (`version`, `date-released`) had drifted to `v5.0.0-beta.2` despite the beta.3 release on 2026-04-25. All version-declaring files (`Cargo.toml`, `CITATION.cff`, `docs/VERSION.json`, `README.md`, `BETA.md`, `CHANGELOG.md`) updated to `v5.0.0-beta.4` / `2026-04-26`.

### Verification

The full README "5-Minute Proof" canonical battery was run before tagging:

- `cargo test -p instnct-core --release` — all unit, integration, and doc tests pass (exit 0).
- `python tools/run_grower_regression.py` — B0 engine-freeze contract: regression matrix completes, evidence bundle written, Golden Check **PASS**.
- `python tools/run_byte_opcode_acceptance.py` — B1 promotion gate: byte/opcode v1 LUT-translator path correct on all probe entries; direct-path negative control behaves as expected (selective MISSes).
- `python tools/check_public_surface.py` — public-surface drift check passes.
- Wiki cross-link audit on the two modified wiki pages — all internal links resolve.

## [5.0.0-beta.3] - 2026-04-25

### Changed — 2026-04-25: branch consolidation + tools cleanup pass

Repo-wide cleanup that merges the parallel research tracks back into a single `main` and prunes 53 legacy scripts from the previous research arc. Per the ARCHIVE.md protocol, every removed surface is preserved at a uniquely-named tag.

- **Branch consolidation**: 6 branches collapsed into single `main`. The `codex/phase-b-logging-smoke` Phase A→B sweep tooling (11 commits) was merged via 3-way merge with theirs/codex resolution on 4 add/add or content conflicts in `evolve_mutual_inhibition.rs`, `diag_dimensionality_sweep.py`, `diag_phase_d0_5_jackpot_aperture.py`, and `docs/PHASE_B_PRE_REG.md` (citation-precision fix on the last). The `claude/review-repo-access-Ug8Si` branch was verified fully redundant (codeql.yml blob hash identical, 7 LCF docs commits already in PR #146). The `research/overnight-sct-empirical-20260423` 8-iteration LOOP-COMPLETE branch and the two `saved/*` historical branches were archive-tagged and removed.
- **Archive tags created**: `archive/main-pre-cleanup-20260425`, `archive/codex-phase-b-logging-smoke-20260425`, `archive/research-overnight-sct-empirical-20260425`, `archive/saved-neuron-one-wip-20260425`, `archive/saved-pre-connectome-research-20260425`, `archive/claude-review-repo-access-20260425` — six branch heads preserved as immutable snapshots.
- **Tools/ trim**: 74 entries → 22. Removed 52 scripts from the 2026-04-17 → 2026-04-19 byte-pair merger / Block C quantization / word-tokenizer research lines (Block C activation/quantization 14, byte-pair merger 10, byte Huffman/L2 8, word tokenizer 6, MLP baselines 4, probe-weight 2, modal/sweep 2, pretokenize 2, misc 4). Preserved at tag `archives/tools-cleanup-20260425`. Note: `diag_byte_unit_widen_sweep.py` was initially staged for archival but restored after a post-archive verification caught that `build_byte_unit.py` (a kept reproduction script) imports from it; preserved in tools/ as an internal dependency only. Net diff: −13,884 lines.
- **`tools/README.md` rewrite**: previous version listed canonical scripts (`diag_qat_ste`, `diag_byte_pair_merger_lookup_codebook`, `diag_byte_single_w_*` series) that had been archived in the 2026-04-18/20 cleanup passes; the file was stale relative to the actual tree. Rewritten to reflect the current 21-entry layout: public-beta contract scripts, CI utilities, Block A deploy artifacts, and the active Phase A/B/D research line.
- **SDK comment updates**: `Python/block_c_embedder/embedder.py` docstring + `README.md`, `Python/block_b_merger/merger.py`, and `Rust/src/block_b_merger/mod.rs` comments updated to reference the archive tag instead of dangling `tools/<script>.py` paths. Runtime is unchanged; only documentation strings move.
- **`.gitignore` fix**: `instnct-core/target/` added to the rust-build-artifacts gate (the workspace-member `target/` directory was previously untracked-but-not-ignored).
- **Wiki sync**: 5 LCF sub-pages created on the GitHub wiki (`Local-Constructability-Framework.md`, `Interference-Dynamics.md`, `Mutation-Selection-Dynamics.md`, `Constructed-Computation.md`, `Cognitive-Emergence-Speculative.md`); 4 existing pages updated (`Home.md`, `Theory-of-Thought.md` and `Structured-Chaos-Theory.md` as redirect stubs, `_Sidebar.md` for nav).

### Added — 2026-04-23/25: Phase A → B → D mutation-selection research line

Multi-seed dimensionality / mutation-selection / acceptance-aperture study built on `evolve_mutual_inhibition` and `evolve_bytepair_proj` fixtures, replacing the single-seed observations from the late beta.2 era.

- **Phase A baseline (`docs/research/PHASE_A_BASELINE.md`)**: H ∈ {128, 192, 256, 384} × 5-seed sweep on the byte-pair prediction fixture. Measured peak_acc, accept-rate, alive_frac, edges; revealed an inverted-U with peak at H=256 (mean 5.28% ± 1.79pp) and a monotonic accept-rate collapse 78→42→13% as H grows.
- **Phase B confound-vs-intrinsic test (`docs/research/PHASE_B_VERDICT.md`, `docs/PHASE_B_PRE_REG.md`)**: pre-registered 5-arm × 5-seed protocol at fixed H=384 testing whether the H=384 decline is intrinsic or driven by step budget / jackpot size / propagation depth / input-channel saturation. Drives the C_K constructability metric on per-candidate logs.
- **Phase B.1 horizon × accept-ties follow-up (`docs/research/PHASE_B1_PRE_REG.md`, `docs/research/PHASE_B1_VERDICT.md`)**: 3-tier horizon scan (S20 / S40 / S80) crossed with strict / ties acceptance; identifies the binding constraint as the acceptance valve at H=384.
- **Phase D0/D0.5/D1 acceptance-aperture series**: D0 acceptance-aperture metric on candidate logs (`docs/research/PHASE_D0_ACCEPTANCE_APERTURE.md`); D0.5 offline K-resampling (`docs/research/PHASE_D0_5_JACKPOT_APERTURE.md`) separates the jackpot/sampling aperture from the acceptance valve; D1 zero-drive policy K × zero_p factorial sweep.
- **Driver consolidation in `tools/diag_dimensionality_sweep.py`**: single multi-mode driver covering default H-sweep + `--phase-b` + `--phase-b1` + `--phase-d1` arms, with `ThreadPoolExecutor`-based per-cell parallelism (`--jobs N`).
- **Phase B CLI on `evolve_mutual_inhibition.rs`**: 14 new flags (`--jackpot`, `--ticks`, `--accept-ties`, `--accept-policy`, `--neutral-p`, `--accept-epsilon`, `--input-scatter`, `--candidate-log`, `--checkpoint-at-end`, `--panel-interval`, `--panel-log`, `--phase`, `--arm`, `--run-id`); `AcceptancePolicy` enum (Strict / Ties / ZeroP / Epsilon) plumbed through `instnct-core/src/evolution.rs` (+543 lines); `RunMeta` JSON serialization for reproducible artifact provenance; extended `SUMMARY` JSON with phase / arm / run_id / horizon_steps / accept_ties / accept_policy / neutral_p / accept_epsilon fields.
- **Five new analyzers**: `analyze_phase_a_baseline.py`, `analyze_phase_b_verdict.py`, `analyze_phase_b1_verdict.py`, `analyze_phase_d1_verdict.py`, `analyze_acceptance_aperture.py`; one new diagnostic `diag_constructability_analysis.py` (C_K decomposition regression across V_raw / M_pos / A / I_proxy / D_eff / cost_eval / R_neg).
- **CodeQL workflow** (`.github/workflows/codeql.yml`): pinned to `{python, rust, actions}` to drop the failing default `c-cpp` matrix job that was producing constant red CI.

### Added — 2026-04-21/22: ABC-Brain integration, fitness sweep, crystallize, ablation

First end-to-end wiring of the frozen ABC feature pipeline into the INSTNCT brain. Intensive experimentation covering fitness function optimization, structural experiments, crystallize port, and ablation study revealing single-attractor topology collapse.

- **ABC-to-Brain char-level integration**: C-embedding matches SdrTable at ~25% (multi-seed validated). First end-to-end wiring confirms embedding quality is not the bottleneck — brain topology is.
- **Byte-pair prediction (397 class)**: 7.1% peak with smooth cosine fitness (frequency baseline 4.2%). Real signal on a much harder task, but brain topology limits further progress.
- **Fitness function sweep (10 variants)**: smooth linear cosine champion. Dominates stepwise, argmax, pure-accuracy, and other cosine variants. The fitness signal shape remains the single biggest lever for mutation-selection.
- **Crystallize ported from Python to Rust**: grow-prune-regrow cycles validated. Converges to compact circuits that retain functional accuracy.
- **Ablation study**: systematic ablation reveals single-attractor topology collapse — 7 dominant neurons form a bottleneck. The brain converges to one attractor basin instead of developing competing pathways. This is the core pathology limiting current accuracy.
- **Structured Chaos Theory v1.0**: three laws (Single Constraint, Anti-Monopoly, Opponent) formulated from accumulated experimental evidence. Learning formula: `S x sensitivity / dimensions`. Added to `docs/wiki/Theory-of-Thought.md`.

### Added (negative results) — 2026-04-21/22

- **Edge weights [1-3]: worse than binary** — weighted edges degrade performance vs binary {0,1} masks. Signal-to-noise ratio degrades when edge precision increases. Reconfirms topology > edge precision.
- **Multi-channel input: worse than single** — dual-input and multi-channel injection schemes all worse. Dimension curse: extra channels increase search space faster than they add useful signal.

### Added — 2026-04-21: Block C byte-pair embedder champion + deploy SDK

Canonical ABC-pipeline-ready Block C embedder trained, quantized, and packed. Full bytes-in / embeddings-out path now available as pure numpy.

- **Training** (Modal L4 GPU, ~$9 total): full-softmax next-pair CE on 100 MB FineWeb-EDU, E=32 / H=128 / context=16, 3 seeds (1, 3, 7). Two-phase:
  - v1: LR=0.1 from scratch — diverged after ep3 (peak acc@1 31%), but revealed strong syntactic clusters.
  - v2 (champion): warm-start emb from v1 ep3 + LR=0.03 + cosine decay, 10 epochs — **acc@1 34.06 ± 0.82%**, no divergence, clusters tightened and extended.
- **Intelligent quantization** (`tools/diag_bytepair_mixed_quant.py`): hot-vs-cold split by corpus frequency. 3,386 pairs w/ freq ≥ 5 quantized to per-channel int4 α=0.5; 62,150 cold pairs collapse to one shared OOV vector. Cluster overlap with float reference: 74.4% on hot-restricted top-5.
- **Baked artifact**: `output/block_c_bytepair_champion/packed.bin` (62,528 B / 61 KB, **134× compression** vs float32 8.39 MB). Format: `VCBP` v1, per-channel scales fp16, shared OOV fp16, 65,536-bit hot bitmap, int4-packed hot rows. Bake script: `tools/bake_block_c_bytepair.py`.
- **Python deploy SDK**: `Python/block_c_embedder/embedder.py` — `L2Embedder.load_default()`, `embed_id`, `embed_ids`, `encode_bytes`. Zero ML deps.
- **Chain A+B+C stress test** (10 invariants, all pass): A round-trip lossless, B sign-match lossless, C header + scheme, C determinism, OOV sharing across 100+ cold pairs, hot uniqueness > 99.5%, semantic cluster preservation (`. ` → `! `/`? `/`.\n`, `, ` → `; `/`: `, `' t'` → `'\nt'`/`'(t'`/`'-t'`), 100 KB corpus encode, edge cases (empty/single/odd/binary). Script: `Python/block_c_embedder/tests/test_chain_a_b_c.py`.
- **Learned clusters** (emergent from data, no hand-crafted labels): word-start equivalence (`' t'` = `'\nt'` = `'(t'` = `'-t'`), case-invariance (`th`↔`Th`, `he`↔`He`, `in`↔`In`, `on`↔`On`, `an`↔`An`), sentence-terminator group, clause-punct group, function-word group (`in`/`by`/`on`).
- **Infrastructure**: Modal app wrapper `tools/modal_block_c.py` with L4/T4/A10G tier functions, volume commit-hook for mid-run progress polling, `tools/monitor_sweep.py` live-status tool, chunked tokenizer `tools/pretokenize_chunked.py`.

### Changed — 2026-04-20: public-release cleanup

Four-phase main-branch cleanup to prepare the repo for public release. All phases follow the same pattern: content preserved (archive branch/tag or git history), main narrowed to live surfaces.

- **Python research lane archived**: `instnct/` (pre-Rust migration lane) moved to tag `archives/python-research-20260420`. Mainline Python surface is now `Python/` (deploy SDK, Block A + B, pure numpy, zero ML deps). README, VALIDATED_FINDINGS, VERSION.json, CONTRIBUTING all updated.
- **Docs legacy Pages archived**: 62 orphan files under the pre-Blocks site nav (`docs/instnct/`, `docs/byte-embedder/`, `docs/research/`, `docs/rust/`, `docs/pages/brain_replay/`, `docs/vraxion-connectome-explorer.html`) removed. Current `docs/` surface is Home + Blocks A-E + Legacy detail view + Wiki mirror.
- **Output scratch tree pruned**: `output/` went from 49 MB / ~160 run-dumps to 4.1 MB / 3 champion folders (`byte_unit_champion_binary_c19_h16/`, `merger_single_w_huffman_pack/`, `word_tokenizer_champion/`). Scratch was gitignored so not recoverable; source scripts in `tools/` can regenerate any run.
- **Tools/ Fázis 6 trim**: 79 scripts → 29 canonical (champion build/verify/acceptance, canonical sweep/methodology, active frontier L2 + word tokenizer). The 50 archived scripts preserved at tag `archives/tools-legacy-diag-20260420`.
- **CI modernized**: `.github/workflows/ci.yml` rewritten for `Python/` deploy SDK + `pytest`; `tools/check_public_surface.py` rewritten for the Blocks A-E nav; `.github/pull_request_template.md` scope checklist updated.

Commits: `56575ab` (instnct/ archival), `92f313b` (docs Pages), `7571356` (output/), `c7dace4` (tools/ trim), `5b006a5` (CI modernization), `a85dde5` (branch → tag conversion), `831e971` (pytest install fix).

### Added — Cluster 16: lexical-to-neural bridge (2026-04-19)

- **Word Tokenizer V2 hybrid champion** (PR #130): whole-word + subword + byte-fallback, `whole_ratio=0.9375`, 32,294 vocab. Real Huffman compression **30.43%** on 10 MB FineWeb-EDU (0.46pp above bzip2, 7.19pp below gzip). 1.26% byte-fallback, 95.90% LEARNED coverage, 0/2000 unreachable tokens, 14/14 adversarial edge cases pass. Parameter choice matches SuperBPE τ=0.9 ([arXiv:2503.13423](https://arxiv.org/abs/2503.13423)). Frozen public artifact at `output/word_tokenizer_champion/`.
- **Word Embedder V1 scaffold** (PR #131): 32,294 × 64 Xavier-init lookup table, 2.07M params (8.27 MB f32 / 2.07 MB int8). Forward-pass verified text → `[N, 64]` tensor. Untrained.
- **Nano Brain V1 scaffold** (PR #132): 2-layer causal transformer, 64 dim, 4 heads, tied embedder/output head, 2.18M total params. Forward-pass verified end-to-end (text → logits). Untrained.
- **Adversarial + sanity battery** for tokenizers (`tools/diag_word_tokenizer_adversarial.py`, `_v2.py`, `_champion_freeze.py`): round-trip on 10 MB, per-input-byte fallback rate, real Huffman compression, gzip/bzip2/lzma baselines, unreachable-token audit, edge-case battery.

### Added — Cluster 17: low-bit byte-unit activation-precision sweep (2026-04-19)

- Alternative L0 champion: **binary + C19 + H=16** (PR #137)

GPT's exhaustive activation-precision sweep tested all (precision, activation) pairs across binary, ternary, 2-bit, and int4 widths combined with tanh, ReLU, and C19 activations. The sweep found that C19 + binary weights reaches 100% exact lossless byte reconstruction at H=16 — the smallest hidden width of any tested combination. The full sweep matrix: tanh + 2-bit @ H=12 (smallest for 2-bit); tanh + ternary @ H=32; C19 + binary @ H=16 (smallest overall). Weight-reload and LUT-based round-trip both verify 256/256. Artifacts at `output/byte_unit_champion_binary_c19_h16/`: weights JSON (6.5 KB, 26% smaller than the int4 champion's 8.9 KB), raw int8 LUT (4 KB), and C header (30 KB).

This is an **alternative** champion — the int4 C19 H=24 model remains the proven production artifact (committed LUT at `tools/byte_embedder_lut.h`). The binary + C19 + H=16 result is a validated alternative for constrained-width deployments.

Reproduce: `python tools/build_byte_unit.py`

### Added — Cluster 18: L1 merger autonomous compression loop (2026-04-19)

- **Native 7-bit identity merger candidate**: H=120 identity autoencoder + 7-bit integer weights + 3-bit `b1`/`b2` biases + fp32 alpha = **3,421 B (3.34 KB) exact lossless** — ~0.55% smaller than the Huffman-packed champion and **native** (no decode step). Seeds 7 and 42 both reach 100%.
- **Codebook expressivity ladder (single-W H=81 bake probe)**: binary 0.25% → ternary 1.82% → 3-bit 17.47% → 4-bit 29.28% → 5-bit 50.19% → 6-bit 74.08% → 7-bit 89.17%. Below 4 bit/weight the problem is **not representable**, no amount of QAT or LBFGS rescues it — it is a representation-space ceiling, not an optimization failure.
- **Confirmed negative results**: dual-W architecture does NOT rescue binary (all 10 dual-W binary multi-seed runs at H=48 → 0%); multi-seed binary at H=81/128/192/256 all < 10%; alpha scaling cannot be eliminated (seed 42 → 65,480 bad without alpha); C19 aux params (`c`, `rho`, biases) cannot be post-hoc quantized at int8 even on the float exact model.
- **New tools**: `diag_byte_pair_merger_widen_sweep.py` (activation × codebook × H × single/dual-W sweep with Adam+LBFGS), `diag_byte_pair_merger_bake_probe.py` (codebook expressivity measurement), `diag_byte_pair_merger_perchannel_bake.py`, `diag_byte_pair_merger_minimize.py` (deploy-bytes ranker), `diag_byte_pair_merger_aux_quant_probe.py`, `diag_byte_pair_merger_float_aux_quant_probe.py`, `diag_byte_pair_merger_alpha_ablation.py`.
- Full findings draft: `docs/wiki/COMPRESSION_LOOP.md`.

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

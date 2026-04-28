# `tools/`

Helper scripts and deploy artifacts that live alongside the Rust workspace
in `instnct-core/` and the Python deploy SDK in `Python/`. The directory was
trimmed to its current shape on 2026-04-25 as part of the v5.0.0-beta.3
cleanup pass; 53 legacy scripts from the 2026-04-17 → 2026-04-19 byte-pair
merger / Block C quantization research lines were archived to tag
[`archives/tools-cleanup-20260425`](https://github.com/VRAXION/VRAXION/releases/tag/archives%2Ftools-cleanup-20260425).

Restore any archived script via:

```bash
git show archives/tools-cleanup-20260425:tools/<filename>
git checkout archives/tools-cleanup-20260425 -- tools/<filename>
```

For the timeline of the research lines those scripts produced, see the
[Timeline-Archive wiki page](https://github.com/VRAXION/VRAXION/wiki/Timeline-Archive)
(blueprints recorded as Clusters 9, 10, 11, 12).

## Public-beta contract scripts

Both are referenced from `README.md` as the 5-minute proof. They must stay
green on `main`.

| Script | Purpose |
| --- | --- |
| [`run_grower_regression.py`](run_grower_regression.py) | Canonical grower regression bundle (`docs/GROWER_RUN_CONTRACT.md`); produces append-only evidence under `target/grower-regression/<timestamp>/`. The B0 engine-freeze gate. |
| [`run_byte_opcode_acceptance.py`](run_byte_opcode_acceptance.py) | Byte/opcode v1 exact-translator acceptance harness (B1 promotion gate; see [`docs/BYTE_OPCODE_V1_CONTRACT.md`](../docs/BYTE_OPCODE_V1_CONTRACT.md)). |

## CI utilities

Invoked by `.github/workflows/ci.yml`:

| Script | Purpose |
| --- | --- |
| [`check_public_surface.py`](check_public_surface.py) | Verifies the public-facing docs (README, BETA, blocks/) agree with the canonical code path on `main`. |
| [`sync_wiki_from_repo.py`](sync_wiki_from_repo.py) | Mirrors `docs/wiki/` into the `VRAXION.wiki.git` submodule. CI runs `--dry-run`; commit-side mirrors push. |

## Block A (byte unit) deploy artifacts and reproduction

| File | Description |
| --- | --- |
| [`byte_embedder_lut.h`](byte_embedder_lut.h) | C/Rust-importable production LUT for the int4 C19 H=24 byte-unit champion. |
| [`byte_embedder_lut_int8.json`](byte_embedder_lut_int8.json) | int8 variant of the byte-unit LUT (alternative deploy). |
| [`byte_embedder_lut_int8_nozero.json`](byte_embedder_lut_int8_nozero.json) | int8 LUT with the zero quantization level dropped (constrained-width deploy variant). |
| [`byte_unit_winner_int4.json`](byte_unit_winner_int4.json) | int4 weights of the byte-unit champion, paired with `byte_embedder_lut.h`. |
| [`build_byte_unit.py`](build_byte_unit.py) | Reproduce the byte-unit champion bake from the source corpus (referenced from `CHANGELOG.md`). |
| [`diag_byte_unit_widen_sweep.py`](diag_byte_unit_widen_sweep.py) | Internal dependency of `build_byte_unit.py` — the float training + activation/quant search loop the bake script imports. Not invoked directly. |

## Phase A → B → D research line (active)

The 2026-04-23 → 2026-04-25 mutation/selection/dimensionality study runs on
[`instnct-core/examples/evolve_mutual_inhibition.rs`](../instnct-core/examples/evolve_mutual_inhibition.rs)
(and `evolve_bytepair_proj.rs` for the grow-prune fixture). The driver and
analyzers below produce the artifacts under
[`docs/research/`](../docs/research/) and `output/dimensionality_sweep/<timestamp>/`.

### Driver

| Script | Purpose |
| --- | --- |
| [`diag_dimensionality_sweep.py`](diag_dimensionality_sweep.py) | Multi-mode driver: default H-sweep + `--phase-b` + `--phase-b1` + `--phase-d1` + `--phase-d2` cross-H + `--phase-d3` / `--phase-d3-fine` SAF K-lock + `--phase-d4` softness + `--phase-d6` trajectory-field + `--phase-d7` operator-bandit + `--phase-d8` archive/scan/cell sub-arms. ThreadPoolExecutor for parallel cells via `--jobs N`. |

### Phase A / B / B.1 analyzers

| Script | Phase | Purpose |
| --- | --- | --- |
| [`analyze_phase_a_baseline.py`](analyze_phase_a_baseline.py) | A | Aggregates the H ∈ {128, 192, 256, 384} × 5-seed baseline; emits the inverted-U plot. |
| [`analyze_phase_b_verdict.py`](analyze_phase_b_verdict.py) | B | Confound-vs-intrinsic statistical readout: B0 vs B1..B4 Welch t-tests, decomposition regression. |
| [`analyze_phase_b1_verdict.py`](analyze_phase_b1_verdict.py) | B.1 | Horizon × accept-ties tie-policy verdict; reads the `panel_timeseries.csv` candidate logs. |

### Phase D / acceptance-aperture / SAF K-lock analyzers

| Script | Phase | Purpose |
| --- | --- | --- |
| [`diag_phase_d0_aperture.py`](diag_phase_d0_aperture.py) | D0 | Operationalizes the acceptance-aperture metric on B.1 candidate logs. |
| [`analyze_acceptance_aperture.py`](analyze_acceptance_aperture.py) | D0 | Companion analyzer for the D0 acceptance-aperture verdict. |
| [`diag_phase_d0_5_jackpot_aperture.py`](diag_phase_d0_5_jackpot_aperture.py) | D0.5 | Offline K-resampling on B.1 logs (K ∈ {1, 2, 3, 5, 9}); separates the jackpot/sampling aperture from the acceptance valve. |
| [`diag_phase_d0_6_minimum_useful.py`](diag_phase_d0_6_minimum_useful.py) | D0.6 | Minimum-useful-improvement threshold sweep. |
| [`analyze_phase_d1_verdict.py`](analyze_phase_d1_verdict.py) | D1 | Zero-drive policy K × zero_p factorial verdict. |
| [`analyze_phase_d2_cross_h.py`](analyze_phase_d2_cross_h.py) | D2 | Cross-H replication (H ∈ {128, 256, 384}) of the D1 (K, policy) winner; produces [`docs/research/PHASE_D2_CROSS_H_VERDICT.md`](../docs/research/PHASE_D2_CROSS_H_VERDICT.md). |
| [`analyze_phase_d3_klock.py`](analyze_phase_d3_klock.py) | D3 | Coarse K-axis lock under strict acceptance (`tau=0`, `s=0`); produces [`docs/research/PHASE_D3_K_LOCK_VERDICT.md`](../docs/research/PHASE_D3_K_LOCK_VERDICT.md). |
| [`analyze_phase_d3_fine_k.py`](analyze_phase_d3_fine_k.py) | D3.1 | Fine K-grid (K ∈ {15, 18, 21, 24}) at H=256 + SAF K formula readout; produces [`docs/research/PHASE_D3_FINE_K_VERDICT.md`](../docs/research/PHASE_D3_FINE_K_VERDICT.md) and [`docs/research/SAF_K_FORMULA_LOCK.md`](../docs/research/SAF_K_FORMULA_LOCK.md). |
| [`analyze_phase_d4_softness.py`](analyze_phase_d4_softness.py) | D4 | Softness arm (`zero_p`) lock under the SAF K(H) lock; produces [`docs/research/PHASE_D4_SOFTNESS_VERDICT.md`](../docs/research/PHASE_D4_SOFTNESS_VERDICT.md). |
| [`analyze_phase_d6_trajectory_field.py`](analyze_phase_d6_trajectory_field.py) | D6 / D6.1 | Trajectory-field audit + falsification audit (early feature model + group-CV + trajectory alignment); produces [`docs/research/PHASE_D6_TRAJECTORY_FIELD_AUDIT.md`](../docs/research/PHASE_D6_TRAJECTORY_FIELD_AUDIT.md). |
| [`analyze_phase_d7_operator_bandit.py`](analyze_phase_d7_operator_bandit.py) | D7 / D7.1 | Operator-sampling-weight bandit (D7_BASELINE vs D7_PRIOR_EWMA vs D7_STATIC_PRIOR) under locked SAF v1; produces [`docs/research/PHASE_D7_OPERATOR_BANDIT_AUDIT.md`](../docs/research/PHASE_D7_OPERATOR_BANDIT_AUDIT.md). |
| [`analyze_phase_d8_archive_psi_replay.py`](analyze_phase_d8_archive_psi_replay.py) | D8.1 | Archive-Psi replay audit; produces [`docs/research/PHASE_D8_ARCHIVE_PSI_REPLAY_AUDIT.md`](../docs/research/PHASE_D8_ARCHIVE_PSI_REPLAY_AUDIT.md). |
| [`analyze_phase_d8_scan_depth_knee.py`](analyze_phase_d8_scan_depth_knee.py) | D8.2 | Scan-depth knee audit; produces [`docs/research/PHASE_D8_SCAN_DEPTH_KNEE_AUDIT.md`](../docs/research/PHASE_D8_SCAN_DEPTH_KNEE_AUDIT.md). |
| [`analyze_phase_d8_cell_coherence.py`](analyze_phase_d8_cell_coherence.py) | D8.3 | Cell-coherence audit (cluster cohesion across H/seed); produces [`docs/research/PHASE_D8_CELL_COHERENCE_AUDIT.md`](../docs/research/PHASE_D8_CELL_COHERENCE_AUDIT.md). |
| [`analyze_phase_d8_frontier_pointer_replay.py`](analyze_phase_d8_frontier_pointer_replay.py) | D8.4 | Frontier-pointer replay audit; produces [`docs/research/PHASE_D8_FRONTIER_POINTER_REPLAY_AUDIT.md`](../docs/research/PHASE_D8_FRONTIER_POINTER_REPLAY_AUDIT.md). |
| [`analyze_phase_d8_instrumentation.py`](analyze_phase_d8_instrumentation.py) | D8.5 | Instrumentation logging audit; produces [`docs/research/PHASE_D8_INSTRUMENTATION_AUDIT.md`](../docs/research/PHASE_D8_INSTRUMENTATION_AUDIT.md). |
| [`analyze_phase_d8_archive_parent.py`](analyze_phase_d8_archive_parent.py) | D8.6 / D8.6.P2 | Archive-parent microprobe (P1 + P2 model export); produces [`docs/research/PHASE_D8_ARCHIVE_PARENT_MICROPROBE.md`](../docs/research/PHASE_D8_ARCHIVE_PARENT_MICROPROBE.md). Pairs with [`export_phase_d8_p2_model.py`](export_phase_d8_p2_model.py) (P2 model export shim). |
| [`build_phase_d8_cell_atlas.py`](build_phase_d8_cell_atlas.py) | D8.7 | Cell-atlas dashboard builder (Pokédex card grid HTML + cell/neighbor/candidate CSV bundle); produces [`docs/research/PHASE_D8_CELL_ATLAS.md`](../docs/research/PHASE_D8_CELL_ATLAS.md). Imported by `analyze_phase_d8_cell_scan_delta.py`. |
| [`analyze_phase_d8_cell_scan_delta.py`](analyze_phase_d8_cell_scan_delta.py) | D8.7 | Cell-scan-delta observer (atlas re-scan after-spin diff); produces [`docs/research/PHASE_D8_CELL_SCAN_DELTA.md`](../docs/research/PHASE_D8_CELL_SCAN_DELTA.md). |
| [`diag_constructability_analysis.py`](diag_constructability_analysis.py) | post-D | C_K decomposition regression across all arms (V_raw, M_pos, A, I_proxy, D_eff, cost_eval, R_neg). |
| [`diag_byte_unit_latent_dim_sweep.py`](diag_byte_unit_latent_dim_sweep.py) | A↔Block A | Latent-dim sweep for byte-unit-fixture cross-check. |

## How to run

Most analyzers use the same default I/O layout, mirroring
`output/dimensionality_sweep/<timestamp>/` produced by the driver:

```bash
python tools/diag_dimensionality_sweep.py --phase-b \
    --H-values 384 --seeds 5 --steps 20000 \
    --out output/phase_b_$(date +%s)

python tools/analyze_phase_b_verdict.py \
    --root output/phase_b_<timestamp>
```

Each analyzer accepts `--root` (input run dir) and `--out` (output report
dir, defaults to a sibling under `output/`); use `--help` per script for
flags.

## Hardware

The current Phase A/B/D line runs on a single CPU core via `cargo run
--release` invoked from the Python driver. Multi-core scaling is a
per-cell parallelism (`--jobs N` in `diag_dimensionality_sweep.py`); each
worker forks one cargo invocation. No GPU dependency on `main` — the
legacy GPU sweeps (CUDA / `torch`) lived in the now-archived `diag_qat_*`
and `diag_quant_sweep_gpu*` scripts.

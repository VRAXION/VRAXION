# Changelog

All notable changes to VRAXION are documented here.
Releases: https://github.com/VRAXION/VRAXION/releases

---

## v3.2.000 (2026-02-17) — Architecture Complete

All 10 LCX bottleneck projection levers locked via deterministic GPU/CPU probes.

- **Bottleneck design:** `lcx_read [D] → Lin(D→D/10) → C19 → Lin(D/10→D/10) → C19 → Lin(D/10→D) × zoom_gate → + hidden`
- **C19 activation** replaces GELU in all 6 processing-layer sites
- **Orthogonal init** for bottleneck layers (+0.8% vs kaiming)
- **No residual skip** (proj(x) only — memory-space is not brain-space)
- **No LayerNorm** in bottleneck path
- **Zoom gate init 0.0** (balanced sigmoid=0.5; model opens it to 0.6)
- Mini-model proof: +13.3% bit accuracy from LCX at tt=1

## v3.0.001 (2026-02-16) — Dashboard Reorg and Score Margin Telemetry

- Dashboard reorganized: 66 panels across 8 sections (was 74)
- Score Margin telemetry: cosine similarity gap between top-K winner and first loser
- Write heat fix: max-pool downsample replaces point-sample rank bins
- Score Margin trajectory: 0.013 → 0.151 over ~250 Beta steps (routing learning confirmed)

## v3.0.000 (2026-02-16) — Diamond Code: Hash LCX Memory and Progressive Training

- **SwarmByteRingModel**: 101M params (D=6180, depth=12, ring=62)
- **Hash LCX Memory**: SimHash bucketed external memory (2,000 slots, expandable to 100K+)
- **Progressive training**: 3 curriculum axes (thought depth, sensory width, memory expansion)
- **Grafana + InfluxDB** observability stack with live controls
- Bug fixes: memory_ring detach (36x activation spike), LCX hash write detach (slow leak), Grafana dual metrics, control panel zombies
- 84 dead-end scripts archived; 19 core + 4 tools remain

## v2.10.580 (2026-02-11) — Swarm Intelligence Architecture

- **SwarmByteRingModel**: N autonomous beings sharing a ring memory bus
- **sqrt(N) scaling law**: optimal receptive field K = N^(1/2) (TOT-H008, E2)
- RF4 experiment: bit_acc 57% → 82%, ensemble_benefit +0.26
- Bug fixes: jump gate init bias, max coverage balancing, bit 62/63 freebie

## v2.10.579 (2026-02-08)

- Internal development snapshot

## v2.10.576 (2026-02-05) — Cadence Snapshot (Pre-release)

- Internal cadence snapshot for versioning baseline

## v1.0.0 (2026-01-22) — PRIME C-19: Canonical Public Snapshot

- First public release for Zenodo archiving
- Clean baseline: code-only, no datasets/checkpoints shipped
- License: PolyForm Noncommercial 1.0.0
- Zenodo DOI minted

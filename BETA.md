# `v5.0.0 Public Beta` Contract

This document defines the current Rust-first public beta surface for Vraxion.

## Canonical runner

The only canonical public beta run path in this tranche is:

```powershell
cargo run --release --example evolve_language -- <corpus-path> `
  --steps 30000 `
  --seed-count 6 `
  --seed-base 42 `
  --full-len 2000 `
  --report-dir target/beta-report
```

Arguments:

- `<corpus-path>`: positional corpus path; if omitted, the runner falls back to the current local Fineweb path
- `--steps`: evolution steps per seed
- `--seed-count`: number of parallel seeds
- `--seed-base`: base seed; later seeds use a fixed stride
- `--full-len`: evaluation length for public summaries
- `--report-dir`: directory for the evidence bundle

## Required artifacts

Each canonical beta run should emit:

- `run_cmd.txt`
- `env.json`
- `metrics.json`
- `summary.md`

These are the minimum public-beta evidence bundle. Extra logs, checkpoints, and plots are useful, but not required.

## What counts as a pass

A canonical beta run counts as passing when:

- the runner completes without panic
- all four evidence files are written
- `metrics.json` contains per-seed results and aggregate summary
- the run is reproducible under the same command and corpus

The canonical runner now uses **smooth cosine-bigram fitness** and **1+9 jackpot selection** (9 candidate mutations per step, best wins). Peak accuracy: **24.6%** next-character prediction on English text.

## Known limitations

- Seed variance remains high: best seed reaches 24.6%, worst may fall below 15%.
- Addition learning (0-4 + 0-4) reaches 80% from empty start, proving real computation, but does not yet scale to larger digit ranges.
- Experimental examples remain in-repo for research continuity, but they are not the public beta contract.

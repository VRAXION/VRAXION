# `v5.0.0 Public Beta` Contract

This document defines the current Rust-first public beta surface for Vraxion.

## Canonical runner

The only canonical public beta run path in this tranche is:

```powershell
cargo run --release --example evolve_language -- <corpus-path> `
  --steps 15000 `
  --seed-count 12 `
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

This beta does **not** require a new accuracy regime beyond the current `17-18%` stable band.

## Known limitations

- The tested Rust language recipe still converges into a stable `17-18%` band under the current 1+1 ES regime.
- Transient `19.1-19.6%` peaks are not yet promoted as stable breakthroughs.
- Pocket, breed, shared-female, and Watts-Strogatz work narrowed the frontier, but did not create a new default recipe.
- Experimental examples remain in-repo for research continuity, but they are not the public beta contract.

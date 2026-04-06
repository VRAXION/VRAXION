# Draft Release Notes: `v5.0.0 Public Beta`

## Summary

`v5.0.0 Public Beta` promotes the Rust `instnct-core` lane into the main public implementation surface for INSTNCT. This beta is about a clean, reproducible runtime and evolution substrate, not a claim that the current language ceiling has already been broken.

## Highlights

- Rust-first public beta surface through `instnct-core`
- Curated crate-root API for network construction, propagation, evolution, SDR input, and checkpoint persistence
- Canonical public beta runner: `examples/evolve_language.rs`
- Reproducible evidence bundle: `run_cmd.txt`, `env.json`, `metrics.json`, `summary.md`
- Minimal GitHub Actions validation for the beta surface

## Current doctrine

- The current stable public release remains `v4.2.0`.
- The Rust beta lane is serious and reproducible, but still beta.
- The tested 1+1 ES language recipe still converges into a stable `17-18%` band.
- Pocket-pair, shared-female, and Watts-Strogatz runs narrowed the frontier, but did not create a promoted replacement recipe.
- Python remains part of the broader project as a reference line for developers; the stable beta contract in this repo is Rust.

## Known limitations

- No promoted breakthrough beyond the current stable band yet
- Public beta claims are about implementation maturity, reproducibility, and honest reporting
- Experimental examples remain research surfaces, not compatibility promises

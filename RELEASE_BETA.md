# Release Notes: `v5.0.0-beta.1`

## Summary

`v5.0.0-beta.1` promotes the Rust `instnct-core` lane into the main public implementation surface for INSTNCT. This release achieves **24.6% peak** next-character prediction accuracy on English text, matching the Python reference implementation.

## Highlights

- **24.6% peak accuracy** — smooth cosine-bigram fitness + 1+9 jackpot selection
- **Addition learning** — 80% accuracy on 0-4 + 0-4 from empty network (83 edges)
- **Empty start superiority** — sparse evolution builds targeted circuits (80% with 83 edges vs 64% with 3400 prefilled edges)
- Rust-first public beta surface through `instnct-core`
- Curated crate-root API for network construction, propagation, evolution, SDR input, and checkpoint persistence
- Canonical runner: `examples/evolve_language.rs`
- 150 tests, zero unsafe, full docs

## New in beta.1

- `evolution_step_jackpot()` — multi-candidate (1+N) ES selection
- `Int8Projection::raw_scores()` — full score vector for smooth fitness
- Smooth cosine-bigram fitness as default (replaces binary argmax accuracy)
- Mutation schedule: W projection 10% → 5%, channel 5% → 10%

## Run the canonical beta

```powershell
cargo run --release --example evolve_language -- <corpus-path> `
  --steps 30000 `
  --seed-count 6 `
  --report-dir target/beta-report
```

## Known limitations

- Seed variance remains high (best seed 24.6%, worst may fall below 15%)
- Addition learning works for small digits (0-4) but does not yet scale to larger ranges
- The Python reference line remains in-repo for developers; the stable beta contract is Rust

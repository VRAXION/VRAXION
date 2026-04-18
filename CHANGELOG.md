# Changelog

## v5.0.0-beta.1 (2026-04-06)

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

# Examples

This directory holds runnable example binaries for the `instnct-core` Rust
workspace. Run any one with:

```bash
cargo run --release -p instnct-core --example <name>
```

## Current mainline runner

- [`neuron_grower.rs`](neuron_grower.rs) — the bias-free Rust grower that
  ships as the current canonical code path on `main`. Driven through the
  B0 engine-freeze contract harness at
  [`tools/run_grower_regression.py`](../../tools/run_grower_regression.py)
  per [`docs/GROWER_RUN_CONTRACT.md`](../../docs/GROWER_RUN_CONTRACT.md).
- [`neuron_infer.rs`](neuron_infer.rs) — companion inference runner for
  saved grower checkpoints.
- [`byte_opcode_grower.rs`](byte_opcode_grower.rs) — byte/opcode v1 builder
  used by the B1 promotion gate at
  [`tools/run_byte_opcode_acceptance.py`](../../tools/run_byte_opcode_acceptance.py)
  per [`docs/BYTE_OPCODE_V1_CONTRACT.md`](../../docs/BYTE_OPCODE_V1_CONTRACT.md).

## Active research runners (Phase A → B → D)

The 2026-04-23 → ongoing mutation-selection / dimensionality / acceptance-
aperture / Search Aperture Function research line; driven from the
Python tools in [`tools/`](../../tools/).

- [`evolve_mutual_inhibition.rs`](evolve_mutual_inhibition.rs) — primary
  fixture for Phase A/B/B.1/D0/D0.5/D1/D2/D3/D3.1/D4 sweeps.
- [`evolve_bytepair_proj.rs`](evolve_bytepair_proj.rs) — grow-prune
  fixture used alongside `evolve_mutual_inhibition` for Phase A baseline
  and downstream verdicts.
- [`diag_phase_b_panel.rs`](diag_phase_b_panel.rs) — Phase B panel
  diagnostic.

## Reference / historical runners

- [`evolve_language.rs`](evolve_language.rs) — the language-evolution
  runner that produced the released `v5.0.0-beta.1` 24.6% next-character
  result. Retained as a released reference lane, not the active mainline.
- [`evolve_abc_char.rs`](evolve_abc_char.rs) — ABC-pipeline char-level
  integration probe.
- [`evolve_breed.rs`](evolve_breed.rs) — breed/evolution probe.
- [`evolve_bytepair.rs`](evolve_bytepair.rs) — byte-pair evolution probe.
- [`diag_bytepair.rs`](diag_bytepair.rs) — byte-pair diagnostic.
- [`diag_5way_fineweb.rs`](diag_5way_fineweb.rs) — five-way FineWeb
  diagnostic (parquet feature).
- [`extract_fineweb_txt.rs`](extract_fineweb_txt.rs) — corpus extraction
  utility (parquet feature).
- [`ablation_test.rs`](ablation_test.rs) — ablation harness.
- [`adversarial_test.rs`](adversarial_test.rs) — adversarial harness.
- [`analyze_checkpoint.rs`](analyze_checkpoint.rs) — offline checkpoint
  reader.
- [`chain_diagnosis.rs`](chain_diagnosis.rs) — cross-block chain
  diagnostic.
- [`trace_signal.rs`](trace_signal.rs) — single-signal trace utility.

## Archive

Earlier experiments from the 2026-04-17 era (addition / pocket / chip /
abstract-core v1..v4 / connectome / flybrain / mirror lines, 56 files
total) lived under `examples/archive/2026-04/` and were moved off `main`
on 2026-04-27 per [`ARCHIVE.md`](../../ARCHIVE.md). They are preserved
at the immutable content-snapshot tag
[`archives/instnct-examples-2026-04-archive-20260427`](https://github.com/VRAXION/VRAXION/tree/archives/instnct-examples-2026-04-archive-20260427).

Restore any archived example via:

```bash
git show archives/instnct-examples-2026-04-archive-20260427:instnct-core/examples/archive/2026-04/<filename>.rs
git checkout archives/instnct-examples-2026-04-archive-20260427 -- instnct-core/examples/archive/2026-04/<filename>.rs
```

## Parquet feature

Two examples require the `parquet` feature (off by default to keep the
default cold-build cheap):

```bash
cargo run --release -p instnct-core --example extract_fineweb_txt --features parquet
cargo run --release -p instnct-core --example diag_5way_fineweb --features parquet
```

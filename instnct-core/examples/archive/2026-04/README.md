# Archived Experiments — 2026-04-17

Historical experiments archived 2026-04-17. Not currently active. Represent exploration paths leading to current Beukers + quantization work.

These files are preserved for reference but are no longer part of the active research surface. They cover earlier explorations:

- **addition_*** — the binary-addition toy task that seeded early growth/freeze/stateful studies
- **abstract_core_v1..v4** — earlier iterations of the abstract-core stack (v5 is the current active version, kept in place)
- **pocket_*** — pocket/evolution ensemble exploration, superseded by Beukers gate work
- **byte_*** / **byte_opcode_*** — byte-ALU / opcode-interpreter exploration
- **grid3_*** / **connectome_*** / **flybrain*** / **breed_*** — connectivity / bio-inspired exploration
- **chip_*** / **circuit_*** / **conv_*** — chip/circuit/convolution connectivity studies
- **mirror_*** / **all_binary_mirror** — older mirror-autoencoder studies superseded by newer mirror work

Active work lives in the parent `examples/` directory:
- `diag_*.rs` — current diagnostic sweeps (Beukers gate, quantization ladders)
- `beukers_*.rs` — Beukers-gate neuron studies
- `quant_*.rs` / `quantize_*.rs` — quantization pipelines
- `extract_fineweb_txt.rs` / `parquet_fineweb` — corpus pipeline
- `abstract_core_v5_400ep.rs` — current abstract-core version
- `neuron_grower.rs` — current growing-network builder

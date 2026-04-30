# Getting Started with VRAXION

_Last updated: 2026-04-30 — against release `v5.0.0-beta.8`_

---

## What is VRAXION?

VRAXION is a research project building **INSTNCT**: a gradient-free, self-wiring neural
architecture that learns by changing its own directed graph rather than running
backpropagation through a fixed topology. The canonical implementation lives in Rust
(`instnct-core/`). A frozen byte-level lexical pipeline (L0 + L1 + Word Tokenizer V2)
and a Python deploy SDK (`Python/`) sit alongside the Rust core, exposing the same
frozen champion artifacts without any ML-framework dependency. The project is in active
beta: results are published with explicit scope labels (current mainline, validated
finding, or experimental direction) so the public story stays accurate.

For the full architecture thesis and status taxonomy, start with
[`README.md`](../README.md).

---

## Where to find the latest release

**Current public release tag:** `v5.0.0-beta.8`

- GitHub release page: https://github.com/VRAXION/VRAXION/releases/tag/v5.0.0-beta.8
- Release artifact manifest: [`docs/releases/v5.0.0-beta.8-artifacts.md`](releases/v5.0.0-beta.8-artifacts.md)
- Checkpoint name: `seed2042_improved_generalist_v1.ckpt`
- Stable local path (not tracked by Git — see note below):
  `output/releases/v5.0.0-beta.8/seed2042_improved_generalist_v1.ckpt`
- SHA256: `D63200504C5B4A6EA2134FD26E3E3D7CB75FF05884236DE2CC6E206BB4BA8D54`

> **Note on the checkpoint binary.** `.ckpt` files are listed in `.gitignore` and are
> not committed. If a GitHub release is attached for this tag, the `.ckpt` and
> `.sha256` files will be available as release assets there. Verify the checksum
> against the value above before use.

The prior release is `v5.0.0-beta.7` (Phase D9 smooth+accuracy specialist,
`seed2042_improved_v1`). The full release history is in [`CHANGELOG.md`](../CHANGELOG.md).

---

## What is in beta.8 — and what it is not

### What it is

`v5.0.0-beta.8` adds **Phase D9.2 multi-objective generalist confirmation** on top of
the beta.7 specialist checkpoint.

The D9.2 arc produced a validated H=384 research checkpoint,
`seed2042_improved_generalist_v1`, that improves all four tracked metrics (smooth,
accuracy, echo, unigram) relative to the seed2042 baseline. The D9.2b confirmation
ran 30 fresh seeds at two eval lengths (4000 and 16000) and passed all strict gates.

### What it is not — explicit scope boundary

- `seed2042_improved_generalist_v1` is a **validated H=384 research checkpoint**. It is
  not a replacement for the current public mainline grower (`neuron_grower.rs`).
- This checkpoint lives in the **direct-genome research lane** (`evolve_mutual_inhibition`
  substrate, H=384, seed2042). It does not invalidate or supersede the grower mainline.
- The **Python deploy SDK** (`Python/`) is currently scaffolding. `block_a_byte_unit`
  and `block_b_merger` have working code and tests; the higher blocks (Embedder V1,
  Nano Brain V1) are untrained scaffolds awaiting training artifacts. The Python
  `README.md` in that folder says "under construction" — that is the accurate description.
- The **Rust deploy SDK** (`Rust/`) is also a placeholder with no ported code yet.
  `instnct-core/` is the Rust research mainline; `Rust/` is a future deploy target.
- The **canonical Rust code path on `main`** remains
  `instnct-core/examples/neuron_grower.rs`.

---

## 5-Minute verification

To verify the Rust library and grower regression without the checkpoint:

```bash
cargo test -p instnct-core
python tools/run_grower_regression.py
python tools/run_byte_opcode_acceptance.py
```

To verify the Python deploy SDK (Block A + B only — Blocks C–E are not yet ported):

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1  |  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m compileall Python tools
python -m pytest Python/ -q
python tools/check_public_surface.py
```

See the full proof context in [`README.md`](../README.md#5-minute-proof).

---

## Reading order for newcomers

| Step | Document | Why |
|------|----------|-----|
| 1 | [`README.md`](../README.md) | Status taxonomy, architecture thesis, current mainline, 5-minute proof |
| 2 | [`BETA.md`](../BETA.md) | Current public-beta contract: what B0 and B1 gates require, what does not count |
| 3 | [`VALIDATED_FINDINGS.md`](../VALIDATED_FINDINGS.md) | Full evidence summary with scope labels; grower lane, D9/D9.2 research checkpoints, byte-level pipeline findings |
| 4 | [`docs/GROWER_RUN_CONTRACT.md`](GROWER_RUN_CONTRACT.md) | Canonical grower run contract (if working with the mainline builder) |
| 5 | [`docs/BYTE_OPCODE_V1_CONTRACT.md`](BYTE_OPCODE_V1_CONTRACT.md) | Byte/opcode v1 exact-translator contract (B1 promotion gate) |
| 6 | [`CHANGELOG.md`](../CHANGELOG.md) | Full per-release history |

If you want per-block Python deploy SDK detail:
[`Python/block_a_byte_unit/README.md`](../Python/block_a_byte_unit/README.md) and
[`Python/block_b_merger/README.md`](../Python/block_b_merger/README.md).

---

## What is next

Active work is in two areas. These are ongoing research directions, not scheduled
delivery commitments.

**D9.4 / causal basin work.** Phase D9.4 confirmed the beta.8 checkpoint's causal
explanation (`EDGE_THRESHOLD_COADAPTATION`) across two eval lengths. Phase D9.3a ran
a quadtree tessellation scan in the local perturbation space around the beta.8 basin.
Both are documented in `docs/research/` and in the `[Unreleased]` section of
`CHANGELOG.md`.

**D10 basin universality dossier.** Phase D10 is a conservative long-horizon evidence
gate investigating whether the beta.8 H=384 basin is local, task-specific,
scaling-promising, or universal. The dossier lives at
[`docs/research/PHASE_D10_BASIN_UNIVERSALITY_DOSSIER.md`](research/PHASE_D10_BASIN_UNIVERSALITY_DOSSIER.md).
No promotions are implied by active D10 runs; high-H scaling remains blocked until
evaluator trust and non-seed2042 wiring-prior signal both pass their gates.

**Next public milestone:** grower-based `v5.0.0 Public Beta`. The gate is the B1
byte/opcode v1 exact translator freeze; see [`BETA.md`](../BETA.md) for the full
contract.

---

## License

- Noncommercial: [`LICENSE`](../LICENSE)
- Commercial terms and brand rights: [`legal/LEGAL.md`](../legal/LEGAL.md)
- Citation: [`CITATION.cff`](../CITATION.cff)

The software license does not grant rights to use the **VRAXION** or **INSTNCT** names
or brand assets except as described in `legal/LEGAL.md`.

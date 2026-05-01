# Getting Started with VRAXION

_Last updated: 2026-05-01 — against release `v5.0.0-beta.9`_

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

**Current public release tag:** `v5.0.0-beta.9`

- GitHub release page: https://github.com/VRAXION/VRAXION/releases/tag/v5.0.0-beta.9
- Checkpoint name: `seed2042_improved_generalist_top01_v2.ckpt`
- Stable local path (not tracked by Git — see note below):
  `output/releases/v5.0.0-beta.9/seed2042_improved_generalist_top01_v2.ckpt`
- SHA256: `b76789c42f4349ee28c18ce97bc5f0811a89c9b138e6ecdb86fa55626f019ddb`
- Scope doc (also in the release folder): `output/releases/v5.0.0-beta.9/CLAIM_SCOPE.md`

> **Note on the checkpoint binary.** `.ckpt` files are listed in `.gitignore` and are
> not committed. If a GitHub release is attached for this tag, the `.ckpt` and
> `.sha256` files will be available as release assets there. Verify the checksum
> against the value above before use.

The prior release is `v5.0.0-beta.8` (D9 smooth+accuracy specialist after artifact
hardening, `seed2042_improved_generalist_v1`). The release before that is
`v5.0.0-beta.7`. The full release history is in [`CHANGELOG.md`](../CHANGELOG.md).

---

## What is in beta.9 — and what it is not

### What it is

`v5.0.0-beta.9` adds **Phase D10u state-anchored generalist confirmation** on top of
the beta.8 generalist baseline.

The D10u arc produced a release-candidate research checkpoint,
`seed2042_improved_generalist_top01_v2`, that passed the full D10r-v8 adversarial
gate (state identity, projection-null, and shared-shuffle controls) at
`eval_len=16000` across 30 fresh evaluation seeds, sharded into 6 independent runs:
30/30 pass, 0/30 fail, no blocker. Min trusted_mo_ci_low: +0.084493. Min
real_mo_ci_low: +0.178087.

### What it is not — explicit scope boundary

- `seed2042_improved_generalist_top01_v2` is a **release-candidate research
  checkpoint**. It is not a replacement for the current public mainline grower
  (`neuron_grower.rs`).
- This checkpoint lives in the **direct-genome research lane** (`evolve_mutual_inhibition`
  substrate, H=384, seed2042). It does not invalidate or supersede the grower mainline.
- **Cross-seed and cross-H generalization is not yet established.** D10b showed
  the recipe is seed-sensitive (seed_2042 found a basin; seeds 42, 1042, 3042, 4042
  did not under shallow scout). H=512 / H=8192 scaling remain blocked.
- **CPU/Rust cross-check is the next gate.** The 16k confirm ran on the
  Python/GPU eval lane. A second-runtime cross-check is recommended before any
  final ship decision.
- The **Python deploy SDK** (`Python/`) is partial. `block_a_byte_unit`,
  `block_b_merger`, and `block_c_embedder` have working code and tests; the
  higher blocks (Embedder V1, Nano Brain V1) are untrained scaffolds awaiting
  training artifacts.
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

# VRAXION

VRAXION is building **INSTNCT**: a gradient-free self-wiring architecture that learns by changing its own directed graph instead of running backpropagation through a fixed topology.

> **Core thesis**
>
> **Inference emerges as the fixed point of destructive interference.**
>
> Signal enters a structured recurrent substrate, incompatible propagation paths cancel through destructive interference, and the surviving pattern — the fixed point — is read out as inference. This is a research thesis for the architecture line, not a separate claim of achieved sentience.

This repository is meant to be a credible front door for technical buyers and engineers. It should let a first-time reader answer five things quickly:

1. what VRAXION is,
2. why the architecture is different,
3. what is actually proven,
4. what the current canonical code path is,
5. how to verify one claim in minutes.

## Release Snapshot

- **Current public release tag:** [`v5.0.0-beta.9`](https://github.com/VRAXION/VRAXION/releases/tag/v5.0.0-beta.9) (Phase D10u state-anchored wiring search + `top_01` release-candidate research checkpoint — 30/30 seeds pass adversarial 16k gate, H=384 only, not a mainline replacement)
- **Prior release:** [`v5.0.0-beta.8`](https://github.com/VRAXION/VRAXION/releases/tag/v5.0.0-beta.8) — Phase D9.2 multi-objective confirmation + `seed2042_improved_generalist_v1` validated H=384 research checkpoint
- **Next public milestone:** grower-based `v5.0.0 Public Beta`
- **Current mainline code path on `main`:** [`instnct-core/examples/neuron_grower.rs`](instnct-core/examples/neuron_grower.rs)
- **Python deploy SDK:** [`Python/`](Python/) — Block A + B, pure numpy, no ML framework dependency
- **Historical Python research lane:** frozen at tag [`archives/python-research-20260420`](https://github.com/VRAXION/VRAXION/tree/archives/python-research-20260420) (was `instnct/` — archived 2026-04-20 after migration to Rust `instnct-core/`)

## Why This Architecture Is Different

INSTNCT is built around a small set of unusual choices:

- **Stagewise self-wiring**: the graph grows neuron by neuron instead of optimizing a fixed dense topology.
- **Scout-first search**: a cheap all-signal probe ranks promising parents and pair interactions before exhaustive ternary search.
- **Bias-free threshold neurons**: the persistent grower now stores neurons directly as `dot >= threshold`, without redundant bias search.
- **Append-only evidence**: canonical runs are expected to emit reproducible evidence bundles and resumable state, not just ad hoc logs.

The canonical grower contract is [`docs/GROWER_RUN_CONTRACT.md`](docs/GROWER_RUN_CONTRACT.md).

## Status Taxonomy

To keep the public story truthful, this repo uses three labels consistently:

- **Current mainline**: what is actually shipped in code on `main`.
- **Validated finding**: a result backed by a concrete experiment, but not yet promoted into the canonical code path.
- **Experimental branch**: an active build target or design direction that is not yet a validated default.

If code and docs disagree, **code wins for “Current mainline.”**

The repo-tracked docs are the canonical public source. The GitHub wiki is a secondary mirror, not an independent source of truth.

Historical archives are preserved as immutable tags (no long-lived archive branches remain). See [`ARCHIVE.md`](ARCHIVE.md) for the full list, including the previous-era surface freezes and per-cleanup branch-head snapshots.

Retired line names and older local folders belong in [Project Timeline](https://github.com/VRAXION/VRAXION/wiki/Timeline-Archive), not in the current front-door stack.

## Current State

### Current mainline

- The live canonical path on `main` is [`instnct-core/examples/neuron_grower.rs`](instnct-core/examples/neuron_grower.rs).
- The grower is a bias-free threshold builder with scout-ranked parent search, resumable state, and append-only checkpoint snapshots.
- The old Rust language runner remains the latest released public tag, but it is no longer the current mainline code path on `main`.
- The Python `graph.py` lane remains in-repo as historical reference/support, not the public mainline.
- The next promotion gate is byte/opcode v1 with a frozen exact translator; until that lands, the grower is the active mainline builder and beta-prep surface.

### Evidence snapshot

- **Bias-free threshold grower** is the canonical persistent grower representation on `main`; redundant bias search was removed from state and search.
- **Scout oracle** is part of the mainline builder: single-signal ranking, connect-all probe, and pair-lift shortlist all run before ternary search.
- **Non-strict accept gate** unlocked compositional stepping-stones; `four_parity` now reaches 100% instead of stalling on equal-val intermediates.
- **Released beta.1 language-evolution result** (`24.6%` next-character prediction) remains a shipped public result, but it is now a released reference lane rather than the active mainline path on `main`.

Raw run dumps, archived sweeps, and older exploratory surfaces are preserved at archive tags listed in [`ARCHIVE.md`](ARCHIVE.md), not on active `main`.

The canonical evidence summary lives in [`VALIDATED_FINDINGS.md`](VALIDATED_FINDINGS.md).

### Experimental branch

- The current next candidate is **byte/opcode v1**: `1 byte + 4 opcode -> 1 byte` with a frozen exact translator over grower latent.
- It is the main promotion gate to public beta, not yet frozen as the live contract.

## 5-Minute Proof

### Rust (primary surface)

```bash
cargo test -p instnct-core
python tools/run_grower_regression.py
python tools/run_byte_opcode_acceptance.py
```

### Python (reference surface)

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python -m compileall Python tools
python -m pytest Python/ -q
python tools/check_public_surface.py
```

These commands verify:

- the Rust library compiles and all tests pass,
- the grower regression bundle is reproducible and public-facing,
- the byte/opcode v1 exact translator export/reload path is reproducible and exact,
- the Python deploy SDK (`Python/` — Block A + B) runs lossless end-to-end,
- the public-facing docs agree with the canonical code path.

## Read Next

- [`VALIDATED_FINDINGS.md`](VALIDATED_FINDINGS.md) — canonical evidence summary
- [`docs/BYTE_OPCODE_V1_CONTRACT.md`](docs/BYTE_OPCODE_V1_CONTRACT.md) — byte/opcode v1 exact translator contract
- [`Python/block_a_byte_unit/README.md`](Python/block_a_byte_unit/README.md) + [`Python/block_b_merger/README.md`](Python/block_b_merger/README.md) — deploy SDK per-block entry
- [VRAXION architecture page (INSTNCT)](https://github.com/VRAXION/VRAXION/wiki/INSTNCT-Architecture)
- [Issue #114](https://github.com/VRAXION/VRAXION/issues/114) — current next build target

## License

- Noncommercial: [LICENSE](https://github.com/VRAXION/VRAXION/blob/main/LICENSE)
- Commercial terms: [legal/LEGAL.md](legal/LEGAL.md)
- Brand and trademark notice: [legal/LEGAL.md](legal/LEGAL.md)
- Citation: [CITATION.cff](CITATION.cff)

The software license does not grant rights to use the **VRAXION** or **INSTNCT** names, logos, or brand assets except as described in [legal/LEGAL.md](legal/LEGAL.md).

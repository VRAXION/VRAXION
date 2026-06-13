# Contributing

## Contributor Terms

Canonical packaged contributor terms are mirrored at
[`docs/legal/VRAXION_CONTRIBUTOR_TERMS_V1.md`](docs/legal/VRAXION_CONTRIBUTOR_TERMS_V1.md).

By submitting a contribution to VRAXION, you agree that your contribution is
licensed to the project under the terms below.

### 1. Your contribution

"Contribution" means any code, documentation, model configuration, test, issue
text, pull request, design, comment, patch, benchmark, script, or other material
you intentionally submit to the project.

### 2. Rights you grant

You grant the VRAXION project, Daniel Kenessy, and any future VRAXION legal
entity or foundation a worldwide, perpetual, irrevocable, non-exclusive,
royalty-free license to:

- use your Contribution;
- copy it;
- modify it;
- publish it;
- distribute it;
- sublicense it as part of VRAXION;
- relicense it under the VRAXION Community Source License, later VRAXION
  licenses, and separate commercial agreements;
- enforce rights in the combined project;
- include it in VRAXION-powered commercial hosted licensing;
- include it in the royalty model funding the VRAXION Forever Prize.

### 3. Patent grant

You grant a patent license for any patent claims you can license that would
necessarily be infringed by your Contribution as submitted or as combined with
the project.

If you or your affiliate bring a patent claim against the project or its users
over your Contribution, your patent grant may terminate.

### 4. Your promise

You promise that:

- you have the right to submit the Contribution;
- the Contribution is your original work or you have permission to submit it;
- the Contribution does not knowingly include third-party code or content that
  conflicts with the project license;
- you will identify any third-party license obligations known to you.

### 5. No payment

Unless a separate written agreement says otherwise, contributions are voluntary
and unpaid.

Submitting a Contribution does not give you a royalty, ownership interest,
governance right, trademark right, or claim to VRAXION Forever Prize funds.

### 6. Why this exists

VRAXION uses a community/free-use plus commercial-royalty license model.

That model only works if the project can accept contributions without later
losing the ability to:

- keep the code public;
- allow free community use;
- license commercial hosted/API/subscription use;
- fund the VRAXION Forever Prize;
- maintain clear legal continuity after founder death or redirect events.

### 7. Certificate of origin

By submitting a Contribution, you certify:

```text
I have the right to submit this Contribution.
I intentionally license it under the VRAXION Contributor Terms.
I understand that the project may use, modify, distribute, sublicense,
and commercially license it as part of VRAXION.
```

## Repository Contribution Guide

VRAXION is a **public technical repo**. Rigor and reproducibility matter, but so does clarity: the public repo should be understandable to engineers, technical buyers, and contributors without reverse-engineering issue traffic.

## Public Truth Rules

Use the repo taxonomy consistently:

- **Current mainline** = shipped code on `main`
- **Validated finding** = experiment-backed result not yet promoted into the canonical code path
- **Experimental branch** = active build target or prototype direction

If a setting or training recipe is not in the canonical code path, do **not** describe it as the live mainline default.

## Repo Map

### Rust mainline (`instnct-core/`)

- `instnct-core/src/`: library crate — network, propagation, evolution, fitness, eval, corpus
- `instnct-core/examples/neuron_grower.rs`: canonical grower mainline (current self-wiring frontier)
- `instnct-core/examples/evolve_language.rs`: beta language-evolution runner (first published in `v5.0.0-beta.1`; retained as a released reference lane through current beta)
- `instnct-core/examples/`: experimental research examples (not part of compatibility promise)
- `instnct-core/tests/fixtures/`: test corpora and fixtures

### Python deploy SDK (`Python/`)

Pure numpy, zero ML framework dependency. Reads champion artifacts from the repo's `output/` directory.

- `Python/block_a_byte_unit/`: Block A — byte unit, L0, 8 → 16 → 8 tied-mirror autoencoder
- `Python/block_b_merger/`: Block B — byte-pair merger, L1, 32 → 81 → 32 single-W mirror
- `Python/__init__.py`: top-level exports (`ByteEncoder`, `L1Merger`)

### Rust deploy SDK (`Rust/`)

Parallel to `Python/`: same champion artifacts, pure std + serde, zero ML deps.

- `Rust/src/block_a_byte_unit/`: Block A
- `Rust/src/block_b_merger/`: Block B

### Shared

- `docs/`: GitHub Pages website (Home + Blocks A-E + Legacy + Wiki mirror)
- `tools/`: build scripts, sweep probes, regression runners, acceptance checks
- `VALIDATED_FINDINGS.md`: canonical evidence summary
- `docs/VERSION.json`: public release identity source of truth

### Archives

Historical research lanes live on tags, not `main`:

- `archives/python-research-20260420`: frozen `instnct/` lane (pre-Rust migration)
- `archives/tools-legacy-diag-20260420`: pre-Fázis-6 `tools/` tree (79 scripts)

Restore any file via `git show archives/<name>:path/to/file` or `git checkout archives/<name> -- path/to/file`.

## Where To Put Things

- **Rust core changes**: `instnct-core/src/` (run `cargo test -p instnct-core` and `cargo clippy`)
- **Rust experiments**: `instnct-core/examples/` (not part of API contract)
- **Python deploy SDK changes**: `Python/block_a_byte_unit/` or `Python/block_b_merger/` (run `python -m pytest Python/ -q`)
- **Rust deploy SDK changes**: `Rust/src/block_a_byte_unit/` or `Rust/src/block_b_merger/`
- **Build / sweep tooling**: `tools/` (Python only)
- **Public-facing docs**: root docs plus `docs/`
- **Local corpora**: `.traindat` files stay local-only and gitignored

## Documentation Governance

Repo-tracked docs are the **canonical public source**. The GitHub wiki checkout at `VRAXION.wiki/` is mirror output only, not an independent source of truth.

When public-facing docs change:

1. Edit the repo-tracked source first
2. Run `python tools/sync_wiki_from_repo.py`
3. Run `python tools/check_public_surface.py`
4. Only then treat the update as stable public truth

Versioning source of truth: `docs/VERSION.json`. Citation source of truth: `CITATION.cff`.

## Pull Request Requirements

Every PR should include:

- **Why**: what problem it solves
- **What changed**: concise behavior and file summary
- **How to verify**: exact commands

Guardrails:

- Do not commit run artifacts, logs, checkpoints, or large binaries
- Keep changes small and reversible where possible
- If a change affects metrics, state whether the result is current mainline, validated finding, or experimental only

Recommended verification commands:

```bash
# Rust mainline
cargo test -p instnct-core
cargo clippy --all-targets -- -D warnings
cargo doc --no-deps

# Python deploy SDK
python -m compileall Python tools
python -m pytest Python/ -q

# Public surface consistency
python tools/check_public_surface.py
```

## Reproducibility

If you report a metric, include at minimum: commit hash, exact command line, seed list, relevant config values, and output summary. If the claim is important enough for the front door, mirror it into `VALIDATED_FINDINGS.md`.

## Archive Policy

`main` stays intentionally small. Only the active self-wiring graph line belongs here.

**What does not stay on `main`:** superseded repo eras, half-ready research detours, ephemeral review branches, one-off experimental integration branches.

**Practical rule:** if a line is not part of the current self-wiring mainline, archive it. If it still matters historically, keep a tag. If it neither matters historically nor supports the current line, delete it.

---

## Local Setup

```bash
# Rust mainline
cargo build --workspace

# Python deploy SDK
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install pytest  # dev-only, not a runtime dep
```

## Testing

```bash
# Rust
cargo test -p instnct-core
cargo clippy --all-targets -- -D warnings

# Python
python -m compileall Python tools
python -m pytest Python/ -q
python tools/run_grower_regression.py
python tools/run_byte_opcode_acceptance.py

# Public surface consistency
python tools/check_public_surface.py
```

## Filing an Issue

Include: commit hash (`git rev-parse HEAD`), exact command line that reproduces the problem, OS + Python/Rust version, and a minimal reproducer. If the issue involves a metric claim, attach the raw output summary.

## License

Contributions are under the project's [VRAXION Community Source License 1.0](LICENSE) and the Contributor Terms above.

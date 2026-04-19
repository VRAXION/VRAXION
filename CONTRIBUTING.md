# Contributing

VRAXION is a **public technical repo**. Rigor and reproducibility matter, but so does clarity: the public repo should be understandable to engineers, technical buyers, and contributors without reverse-engineering issue traffic.

## Public Truth Rules

Use the repo taxonomy consistently:

- **Current mainline** = shipped code on `main`
- **Validated finding** = experiment-backed result not yet promoted into the canonical code path
- **Experimental branch** = active build target or prototype direction

If a setting or training recipe is not in the canonical code path, do **not** describe it as the live mainline default.

## Repo Map

### Rust (primary surface — `instnct-core/`)

- `instnct-core/src/`: library crate — network, propagation, evolution, fitness, eval, corpus
- `instnct-core/examples/evolve_language.rs`: canonical beta runner
- `instnct-core/examples/`: experimental research examples (not part of compatibility promise)
- `instnct-core/tests/fixtures/`: test corpora and fixtures

### Python (reference surface — `instnct/`)

- `instnct/model/`: self-wiring graph reference implementation
- `instnct/lib/`: shared scoring, data, and logging helpers
- `instnct/tests/`: stress tests, sweeps, probes, and benchmark scripts
- `instnct/recipes/`: training recipes and A/B experiments

### Shared

- `docs/`: GitHub Pages website
- `VALIDATED_FINDINGS.md`: canonical evidence summary
- `docs/VERSION.json`: public release identity source of truth

## Where To Put Things

- **Rust core changes**: `instnct-core/src/` (run `cargo test --lib` and `cargo clippy`)
- **Rust experiments**: `instnct-core/examples/` (not part of API contract)
- **Python model changes**: `instnct/model/`
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
# Rust (primary surface)
cargo test -p instnct-core
cargo clippy --all-targets -- -D warnings
cargo doc --no-deps

# Python (reference surface)
python -m compileall instnct tools
python instnct/tests/test_model.py

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
# Rust (primary surface)
cargo build --workspace

# Python (reference surface)
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Testing

```bash
# Rust
cargo test -p instnct-core
cargo clippy --all-targets -- -D warnings

# Python
python -m compileall instnct tools
python instnct/tests/test_model.py
python tools/run_grower_regression.py
python tools/run_byte_opcode_acceptance.py

# Public surface consistency
python tools/check_public_surface.py
```

## Filing an Issue

Include: commit hash (`git rev-parse HEAD`), exact command line that reproduces the problem, OS + Python/Rust version, and a minimal reproducer. If the issue involves a metric claim, attach the raw output summary.

## License

Contributions are under the project's [PolyForm Noncommercial 1.0.0](LICENSE) license.

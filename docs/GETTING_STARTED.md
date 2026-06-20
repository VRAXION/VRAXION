# Getting Started with VRAXION

_Last updated: 2026-06-21_

## Start Here

```text
branch = main
current_release = v6.1.7
public_surface = sanitized prior-art and release documentation
frontier_work = private review and release-candidate staging
```

This public repository is no longer a frontier experiment warehouse. It keeps
the reviewed public source, claim boundaries, release notes, and high-level
research summaries. Raw datasets, full run ledgers, local artifact samples,
probe runners, and private training traces are intentionally excluded from the
current public tree.

## Repository Layout

| Path | Purpose |
|---|---|
| `vraxion-runtime/` | Public Rust runtime slice retained for historical release continuity |
| `docs/research/*.md` | Sanitized contract/result summaries and claim boundaries |
| `docs/` | Public docs and GitHub Pages front door |
| `scripts/audit_public_surface.py` | Public-surface leak and boundary audit |
| `CODEX_HANDOVER.md` | Fresh-session project handover |
| `target/` | Local generated artifacts; ignored by Git |

## Quick Checks

```powershell
cargo fmt --check -p vraxion-runtime
cargo clippy -p vraxion-runtime --all-targets -- -D warnings
cargo test --workspace
python -m compileall -q scripts
python scripts/audit_public_surface.py
```

## Current Public Boundary

Allowed public claim:

```text
VRAXION v6 preserves a public, sanitized evidence trail for scoped runtime,
operator-library, curriculum, and governance experiments.
```

Not claimed here:

```text
production readiness
final training completion
open-domain assistant capability
Gemma/GPT-like generation
PermaCore or TrueGolden promotion
consciousness or sentience
```

## Reading Order

| Step | Document | Why |
|---|---|---|
| 1 | [`README.md`](../README.md) | Public front-door summary |
| 2 | [`docs/CURRENT_STATUS.md`](CURRENT_STATUS.md) | Current status and claim boundary |
| 3 | [`docs/CURRENT_CAPABILITIES.md`](CURRENT_CAPABILITIES.md) | What the public project currently claims |
| 4 | [`VALIDATED_FINDINGS.md`](../VALIDATED_FINDINGS.md) | Validated finding chain |
| 5 | [`CODEX_HANDOVER.md`](../CODEX_HANDOVER.md) | Fresh-session handover |

## Public Surface Audit

The public audit intentionally fails on tracked raw samples, JSONL ledgers,
local dataset paths, private frontier names, and operational probe surfaces.
The expected first-pass state is:

```text
public audit failure_count = 0
tracked artifact samples = 0
tracked JSONL ledgers = 0
tracked operational probe runners = 0
```

Warnings are allowed when they identify high-level research-result summaries,
legacy runtime binaries, or named dataset families without exposing local paths
or raw samples.

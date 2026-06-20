# VRAXION

VRAXION is a Rust-first research project for a governed Operator runtime.

The active design direction is **α-Sync**: observations are translated into
bounded proposals, proposals pass through Agency/governance checks, and only
accepted proposals may update runtime state or output.

```text
observation
-> scoped Operator / Prismion evaluation
-> proposal field
-> Agency / governance decision
-> commit, reject, defer, ask, or render
```

This repository is the sanitized public release and evidence surface. It is not
the private frontier workspace.

## Current Public Status

Latest public release: [v6.1.7](https://github.com/VRAXION/VRAXION/releases/tag/v6.1.7)

Current public evidence anchor: `E136S atomic multiwrite default-route switch canary guard`

Public runtime surface: [`vraxion-runtime/`](vraxion-runtime/)

The public tree is currently a transitional release surface. A cleaner compiled
Rust package is being hardened privately before it replaces the older public
runtime layout.

## What This Repository Claims

Allowed public claim:

```text
VRAXION is a governed Operator runtime with scoped, evidence-backed mechanics
for proposal generation, Agency-gated commits, bounded routing, and guarded
runtime writes.
```

Not claimed:

```text
open-domain LLM
production assistant
general autonomous intelligence
trained neural model weights
PermaCore / TrueGolden memory
medical, legal, or safety-critical deployment readiness
```

## Why It Is Different

VRAXION does not present itself as a fixed transformer checkpoint. The public
runtime is organized around:

- small scoped Operators instead of unrestricted global actions
- proposal-first state updates instead of direct writes
- Agency/governance gates before commit
- explicit claim boundaries and no-call/defer behavior
- audit-friendly releases and provenance documents

## Public/Private Boundary

This public repository may contain:

- release notes
- public claim boundaries
- compact research summaries
- license, citation, security, and provenance documents
- safe Rust runtime surfaces selected for publication

This public repository must not contain:

- raw datasets
- full run traces
- hidden-checker training cells
- private frontier plans
- local machine paths
- API keys, secrets, or credentials
- executable dataset downloaders or private training runners

See [docs/PUBLIC_SURFACE_POLICY.md](docs/PUBLIC_SURFACE_POLICY.md) and
[PUBLIC_BOUNDARY.md](PUBLIC_BOUNDARY.md).

## Repository Map

| Path | Purpose |
|---|---|
| [`vraxion-runtime/`](vraxion-runtime/) | Current transitional public Rust runtime surface. |
| [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md) | Compact current status and evidence boundary. |
| [`docs/CURRENT_CAPABILITIES.md`](docs/CURRENT_CAPABILITIES.md) | What the public project can and cannot claim. |
| [`docs/research/`](docs/research/) | Historical public research contracts/results kept for provenance. |
| [`docs/releases/v6.1.7.md`](docs/releases/v6.1.7.md) | Release note for the current public release. |
| [`legal/`](legal/) | License, contributor, and provenance legal material. |
| [`scripts/audit_public_surface.py`](scripts/audit_public_surface.py) | Public-surface safety audit. |

## Build And Audit

Run the public Rust checks:

```powershell
cargo fmt --all -- --check
cargo test --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

Run the public-surface audit:

```powershell
python scripts/audit_public_surface.py
```

Expected hard gate:

```text
failure_count should be zero
```

Audit warnings may remain for historical research files while the public surface
is being compacted.

## Release Direction

The next intended public milestone is a smaller Rust package exported from the
private α-Sync core after release-blocker review. That package should replace
the current transitional runtime with a cleaner boundary:

```text
minimal compileable Rust workspace
safe examples
license / citation / security docs
no GUI
no private skillstore
no raw datasets
no training traces
```

## Links

- [Current status](docs/CURRENT_STATUS.md)
- [Current capabilities](docs/CURRENT_CAPABILITIES.md)
- [Validated findings](VALIDATED_FINDINGS.md)
- [Public training boundary](docs/PUBLIC_TRAINING_BOUNDARY.md)
- [Public surface policy](docs/PUBLIC_SURFACE_POLICY.md)
- [Security policy](SECURITY.md)
- [Citation](CITATION.cff)
- [License](LICENSE)

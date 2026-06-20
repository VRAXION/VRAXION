# VRAXION Public Handover

Last updated: 2026-06-21

This file is the public-repository handover. It is intentionally compact.

## Repository Role

```text
VRAXION/VRAXION = sanitized public evidence and release surface
private frontier repo = unreleased implementation, raw datasets, run traces, and training recipes
```

Do not treat this repository as the live frontier workspace. Public changes
should preserve prior-art value, release provenance, and claim boundaries while
avoiding raw data, private training recipes, or operational probe details.

## Current Public State

Current release: `v6.1.7`.

Active public runtime surface: `vraxion-runtime/`.

Current evidence anchor: `E136S atomic multiwrite default-route switch canary guard`.

Public audit hard failures after cleanup pass 001: `0`.

The public runtime surface is historical and transitional. The intended next
public package is a clean Rust export from the private α-Sync core, after the
release-blocker review cycle is closed.

## Current Claim Boundary

Allowed public claim:

```text
VRAXION is a governed Operator runtime with scoped, evidence-backed mechanics
for proposal generation, agency checks, guarded commits, and bounded rendering.
```

Forbidden public claims:

```text
open-domain LLM
general autonomous intelligence
production-ready assistant
PermaCore / TrueGolden memory
unrestricted language understanding
trained neural model weights
```

## Public Maintenance Rules

```text
1. No raw datasets, JSONL run traces, private paths, or full mutation logs.
2. No executable dataset downloaders or frontier training runners.
3. Keep public docs as summaries, not reconstruction manuals.
4. Keep release docs, license, citation, security, and public boundary docs.
5. Use the public audit before every public push.
```

Run:

```powershell
python scripts/audit_public_surface.py
```

Expected public-release baseline:

The expected hard gate is `failure_count` equal to zero.

## Where To Look

- [README.md](README.md): public overview and release links
- [docs/CURRENT_STATUS.md](docs/CURRENT_STATUS.md): compact current status
- [docs/CURRENT_CAPABILITIES.md](docs/CURRENT_CAPABILITIES.md): allowed capability boundary
- [docs/PUBLIC_SURFACE_POLICY.md](docs/PUBLIC_SURFACE_POLICY.md): public/private split
- [PUBLIC_BOUNDARY.md](PUBLIC_BOUNDARY.md): release-surface constraints
- [SECURITY.md](SECURITY.md): security reporting and expectations

## Next Public Work

The next public step should be a generated clean Rust package candidate, not a
manual mirror of the private frontier repo.

# VRAXION Current Status

_Last updated: 2026-06-21_

## Official Status

Public release: `v6.1.7`.

Public source of truth: `main`.

Current public role: sanitized evidence and release surface.

Current evidence anchor: `E136S atomic multiwrite default-route switch canary guard`.

The public repository is intentionally not the frontier workspace. New private
experiments, raw datasets, full traces, and training recipes stay outside this
repository until a separate public-export review approves them.

## Mainline Summary

The public mainline records a progression from:

```text
Rust runtime readiness
-> governed Operator library
-> scoped calculator and text-route skills
-> rank / survival / no-harm gates
-> output-field and idle-tick mechanics
-> guarded atomic multiwrite canary
```

The current public runtime surface is `vraxion-runtime/`. It remains available
for provenance and auditability while the private α-Sync Rust package is being
hardened for a cleaner future public candidate.

## Evidence Boundary

The current public evidence supports only scoped runtime-mechanic claims:

```text
proposal generation
agency-gated commit flow
guarded runtime writes
bounded Operator routing
scoped no-call / defer behavior
deterministic smoke and canary checks
```

It does not support claims of:

```text
open-domain language understanding
production-ready assistant behavior
general intelligence
trained neural model weights
PermaCore / TrueGolden memory
unrestricted autonomous self-improvement
```

## Public Cleanup State

Cleanup pass 001 removed public artifact samples, JSONL traces, probe scripts,
operator dashboards, and direct local dataset paths from the current tree.

Cleanup pass 002 further removes or compacts public material that looked like
dataset rebuild instructions, hidden-oracle training cells, or oversized
internal handover/state dumps.

Before public release work, run:

```powershell
python scripts/audit_public_surface.py
```

The expected hard gate is:

The expected hard gate is `failure_count` equal to zero.

Warnings may remain for historical result documents, but warnings should trend
down as old research contracts are compacted into release summaries.

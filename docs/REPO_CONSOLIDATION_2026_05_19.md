# Repo Consolidation - 2026-05-19

## Goal

Clean the public `main` branch after the 099 bounded local/private release-ready stop condition and the 100 assistant capability-scale probe.

The cleanup keeps the current release/evidence runway visible, while removing obsolete scratch code from active `main`.

## Archive Branch

Before deletion, the full pre-cleanup tree was preserved online:

```text
archive/pre-consolidation-20260519-main-snapshot
```

Use that branch for any removed historical scratch code or legacy probe runner.

## Removed From Active Main

The following code classes were removed from active `main`:

- `tools/_scratch/` historical one-off research scripts
- pre-bounded/non-current `scripts/probes/` runners that are outside the retained 078-100 proof line

The retained probe line is:

```text
078, 079, 079B, 080, 081, 082,
083, 084, 085, 086, 087, 088, 089, 089B,
091, 092, 093, 094, 094B, 095, 096, 097, 098, 099, 100
```

## Kept On Main

The cleanup intentionally keeps:

- `instnct-core/`
- `tools/instnct_service_alpha/`
- `tools/instnct_deploy/`
- `Python/` and `Rust/` reference/deploy surfaces
- `docs/research/` evidence docs
- `docs/wiki/` mirrored wiki source
- `output/` champion artifacts still referenced by active Python/Rust/public-surface loaders
- root `LICENSE`

## Why `output/` Was Not Deleted

The tracked `output/` champion artifacts are still referenced by Python/Rust loaders and public-surface checks. Removing them would require a separate migration of those loaders and docs.

They are therefore kept on `main` for now, even though generic generated output remains ignored by `.gitignore`.

## Current Claim Boundary

Allowed:

```text
local/private bounded-domain release-ready baseline
runner-local assistant capability-scale research signal improved
```

Forbidden:

```text
production public AI service
public API
hosted SaaS
GPT-like assistant readiness
open-domain chat readiness
safety-aligned production system
INSTNCT/AnchorRoute proven as open-domain LM winner
```

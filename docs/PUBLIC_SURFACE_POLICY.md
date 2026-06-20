# VRAXION Public Surface Policy

_Last updated: 2026-06-19_

This repository is the public VRAXION prior-art, release, and sanitized
evidence surface.

The private frontier workspace is referenced publicly only as
`private_frontier_repo`.

## Public Repository Role

The public repository should contain:

- reviewed result summaries
- checker summaries
- reduced sample artifacts
- runtime source that is intended to be public
- reproducible safety and claim-boundary documentation
- release notes and provenance records
- limited artifact samples needed to support public claims

The public repository should not contain:

- raw datasets
- private roadmaps
- full mutation traces
- full red-team corpora
- sensitive failure prompts
- exact private exploit or jailbreak recipes
- secrets, tokens, private keys, or local machine identifiers
- unreviewed frontier implementation details

## Private Frontier Role

The private frontier repository is for:

- unreleased probes
- private implementation work
- frontier planning
- sensitive failure analysis
- private safety and red-team detail
- sanitized-to-public staging

Nothing from the private frontier repository becomes a public claim until it has
gone through explicit export review.

## Raw Data Rule

Raw or sensitive data stays outside GitHub unless explicitly encrypted and
approved for storage.

Current local private data root alias:

```text
local_private_data
```

Every dataset used for frontier work should have a small manifest recording:

```text
source
license_or_permission
sensitivity
allowed_use
public_export_status
storage_location
```

## Export Rule

Before moving anything public, check:

```text
1. Is it necessary for public evidence, release, or prior-art value?
2. Is it sanitized down to claim-supporting minimum detail?
3. Are raw data, secrets, local paths, and sensitive prompts absent?
4. Is the claim boundary explicit?
5. Can the public artifact stand without leaking frontier-only recipes?
```

If any answer is unclear, keep it private.

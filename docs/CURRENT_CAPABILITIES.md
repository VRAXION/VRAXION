# Current Capabilities

This public repository is a public SDK/docs surface, not a runnable engine release.

## Publicly Included

- `alphasync-core`: deterministic public SDK primitives for bounded fabric experiments.
- `alphasync-runtime`: runtime artifact writing, safe summaries, and public validation helpers.
- Public docs for delivery, trademarks, current status, and Pages surfaces.
- Public audit/export guards used by CI to keep private training and engine material out of the repository.
- INSTNCT Pages preview: a static product surface describing the T1 Reflex Engine target, Exact Mode refusal, and Proof Pack posture.

## Publicly Not Included

- Implementation internals and non-public release materials.
- Non-public runtime materials.
- Production hosted API or SaaS deployment.
- Signed runnable T1 binary artifact.
- Non-public operational tooling or diagnostic surfaces.

## Current INSTNCT Claim Scope

INSTNCT is presented publicly as the first VRAXION engine line, with the T1 Reflex Engine as the first concrete target:

- local-first
- scoped to avoid hosted LLM/model-wrapper dependencies
- Path Selector behavior should choose a supported path before answering
- Exact Mode should refuse unsupported answers
- imagination mode, when present, must be explicit and labeled
- benchmark numbers are scoped selector claims, not universal latency claims

The current public page is an engine preview and verification surface. The T1
Proof Pack is not yet published as a runnable public artifact.

## Current Public Release

Latest public release:
[`public-sdk-p11-20260629`](https://github.com/VRAXION/VRAXION/releases/tag/public-sdk-p11-20260629)

That release contains the public SDK/docs surface and does not include
non-public implementation materials or a signed binary package.

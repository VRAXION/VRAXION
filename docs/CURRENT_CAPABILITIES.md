# Current Capabilities

This public repository is a public SDK/docs surface, not a runnable engine release.

## Publicly Included

- `alphasync-core`: deterministic public SDK primitives for bounded fabric experiments.
- `alphasync-runtime`: runtime artifact writing, safe summaries, and public validation helpers.
- Public docs for delivery, trademarks, current status, and Pages surfaces.
- Public audit/export guards used by CI to keep private training and engine material out of the repository.
- INSTNCT Pages preview: a static, local-first product surface describing the T1 release target and verification posture.

## Publicly Not Included

- Implementation internals and non-public release materials.
- Private binary internals.
- Production hosted API or SaaS deployment.
- Signed runnable T1 binary artifact.
- Training service, mutation service, skill persistence store, or private diagnostic surfaces.

## Current INSTNCT Claim Scope

INSTNCT is presented publicly as a T1 Reflex Class release target:

- local-first
- scoped to avoid hosted LLM/model-wrapper dependencies
- exact mode should refuse unsupported answers
- imagination mode, when present, must be explicit and labeled
- benchmark numbers are scoped selector claims, not universal latency claims

The current public page is a release and verification surface. The binary is not
yet published as a runnable public artifact.

## Current Public Release

Latest public release:
[`public-sdk-p11-20260629`](https://github.com/VRAXION/VRAXION/releases/tag/public-sdk-p11-20260629)

That release contains the public SDK/docs surface and does not include
non-public implementation materials or a signed binary package.

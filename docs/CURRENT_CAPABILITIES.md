# Current Capabilities

This public repository is a boundary surface, not the private engine release.

## Publicly Included

- `alphasync-core`: deterministic public SDK primitives for bounded fabric experiments.
- `alphasync-runtime`: runtime artifact writing, safe summaries, and public validation helpers.
- Public docs for delivery boundaries, trademark boundaries, current status, and Pages surfaces.
- Public audit/export guards used by CI to keep private training and engine material out of the repository.
- INSTNCT Pages preview: a static, local-first product surface describing the T1 release target and verification posture.

## Publicly Not Included

- Private engine source.
- Private binary internals.
- Production hosted API or SaaS deployment.
- Signed runnable T1 binary artifact.
- Training service, mutation service, skill persistence store, or private diagnostic surfaces.

## Current INSTNCT Claim Boundary

INSTNCT is presented publicly as a T1 Reflex Class release target:

- local-first
- not a neural network claim
- exact mode should refuse unsupported answers
- imagination mode, when present, must be explicit and labeled
- benchmark numbers are scoped selector claims, not universal latency claims

The current public page is a release and verification surface. The binary is not
yet published as a runnable public artifact.

## Current Public Release

Latest public boundary release:
[`public-sdk-p11-20260629`](https://github.com/VRAXION/VRAXION/releases/tag/public-sdk-p11-20260629)

That release contains the public SDK/docs boundary and does not include private
engine source or a signed binary package.

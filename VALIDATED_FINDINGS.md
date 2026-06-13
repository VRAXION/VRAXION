# VRAXION Validated Findings

_Last updated: 2026-06-13_

This is the active evidence summary for the current repo state. Historical findings remain in git history, archived wiki state, release notes, and the Timeline Archive. This file is intentionally narrowed to the current sellable mainline.

## Current State

```text
branch = main
head = fffc5a438078592c7ca97fd9a840fa5a7948b353
current slice = E72 Rust curriculum resume preflight
```

## Current Mainline Chain

| Slice | Commit | Finding | Status |
|---|---|---|---|
| E69 | `8b00c352` | Persistent Pocket Library store with registry, tokens, artifacts, ledgers, guarded load, alias survival, safety blockers, safe promotion, and reload snapshot. | Current mainline |
| E70 | `9accc081` | Curriculum runner preflight on top of the Pocket Library base. | Current mainline |
| E71 | `c9dcad01` | Curriculum queue preflight on top of the runner layer. | Current mainline |
| E72 | `fffc5a43` | Curriculum resume preflight on top of the queue layer. | Current mainline |

## Current Validated Claim

> VRAXION has a Rust mainline for persistent Pocket Library governance and resumable curriculum execution preflights.

## Operational Finding

Long-running work must write progress continuously and support resume. The E72 resume preflight is the current proof surface for this operating rule.

## Hard Boundary

This file does not claim:

- hosted SaaS
- public production API readiness
- GPT-like/open-domain assistant readiness
- safety-aligned production deployment
- consciousness or sentience
- that old beta/grower/byte-pipeline results are the current sellable model

## Historical Evidence

Previous bounded-service, open-vocab assistant, beta release, grower, byte-pipeline, and C19/EP results are historical evidence. They can still be useful for research context, but they are not the current mainline unless promoted back into E72+ code and docs.

Primary archive surfaces:

- `archive/branches/2026-06-13/*`
- `archive/wiki/pre-consolidation-2026-06-13`
- [Timeline Archive wiki](https://github.com/VRAXION/VRAXION/wiki/Timeline-Archive)
- Git history of this repository

# VRAXION Current Status

_Last updated: 2026-06-13_

## Official Status

```text
Current source of truth: main
Current runtime slice: fffc5a438078592c7ca97fd9a840fa5a7948b353
Current runtime subject: E72 Rust curriculum resume preflight
Active branch surface: main only
Historical branch heads: archive/branches/2026-06-13/*
```

## Current Mainline

```text
E69 persistent Pocket Library store
-> E70 curriculum runner preflight
-> E71 curriculum queue preflight
-> E72 curriculum resume preflight
```

## What Is Current

- `vraxion-runtime/` is the active Rust runtime surface.
- The current Pocket Library layer handles registry, tokens, artifacts, ledgers, guarded load, reload snapshot, alias survival, safety blockers, and safe promotion.
- The current curriculum layer has runner, queue, and resume preflights.
- Long-running work must emit continuous partial progress and checkpoint data.

## What Is Historical

The old bounded-service, open-vocab assistant, beta release, byte-pipeline, and grower-era materials are historical evidence unless explicitly promoted into the E72+ mainline.

## Claim Boundary

Allowed current claim:

> VRAXION has a Rust mainline for persistent Pocket Library governance and resumable curriculum execution preflights.

Not claimed:

- hosted production service
- public API readiness
- GPT-like/open-domain assistant readiness
- safety-aligned production deployment
- consciousness or sentience

## Next Work

1. Keep public docs aligned to the E72+ mainline.
2. Remove or archive stale repo docs that describe older lines as current.
3. Cut a new release only after the E72+ surface is verified and intentionally packaged.

# VRAXION Current Status

_Last updated: 2026-06-13_

## Official Status

```text
Current source of truth: main
Current runtime slice: 51cd82a11d8f1d2b98ee3e49538c7c26afdb767b
Current runtime subject: E73 Rust final bake preflight
Base runtime slice: fffc5a438078592c7ca97fd9a840fa5a7948b353 E72 Rust curriculum resume preflight
Active branch surface: main only
Historical branch heads: archive/branches/2026-06-13/*
```

## Current Mainline

```text
E69 persistent Pocket Library store
-> E70 curriculum runner preflight
-> E71 curriculum queue preflight
-> E72 curriculum resume preflight
-> E73 unified final-bake preflight
```

## What Is Current

- `vraxion-runtime/` is the active Rust runtime surface.
- The current Pocket Library layer handles registry, tokens, artifacts, ledgers, guarded load, reload snapshot, alias survival, safety blockers, and safe promotion.
- The current curriculum layer has runner, queue, and resume preflights.
- The current final-bake layer validates the locked Rust mechanics from one consolidated binary.
- Long-running work must emit continuous partial progress and checkpoint data.

## What Is Historical

The old bounded-service, open-vocab assistant, beta release, byte-pipeline, and grower-era materials are historical evidence unless explicitly promoted into the E73+ mainline.

## Claim Boundary

Allowed current claim:

> VRAXION has a Rust mainline for persistent Pocket Library governance and resumable curriculum execution preflights.

Current E73 extension:

> VRAXION has a unified Rust final-bake preflight over the locked runtime mechanics.

Not claimed:

- hosted production service
- public API readiness
- GPT-like/open-domain assistant readiness
- safety-aligned production deployment
- consciousness or sentience

## Next Work

1. Keep public docs aligned to the E73+ mainline.
2. Remove or archive stale repo docs that describe older lines as current.
3. Cut a new release only after the E73+ surface is verified and intentionally packaged.

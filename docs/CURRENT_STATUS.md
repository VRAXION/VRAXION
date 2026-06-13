# VRAXION Current Status

_Last updated: 2026-06-13_

## Official Status

```text
Current source of truth: main
Current runtime slice: 0879a2c004cf6a002bd5639d9cb7a759709a41aa
Current runtime subject: E74 Rust final bake API extraction
Base runtime slice: 51cd82a11d8f1d2b98ee3e49538c7c26afdb767b E73 Rust final bake preflight
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
-> E74 reusable final-bake API extraction
```

## What Is Current

- `vraxion-runtime/` is the active Rust runtime surface.
- The current Pocket Library layer handles registry, tokens, artifacts, ledgers, guarded load, reload snapshot, alias survival, safety blockers, and safe promotion.
- The current curriculum layer has runner, queue, and resume preflights.
- The current final-bake layer validates the locked Rust mechanics from one consolidated binary and exposes the same gate through a reusable Rust module.
- Long-running work must emit continuous partial progress and checkpoint data.

## What Is Historical

The old bounded-service, open-vocab assistant, beta release, byte-pipeline, and grower-era materials are historical evidence unless explicitly promoted into the E74+ mainline.

## Claim Boundary

Allowed current claim:

> VRAXION has a Rust mainline for persistent Pocket Library governance and resumable curriculum execution preflights.

Current E74 extension:

> VRAXION exposes the final-bake gate through a reusable Rust library API while preserving the CLI preflight and continuous progress artifacts.

Not claimed:

- hosted production service
- public API readiness
- GPT-like/open-domain assistant readiness
- safety-aligned production deployment
- consciousness or sentience

## Next Work

1. Keep public docs aligned to the E74+ mainline.
2. Remove or archive stale repo docs that describe older lines as current.
3. Cut a new release only after the E74+ surface is verified and intentionally packaged.

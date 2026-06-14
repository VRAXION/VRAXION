# VRAXION Current Status

_Last updated: 2026-06-14_

## Official Status

```text
Current source of truth: main
Current GitHub release: v5.0.0-e78.0
Current runtime slice: 5f335cec3502d6c932e2f40c5c5a3a389eb44b7e
Current runtime subject: E78 canonical final_train campaign entrypoint
Base runtime slice: 0879a2c004cf6a002bd5639d9cb7a759709a41aa E74 Rust final bake API extraction
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
-> E75 final curriculum pocket-generation runner
-> E76 multi-lane final-training supervisor
-> E77 global Pocket Library merge supervisor
-> E78 canonical final_train campaign entrypoint
```

## What Is Current

- `vraxion-runtime/` is the active Rust runtime surface.
- The current Pocket Library layer handles registry, tokens, artifacts, ledgers, guarded load, reload snapshot, alias survival, safety blockers, and safe promotion.
- The current curriculum layer has runner, queue, and resume preflights.
- The current final-bake layer validates the locked Rust mechanics from one consolidated binary and exposes the same gate through a reusable Rust module.
- The current final-training layer runs deterministic curriculum queues through the Pocket Library with checkpoint/progress/partial/final artifact writeout and resume support.
- The current supervisor layer fans out final-training lanes, then merges lane-local candidates into one governed global Pocket Library with dedupe and guarded reload.
- The current canonical entrypoint is `final_train`.
- Long-running work must emit continuous partial progress and checkpoint data.

## What Is Historical

The old bounded-service, open-vocab assistant, beta release, byte-pipeline, and grower-era materials are historical evidence unless explicitly promoted into the E78+ mainline.

## Claim Boundary

Allowed current claim:

> VRAXION has a Rust mainline for governed Pocket Library state, resumable curriculum execution, multi-lane final-training supervision, global Pocket Library merge/dedupe governance, and one canonical `final_train` campaign entrypoint.

Current E78 extension:

> VRAXION has a deterministic Rust `final_train` command that runs the global Pocket Library supervisor over multi-lane E75 final-training lanes, writes progress/manifests/results, blocks redundant clones, and records zero bad commits or unsafe promotions in the E78 evidence run.

Not claimed:

- hosted production service
- public API readiness
- GPT-like/open-domain assistant readiness
- final dataset readiness
- trained model/weights readiness
- safety-aligned production deployment
- consciousness or sentience

## Next Work

1. Define the dataset/curriculum contract for the first real final-training run.
2. Turn pocket capability scoring from deterministic guard fixtures into dataset-backed promotion evidence.
3. Add the trained Pocket Library inference entrypoint once final-training artifacts exist.

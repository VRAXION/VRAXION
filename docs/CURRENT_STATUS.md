# VRAXION Current Status

_Last updated: 2026-06-14_

## Official Status

```text
Current source of truth: main
Current GitHub release: v5.0.0-e79.0
Current runtime slice: a908a838a1119540ed88bc91e10cfcb0bdae92a8
Current runtime subject: E79 training data/curriculum readiness gate
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
-> E79 training data/curriculum readiness gate
```

## What Is Current

- `vraxion-runtime/` is the active Rust runtime surface.
- The current Pocket Library layer handles registry, tokens, artifacts, ledgers, guarded load, reload snapshot, alias survival, safety blockers, and safe promotion.
- The current curriculum layer has runner, queue, and resume preflights.
- The current final-bake layer validates the locked Rust mechanics from one consolidated binary and exposes the same gate through a reusable Rust module.
- The current final-training layer runs deterministic curriculum queues through the Pocket Library with checkpoint/progress/partial/final artifact writeout and resume support.
- The current supervisor layer fans out final-training lanes, then merges lane-local candidates into one governed global Pocket Library with dedupe and guarded reload.
- The current training-data layer validates split/family/capability coverage, scoring policy, inference target coverage, curriculum digest, and candidate rotation before final-training supervisor work starts.
- The current canonical entrypoints are `training_data_readiness` for the standalone gate and `final_train` for the integrated campaign path.
- Long-running work must emit continuous partial progress and checkpoint data.

## What Is Historical

The old bounded-service, open-vocab assistant, beta release, byte-pipeline, and grower-era materials are historical evidence unless explicitly promoted into the E79+ mainline.

## Claim Boundary

Allowed current claim:

> VRAXION has a Rust mainline for governed Pocket Library state, resumable curriculum execution, multi-lane final-training supervision, global Pocket Library merge/dedupe governance, a training-data/curriculum readiness gate, and one canonical `final_train` campaign entrypoint.

Current E79 extension:

> VRAXION has a deterministic Rust `training_data_readiness` command and an integrated `final_train` gate that validates 24 curriculum lessons across train/validation/adversarial splits, 8 families, 8 capability signatures, 16 candidate pockets, scoring policy, and inference target coverage before any global supervisor work starts.

Not claimed:

- hosted production service
- public API readiness
- GPT-like/open-domain assistant readiness
- final production dataset completion
- trained model/weights readiness
- safety-aligned production deployment
- consciousness or sentience

## Next Work

1. Turn pocket capability scoring from deterministic guard fixtures into dataset-backed promotion evidence.
2. Add the trained Pocket Library inference entrypoint once final-training artifacts exist.
3. Run the first real final-training dataset through the E79 readiness gate and E78/E77 supervisor stack.

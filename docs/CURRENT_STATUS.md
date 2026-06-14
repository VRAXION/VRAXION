# VRAXION Current Status

_Last updated: 2026-06-14_

## Official Status

```text
Current source of truth: main
Current GitHub release: v5.0.0-e79.0
Current main head: 56a9cf0305c1bfddd0e9b763b5e0d80fc9ec3bca E85 CALC-SCRIBE mixed-stream inference integration
Latest released runtime slice: a908a838a1119540ed88bc91e10cfcb0bdae92a8 E79 training data/curriculum readiness gate
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
-> E80 dataset-backed pocket capability scoring evidence
-> E81 CALC-SCRIBE v002 multiseed visible-marker training
-> E82 CALC-SCRIBE v003 floor-division confirmation
-> E83 CALC-SCRIBE v003 LocalGolden promotion/reload
-> E84 CALC-SCRIBE transfer and negative-scope probe
-> E85 CALC-SCRIBE mixed-stream inference integration
```

## What Is Current

- `vraxion-runtime/` is the active Rust runtime surface.
- The current GitHub release is `v5.0.0-e79.0`; E80-E85 are post-release tracked evidence on `main`.
- The current Pocket Library layer handles registry, tokens, artifacts, ledgers, guarded load, reload snapshot, alias survival, safety blockers, and safe promotion.
- The current curriculum layer has runner, queue, and resume preflights.
- The current final-bake layer validates the locked Rust mechanics from one consolidated binary and exposes the same gate through a reusable Rust module.
- The current final-training layer runs deterministic curriculum queues through the Pocket Library with checkpoint/progress/partial/final artifact writeout and resume support.
- The current supervisor layer fans out final-training lanes, then merges lane-local candidates into one governed global Pocket Library with dedupe and guarded reload.
- The current training-data layer validates split/family/capability coverage, scoring policy, inference target coverage, curriculum digest, and candidate rotation before final-training supervisor work starts.
- The current canonical entrypoints are `training_data_readiness` for the standalone gate and `final_train` for the integrated campaign path.
- The current post-release evidence layer validates CALC-SCRIBE v003 as a governed LocalGolden scoped Pocket for visible calculation-trace marker validation.
- E84 confirms transfer across explicit visible calc-trace formats while no-calling word-problem text without visible trace framing.
- E85 confirms mixed-stream routing: visible calc traces can call CALC-SCRIBE, invalid traces are rejected, and natural text / final-answer-only / marker-stripped rows no-call.
- Long-running work must emit continuous partial progress and checkpoint data.

## What Is Historical

The old bounded-service, open-vocab assistant, beta release, byte-pipeline, and grower-era materials are historical evidence unless explicitly promoted into the E79+ mainline.

## Claim Boundary

Allowed current claim:

> VRAXION has a Rust mainline for governed Pocket Library state, resumable curriculum execution, multi-lane final-training supervision, global Pocket Library merge/dedupe governance, a training-data/curriculum readiness gate, one canonical `final_train` campaign entrypoint, and post-release evidence for a governed CALC-SCRIBE scoped Pocket that validates visible calculation traces inside a mixed input stream.

Current E85 extension:

> CALC-SCRIBE v003 is a governed LocalGolden scoped Pocket for visible calculation-trace validation. The E85 mixed-stream integration routes to it only when an explicit visible calc trace exists, rejects invalid visible traces, and no-calls natural text or word-problem text without visible trace framing.

Not claimed:

- hosted production service
- public API readiness
- GPT-like/open-domain assistant readiness
- GSM8K solving
- natural-language word-problem solving
- final production dataset completion
- trained model/weights readiness
- Core memory or True Golden promotion
- safety-aligned production deployment
- consciousness or sentience

## Next Work

1. Decide whether E80-E85 should become a new tagged release or stay as post-release main evidence.
2. Convert the E85 managed active-set path from probe evidence into the smallest appropriate runtime-facing surface, if promotion is warranted.
3. Keep expanding governed pocket inference only under explicit scope guards and negative controls.

# VRAXION Validated Findings

_Last updated: 2026-06-14_

This is the active evidence summary for the current repo state. Historical findings remain in git history, archived wiki state, release notes, and the Timeline Archive. This file is intentionally narrowed to the current sellable mainline.

## Current State

```text
branch = main
current_release = v5.0.0-e79.0
current_main_head = 56a9cf0305c1bfddd0e9b763b5e0d80fc9ec3bca
current_main_head_subject = Add E85 calc scribe mixed stream integration
latest_released_runtime_slice = a908a838a1119540ed88bc91e10cfcb0bdae92a8 E79 training data/curriculum readiness gate
base_runtime_slice = 0879a2c004cf6a002bd5639d9cb7a759709a41aa E74 Rust final bake API extraction
```

## Current Mainline Chain

| Slice | Commit | Finding | Status |
|---|---|---|---|
| E69 | `8b00c352` | Persistent Pocket Library store with registry, tokens, artifacts, ledgers, guarded load, alias survival, safety blockers, safe promotion, and reload snapshot. | Current mainline |
| E70 | `9accc081` | Curriculum runner preflight on top of the Pocket Library base. | Current mainline |
| E71 | `c9dcad01` | Curriculum queue preflight on top of the runner layer. | Current mainline |
| E72 | `fffc5a43` | Curriculum resume preflight on top of the queue layer. | Current mainline |
| E73 | `51cd82a1` | Unified Rust final-bake preflight over the locked mechanics. | Current mainline |
| E74 | `0879a2c0` | Final-bake logic extracted into a reusable Rust API while preserving the CLI preflight artifact contract. | Current mainline |
| E75 | `3f519732` | Final curriculum pocket-generation runner with preflight gating, checkpoint/progress artifacts, resume, and Pocket Library growth. | Current mainline |
| E76 | `3b44cfe0` | Multi-lane final-training supervisor with lane fanout, lane artifact preservation, aggregate progress, and zero bad/unsafe lane promotions. | Current mainline |
| E77 | `7e91aaaa` | Global Pocket Library merge supervisor with uid/digest/token dedupe, guarded reload, clone blocking, and global registry artifacts. | Current mainline |
| E78 | `5f335cec` | Canonical `final_train` campaign entrypoint over the global supervisor, multi-lane supervisor, and E75 lane runner stack. | Current mainline |
| E79 | `a908a838` | Training-data/curriculum readiness gate with split/family/capability coverage, scoring/inference contract checks, and `final_train` fail-fast blocking before global supervisor work. | Latest release |
| E80 | `6c4181cf` | Dataset-backed pocket capability scoring and promotion evidence; 8 seeds, 8 workers, 3 promoted, 0 bad promotions. | Post-release main evidence |
| E81 | `b4335206` | CALC-SCRIBE v002 multiseed visible-marker training; 16 seeds/workers, marker validation mean 0.999705, remaining gap isolated to floor division. | Post-release main evidence |
| E82 | `3914a64a` | CALC-SCRIBE v003 closes floor division with validation/action/floor/adversarial minimums at 1.000000. | Post-release main evidence |
| E83 | `4370bacc` | CALC-SCRIBE v003 LocalGolden promotion/reload confirmed with reload match 1.000000, bad promotions 0, and tamper/token/unsafe global-scope blockers. | Post-release main evidence |
| E84 | `0a06c153` | Transfer router confirms explicit visible calc-trace formats and negative scope: no visible calc marker means NO_CALL, not hidden answer inference. | Post-release main evidence |
| E85 | `56a9cf03` | Mixed-stream inference integration routes CALC-SCRIBE only for explicit visible calc traces, rejects invalid visible traces, and no-calls natural text / word-problem rows without trace framing. | Current main head |

## Current Validated Claim

> VRAXION has a Rust mainline for governed Pocket Library state, resumable curriculum execution, multi-lane final-training supervision, global Pocket Library merge/dedupe governance, a training-data/curriculum readiness gate, one canonical `final_train` campaign entrypoint, and post-release evidence for a governed CALC-SCRIBE scoped Pocket that validates visible calculation traces inside a mixed input stream.

Current E85 extension:

> CALC-SCRIBE v003 is a governed LocalGolden scoped Pocket for visible calculation-trace validation. The E85 mixed-stream integration routes to it only when an explicit visible calc trace exists, rejects invalid visible traces, and no-calls natural text or word-problem text without visible trace framing.

## Operational Finding

Long-running work must write progress continuously and support resume. The E72 resume preflight proves the resume path; the E73 final-bake preflight proves the locked chain can be validated from one consolidated Rust entrypoint with progress artifacts; E74 keeps that CLI contract and makes the final-bake gate reusable from Rust code; E75 runs the first deterministic final-training-shaped curriculum and Pocket generation loop with checkpoints and resume; E76 fans that lane work out in parallel; E77 merges lane-local candidates into one governed global Pocket Library; E78 makes that full path available through one canonical command; E79 blocks `final_train` before global supervisor work unless the dataset/curriculum contract is complete.

E80-E85 add scoped post-release pocket evidence. The strongest current finding is not open-ended reasoning; it is governed routing to a visible calculation-trace validator with explicit negative controls.

## Hard Boundary

This file does not claim:

- hosted SaaS
- public production API readiness
- GPT-like/open-domain assistant readiness
- GSM8K solving
- natural-language word-problem solving
- final production dataset completion
- trained model/weights readiness
- Core memory or True Golden promotion
- safety-aligned production deployment
- consciousness or sentience
- that old beta/grower/byte-pipeline results are the current sellable model

## Historical Evidence

Previous bounded-service, open-vocab assistant, beta release, grower, byte-pipeline, and C19/EP results are historical evidence. They can still be useful for research context, but they are not the current mainline unless promoted back into E79+ code and docs.

Primary archive surfaces:

- `archive/branches/2026-06-13/*`
- `archive/wiki/pre-consolidation-2026-06-13`
- [Timeline Archive wiki](https://github.com/VRAXION/VRAXION/wiki/Timeline-Archive)
- Git history of this repository

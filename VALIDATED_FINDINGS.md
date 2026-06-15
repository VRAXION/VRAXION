# VRAXION Validated Findings

_Last updated: 2026-06-15_

This is the active evidence summary for the current repo state. Historical findings remain in git history, archived wiki state, release notes, and the Timeline Archive.

## Current State

```text
branch = main
current_release = v6.1.7
current_evidence_anchor = f32a6f4b
current_evidence_subject = Finalize E127 cycle 40 checkpoint
latest_released_runtime_slice = a908a838a1119540ed88bc91e10cfcb0bdae92a8 E79 training data/curriculum readiness gate
```

## Current Mainline Chain

| Slice | Commit | Finding | Status |
|---|---|---|---|
| E69-E79 | `a908a838` | Rust Pocket Library, curriculum runner/queue/resume, final-bake API, final curriculum runner, multi-lane supervisor, global merge supervisor, canonical `final_train`, and training-data/curriculum readiness gate. | Released runtime foundation |
| E80-E85 | `56a9cf03` | CALC-SCRIBE visible calculation-trace evidence, LocalGolden reload, transfer/negative-scope probe, and mixed-stream no-call integration. | v6 evidence |
| E86-E89 | `a6935e61` | LocalGolden seeded curriculum, sparse active-set selector, survival gauntlet, and Operator naming/schema lock. | v6 evidence |
| E90-E106 | `b75c64cb` | Operator curriculum expansions for text evidence, temporal state, agency guards, output hygiene, active evidence requests, memory hygiene, routing, multi-skill execution, scheduling, grounded answer decisions, clarification repair, multi-turn continuity, compression, and task progress. | v6 evidence |
| E107 | `1fcdf954` | E90-E106 Operator library survival role and regression gauntlet. | v6 evidence |
| E108 | `0389c211` | External dataset transfer and negative-scope no-harm gauntlet. | v6 evidence |
| E109-E118 | tracked on `main` | Rank ladder, probation, no-harm, and CoreCandidate gauntlets. | v6 evidence |
| E119-E126 | tracked on `main` | FineWeb/text-understanding skill mining and Orange/Legendary probation. | v6 evidence |
| E127 | `f32a6f4b` | Overnight cyclic Orange/Legendary text-operator farm: 40 cycles, 382 scoped operators, 0 hard negatives, 0 false commits, 0 wrong-scope calls, 0 unsupported answers. | Current evidence anchor |

## Current Validated Claim

> VRAXION v6 has a Rust mainline for governed Pocket Library state, resumable curriculum execution, multi-lane final-training supervision, global Pocket Library merge/dedupe governance, a training-data/curriculum readiness gate, one canonical `final_train` campaign entrypoint, and governed Operator evidence through E127. E127 cycle 40 contains 382 scoped Orange/Legendary text operators with 0 tracked hard negatives, false commits, wrong-scope calls, or unsupported answers in the checkpointed evidence.

## Operational Finding

Long-running work must write progress continuously and support resume. The Rust runtime layers prove checkpoint/progress/writeout behavior through the E72-E79 chain. The E80-E127 evidence layer is governed research/operator evidence and must remain explicitly scoped until promoted into runtime-facing behavior.

## Hard Boundary

This file does not claim:

- hosted SaaS
- public production API readiness
- GPT-like/open-domain assistant readiness
- Gemma-level/freeform text generation
- GSM8K solving
- natural-language word-problem solving
- final production dataset completion
- trained model/weights readiness
- PermaCore or TrueGolden promotion
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

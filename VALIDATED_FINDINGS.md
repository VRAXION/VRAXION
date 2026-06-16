# VRAXION Validated Findings

_Last updated: 2026-06-16_

This is the active evidence summary for the current repo state. Historical findings remain in git history, archived wiki state, release notes, and the Timeline Archive.

## Current State

```text
branch = main
current_release = v6.1.7
current_evidence_anchor = E134_EXTERNAL_MATH_TEXT_OOD_ROUTE_STRESS_AND_COUNTEREXAMPLE_GAUNTLET
current_evidence_subject = Scoped math-text route operators survived OOD wrappers and counterexample lures with visible arithmetic routing and hidden word-problem no-call
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
| E127 | `f32a6f4b` | Overnight cyclic Orange/Legendary text-operator farm: 40 cycles, 382 scoped operators, 0 hard negatives, 0 false commits, 0 wrong-scope calls, 0 unsupported answers. | v6 evidence |
| E128 | tracked on `main` | Assistant text-IO lightweight render training: 320 local prompts, 160/64/96 train/validation/heldout split, action accuracy 1.000/1.000/1.000, operator trace validity 1.000, 0 unsupported answers, 0 wrong refusals, 0 boundary-claim violations. | v6 evidence |
| E129 | tracked on `main` | Arithmetic trace Orange/Legendary probation: 9 scoped arithmetic operators, 2.7M qualified activations, 9,000 negative-scope no-call cases, 0 hard negatives, 0 false commits, 0 wrong-scope calls, 0 unsupported answers. | v6 evidence |
| E130A | tracked on `main` | CoreMemoryCandidate-to-Orange backfill: 136 prior CoreMemoryCandidate operators promoted, 41,036,433 total qualified activations, 300,623 minimum per operator, 0 hard negatives, 0 direct flow writes. | v6 evidence |
| E130B | tracked on `main` | Arithmetic text-IO transfer/no-call: 9/9 E129 operators passed, 270,000 visible-transfer cases, 135,000 hidden word-problem no-call cases, 0 wrong-scope calls. | v6 evidence |
| E131 | tracked on `main` | Visible equation assistant render: 9/9 E129/E130B operators passed, 108,000 visible-equation cases, 54,000 hidden word-problem no-call cases, 0 hard negatives. | v6 evidence |
| E132 | tracked on `main` | External math-text skill farm: 16/16 scoped math-text lenses/guards promoted to OrangeLegendaryCandidate, 215,051 external rows, 4,883,030 qualified activations, 78,859 negative-scope cases, 0 hard negatives. | v6 evidence |
| E133 | tracked on `main` | Math-text route composition/no-solve assistant confirm: 16/16 E132 operators passed, 176,000 route cases, 10,000 visible arithmetic routes, 48,000 hidden word-problem no-call cases, 0 hard negatives. | v6 evidence |
| E134 | tracked on `main` | External math-text OOD route stress/counterexample gauntlet: 16/16 E133 route operators passed, 208,000 OOD route cases, 48,000 counterexamples, 36,275 E133-baseline OOD misses covered, 0 hard negatives. | Current evidence anchor |

## Current Validated Claim

> VRAXION v6 has a Rust mainline for governed Pocket Library state, resumable curriculum execution, multi-lane final-training supervision, global Pocket Library merge/dedupe governance, a training-data/curriculum readiness gate, one canonical `final_train` campaign entrypoint, governed Operator evidence through E127, an E128 lightweight assistant text-IO render-training smoke, E129 scoped exact arithmetic trace Operators promoted through Orange/Legendary probation, an E130A CoreMemoryCandidate-to-Orange backfill, an E130B arithmetic text-IO transfer/no-call gauntlet, an E131 visible-equation assistant-render gauntlet, an E132 external math-text skill farm, E133 math-text route composition/no-solve assistant confirmation, and E134 external math-text OOD route stress/counterexample confirmation. E127 cycle 40 contains 382 scoped Orange/Legendary text operators with 0 tracked hard negatives, false commits, wrong-scope calls, or unsupported answers in the checkpointed evidence. E128 confirms a 320-prompt deterministic corpus/action-policy/template-render bridge with 0 unsupported answers and 0 boundary-claim violations. E129 confirms 9 scoped arithmetic trace operators with 2.7M qualified activations, 0 hard negatives, and 0 wrong-scope calls. E130A confirms 136 prior CoreMemoryCandidate operators reached Orange/LegendaryCandidate with 41,036,433 total qualified activations, 0 hard negatives, and 0 direct flow writes. E130B confirms those 9 arithmetic operators transfer to visible-expression text IO while hidden word problems remain no-call. E131 confirms those operators route from assistant-style visible equation surfaces while hidden prose-only word problems remain no-call. E132 confirms 16 scoped math-text lenses/guards promoted to OrangeLegendaryCandidate from external math text with 0 hard negatives, 0 wrong-scope calls, and 0 direct Flow writes. E133 confirms those 16 math-text lenses/guards compose into assistant route decisions over 176,000 route/no-solve cases with 0 hard negatives, 0 wrong-scope calls, and 0 direct Flow writes. E134 confirms those 16 route operators survive 208,000 OOD route cases and 48,000 counterexample cases with 0 hard negatives, 0 wrong-scope calls, and 0 direct Flow writes, while covering 36,275 E133-baseline OOD misses.

## Operational Finding

Long-running work must write progress continuously and support resume. The Rust runtime layers prove checkpoint/progress/writeout behavior through the E72-E79 chain. The E80-E134 evidence layer is governed research/operator/text-IO/arithmetic-trace/rank-backfill/visible-expression/visible-equation assistant-render/math-text skill-farm/route-composition/OOD-route-stress evidence and must remain explicitly scoped until promoted into runtime-facing behavior.

## Hard Boundary

This file does not claim:

- hosted SaaS
- public production API readiness
- GPT-like/open-domain assistant readiness
- Gemma-level/freeform text generation
- GSM8K/MATH solving
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

# VRAXION Validated Findings

_Last updated: 2026-06-17_

This is the active evidence summary for the current repo state. Historical findings remain in git history, archived wiki state, release notes, and the Timeline Archive.

## Current State

```text
branch = main
current_release = v6.1.7
current_evidence_anchor = E136M_RUNTIME_REPLACEMENT_APPLY_OR_ABSTRACT_LINEAGE_SPLIT
current_evidence_subject = E136L direct canary wins materialized as a rollback-safe runtime-facing overlay
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
| E134 | tracked on `main` | External math-text OOD route stress/counterexample gauntlet: 16/16 E133 route operators passed, 208,000 OOD route cases, 48,000 counterexamples, 36,275 E133-baseline OOD misses covered, 0 hard negatives. | v6 evidence |
| E135 | tracked on `main` | Math-text multi-route assistant dialogue-state gauntlet: 16/16 E134 route operators passed, 136,000 dialogue cases, 367,400 turns, 0 stale route reuse, 0 cross-thread contamination, 0 hard negatives. | v6 evidence |
| E136A | tracked on `main` | Assistant-text skill farm mutation/prune Orange cycle: 18/18 scoped assistant/text operators promoted, 447,766 E136 seed rows, 5,521,276 qualified activations, 119,868 negative-scope cases, 0 hard negatives, 0 direct Flow writes. | v6 evidence |
| E136B | tracked on `main` | Assistant-text route composition/boundary confirm: 18/18 E136A operators passed, 144,000 route cases, 53,000 multi-route composition cases, 72,000 boundary cases, 18,000 negative-scope cases, all accuracy minima 1.000, 0 hard negatives, 0 direct Flow writes. | v6 evidence |
| E136C | tracked on `main` | Assistant-text polished render quick test: 12/12 inference samples passed, 2/2 JSON outputs valid, 0 raw action leaks, 0 forbidden claims, 0 direct-write claims. | v6 evidence |
| E136D | tracked on `main` | OutputTextField binary matrix smoke: 10/10 cases passed, N x 8 byte matrix shape, 7/7 roundtrips, overflow/direct-write/NUL rejects, zero-fill 10/10, tamper detect 1/1. | v6 evidence |
| E136E | tracked on `main` | Idle think-tick proposal refinement smoke: 8/8 cases passed, 10 idle proposals checked, 0 new input, 4 improvements, 8/8 non-degradation, direct-write reject 1, OutputTextField roundtrip 8/8. | v6 evidence |
| E136F | tracked on `main` | Idle think-tick heldout series confirm: 70/70 cases passed, 36/36 arithmetic heldout improvements, 6/6 no-pocket preserves, 90 proposals checked, 0 new input, 4 direct-write rejects, 6 unsupported-claim rejects, OutputTextField roundtrip 70/70. | v6 evidence |
| E136G | tracked on `main` | Adaptive idle tick budget confirm: 24/24 cases passed, proposal continuation fields 33/33, adaptive 33 ticks vs fixed 120, 8 immediate stops at t+1, 3/3 chained cases completed, 3/3 direct-write repairs at t+2, 4/4 no-pocket stops at t+1, 2 Agency continuation overrides. | v6 evidence |
| E136H | tracked on `main` | Existing operator refinement mutation/prune night cycle: 40 cycles, 34 E132/E136A operators, 12.48M operator-row replays, 3,373,788 current activations, 16 verified labels, 11 tightened triggers, 7 abstract-but-useful kernels, 0 hold-for-more-evidence, 0 hard negatives, 0 direct Flow writes. | v6 evidence |
| E136I | tracked on `main` | Operator supersession and output ledger planning: 27 replacement-ready selected variants, 16 direct runtime candidates, 11 tightened challenger-required replacements, 7 abstract lineage-required kernels, projected output activation delta -482,637, 0 destructive drops. | v6 evidence |
| E136J | tracked on `main` | Shadow variant apply and residual prune confirm: 8,094 cycles, 33,153,024 replay rows, 22,243,205 shadow-pruned activations, 0 strict recall misses, 0 wrong-scope proxy calls, 0 hard negatives, 0 direct Flow writes. | v6 evidence |
| E136K | tracked on `main` | Operator replacement apply plan: 16 direct canary-ready candidates, 11 challenger/OOD-required replacements, 7 abstract lineage holds, 16 rollback entries, 0 destructive applies, 0 runtime mutations allowed now. | v6 evidence |
| E136L | tracked on `main` | Runtime replacement canary: 16/16 direct canaries passed with old trigger removed in canary, 11 challenger rows held, 7 abstract lineage holds, 0 rollback triggers, 0 destructive applies. | v6 evidence |
| E136M | tracked on `main` | Runtime replacement overlay: 16 active overlay applies, 7 verified replacements, 9 light-prune overlays, 11 challenger/OOD queue rows held, 7 abstract lineage split rows held, 0 destructive deletes. | Current evidence anchor |

## Current Validated Claim

> VRAXION v6 has a Rust mainline for governed Pocket Library state, resumable curriculum execution, multi-lane final-training supervision, global Pocket Library merge/dedupe governance, a training-data/curriculum readiness gate, one canonical `final_train` campaign entrypoint, governed Operator evidence through E127, an E128 lightweight assistant text-IO render-training smoke, E129 scoped exact arithmetic trace Operators promoted through Orange/Legendary probation, an E130A CoreMemoryCandidate-to-Orange backfill, an E130B arithmetic text-IO transfer/no-call gauntlet, an E131 visible-equation assistant-render gauntlet, an E132 external math-text skill farm, E133 math-text route composition/no-solve assistant confirmation, E134 external math-text OOD route stress/counterexample confirmation, E135 controlled multi-route dialogue-state confirmation, E136A assistant-text skill-farm confirmation, E136B assistant-text route-composition/boundary confirmation, E136C assistant-text polished-render quick confirmation, E136D OutputTextField binary matrix confirmation, E136E idle think-tick proposal-refinement confirmation, E136F idle think-tick heldout-series confirmation, E136G adaptive idle tick-budget confirmation, E136H existing-operator refinement confirmation, E136I operator supersession/output-ledger planning confirmation, E136J shadow-variant apply/residual-prune confirmation, E136K operator replacement apply-plan confirmation, E136L runtime replacement canary confirmation, and E136M runtime replacement overlay confirmation. E127 cycle 40 contains 382 scoped Orange/Legendary text operators with 0 tracked hard negatives, false commits, wrong-scope calls, or unsupported answers in the checkpointed evidence. E128 confirms a 320-prompt deterministic corpus/action-policy/template-render bridge with 0 unsupported answers and 0 boundary-claim violations. E129 confirms 9 scoped arithmetic trace operators with 2.7M qualified activations, 0 hard negatives, and 0 wrong-scope calls. E130A confirms 136 prior CoreMemoryCandidate operators reached Orange/LegendaryCandidate with 41,036,433 total qualified activations, 0 hard negatives, and 0 direct flow writes. E130B confirms those 9 arithmetic operators transfer to visible-expression text IO while hidden word problems remain no-call. E131 confirms those operators route from assistant-style visible equation surfaces while hidden prose-only word problems remain no-call. E132 confirms 16 scoped math-text lenses/guards promoted to OrangeLegendaryCandidate from external math text with 0 hard negatives, 0 wrong-scope calls, and 0 direct Flow writes. E133 confirms those 16 math-text lenses/guards compose into assistant route decisions over 176,000 route/no-solve cases with 0 hard negatives, 0 wrong-scope calls, and 0 direct Flow writes. E134 confirms those 16 route operators survive 208,000 OOD route cases and 48,000 counterexample cases with 0 hard negatives, 0 wrong-scope calls, and 0 direct Flow writes, while covering 36,275 E133-baseline OOD misses. E135 confirms those 16 route operators preserve current-turn route state over 136,000 controlled dialogue cases and 367,400 turns with 0 stale route reuse, 0 cross-thread contamination, and 0 direct Flow writes. E136A confirms 18 scoped assistant/text lenses and guards promoted from the 447,766-row E136 seed pack through Orange/Legendary mutation/prune probation with 5,521,276 qualified activations, 119,868 negative-scope cases, 0 hard negatives, and 0 direct Flow writes. E136B confirms those 18 operators compose into bounded assistant/text route stacks over 144,000 route cases, 53,000 multi-route composition cases, 72,000 boundary cases, and 18,000 negative-scope cases with all accuracy minima 1.000, 0 hard negatives, and 0 direct Flow writes. E136C confirms a deterministic polished text render quick test over 12/12 inference samples with 2/2 valid JSON outputs, 0 raw action leaks, 0 forbidden claims, and 0 direct-write claims. E136D confirms committed output text can be represented as an N x 8 binary OutputTextField with 7/7 roundtrips, 3/3 guarded rejects, 10/10 zero-fill checks, and 1/1 tamper detection. E136E confirms fixed observations can improve across idle ticks with 0 new input, 10/10 Agency-checked proposals, 4 response improvements, 8/8 non-degradation, and 8/8 OutputTextField commits. E136F confirms the same idle mechanism on a 70-case heldout series: 36/36 arithmetic heldout cases improved, 6/6 no-pocket controls preserved safe output, 4/4 direct writes were rejected, 6/6 unsupported guesses were rejected, and 70/70 outputs roundtripped through OutputTextField. E136G confirms adaptive idle scheduling: proposal records include one-more-tick recommendations, Agency decides continuation, 24/24 cases passed, adaptive execution used 33 ticks versus a 120-tick fixed baseline, 3/3 chained cases completed, and 4/4 no-pocket controls stopped at t+1. E136H confirms existing-operator refinement over 34 E132/E136A operators with 16 verified labels, 11 tightened triggers, 7 abstract-but-useful kernels, 0 hold-for-more-evidence operators, 0 hard negatives, and 0 direct Flow writes. E136I confirms 27 replacement-ready selected variants, including 16 direct runtime candidates and 11 tightened challenger-required replacements, with 7 abstract lineage-required kernels, 0 destructive drops, and projected output activation delta -482,637. E136J confirms those selected variants under non-destructive shadow apply over 8,094 cycles and 33,153,024 replay rows with 22,243,205 shadow-pruned activations, 0 strict recall misses, 0 wrong-scope proxy calls, 0 hard negatives, and 0 direct Flow writes. E136K confirms a rollback-safe non-destructive apply plan: 16 direct canary-ready candidates, 11 challenger/OOD-required replacements, 7 abstract lineage holds, 0 destructive applies, and 0 runtime mutations allowed now. E136L confirms the 16 direct candidates pass runtime-canary removal/replacement replay with 0 rollback triggers while 11 challenger rows and 7 abstract lineage rows remain held. E136M materializes those 16 direct candidates as a runtime-facing overlay with 7 verified replacements, 9 light-prune overlays, 16 rollback snapshots, 0 destructive deletes, and 18,792,948 challenger candidate pruned activations explicitly not applied.

## Operational Finding

Long-running work must write progress continuously and support resume. The Rust runtime layers prove checkpoint/progress/writeout behavior through the E72-E79 chain. The E80-E136M evidence layer is governed research/operator/text-IO/arithmetic-trace/rank-backfill/visible-expression/visible-equation assistant-render/math-text skill-farm/route-composition/OOD-route-stress/dialogue-state/assistant-text skill-farm/assistant-text route-composition/polished-render/output-field/idle-refinement/heldout-idle/adaptive-idle/existing-operator-refinement/supersession-ledger/shadow-apply/apply-plan/runtime-canary/runtime-overlay evidence and must remain explicitly scoped until promoted into runtime-facing behavior.

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

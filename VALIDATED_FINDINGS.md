# VRAXION Validated Findings

_Last updated: 2026-06-19_

This is the active evidence summary for the current repo state. Historical findings remain in git history, archived wiki state, release notes, and the Timeline Archive.

## Current State

```text
branch = main
current_release = v6.1.7
current_evidence_anchor = E136S_ATOMIC_MULTIWRITE_DEFAULT_ROUTE_SWITCH_CANARY_GUARD
current_evidence_subject = guarded atomic multiwrite switch canary with snapshot + preview/apply match before default-route writes
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
| E136M | tracked on `main` | Runtime replacement overlay: 16 active overlay applies, 7 verified replacements, 9 light-prune overlays, 11 challenger/OOD queue rows held, 7 abstract lineage split rows held, 0 destructive deletes. | v6 evidence |
| E136N | tracked on `main` | Primary/secondary variant governance: 34 operators with one primary and one secondary each, 16 primary-active overlays, 16 rollback secondaries, 11 challenger secondaries, 7 lineage-hold secondaries, 0 retired variants. | v6 evidence |
| E136N2 | tracked on `main` | Agency Matrix arbitration smoke: 118 training examples converged in 2 epochs, 146 proposal-bundle cases, baseline accuracy 0.232877, Agency Matrix accuracy 1.000000, 10 Flow chunks, 0 unsafe commits, 0 challenger promotions. | v6 evidence |
| E136N3 | tracked on `main` | Parallel direct-write A/B smoke: 123 cases, direct-write accuracy 0.089431, Agency-gated accuracy 1.000000, 34 direct-write unsafe commits, 102 direct-write nondeterministic cases, 11 direct-write safe controls passed, 0 Agency-gated runtime direct writes. | v6 evidence |
| E136N4 | tracked on `main` | Agency-gated atomic multi-write confirm: 225 cases, accuracy 1.000000, 191 atomic writes, 0 partial writes, 0 runtime direct writes. | v6 evidence |
| E136O | tracked on `main` | Oracle-free challenger/OOD runtime replacement gauntlet: 7,146 full cases, full accuracy 1.000000, 3,667 atomic writes, 0 partial writes, 0 runtime direct writes. | v6 evidence |
| E136P | tracked on `main` | Runtime atomic multiwrite implementation preview: Rust atomic batch API and preview path, 10/10 cases, production apply disabled. | v6 evidence |
| E136Q | tracked on `main` | Runtime overlay canary atomic multiwrite confirm: 10/10 canary cases, default route unchanged, rollback snapshots 10, production apply disabled. | v6 evidence |
| E136R | tracked on `main` | Atomic multiwrite pre-apply decision gauntlet: 1,536/1,536 default-route regression cases and 2,048/2,048 canary stress cases succeeded, production apply disabled. | v6 evidence |
| E136S | tracked on `main` | Atomic multiwrite default-route switch canary guard: 1,536/1,536 default-route cases and 2,048/2,048 switch cases succeeded, 1,024 guarded applies, 1,024 guarded blocked no-apply cases, 0 false applies, 0 runtime direct writes. | Current evidence anchor |

## Current Validated Claim

> VRAXION v6 has a Rust mainline for governed Pocket Library state, resumable curriculum execution, multi-lane final-training supervision, global Pocket Library merge/dedupe governance, a training-data/curriculum readiness gate, one canonical `final_train` campaign entrypoint, governed Operator evidence through E136S. E136N2 confirmed a trained Agency Matrix arbitration smoke over the E136N surface with 118 training examples, 146 proposal-bundle cases, Agency Matrix accuracy 1.000000, 10 Flow chunks, 0 unsafe commits, and 0 challenger promotions. E136N3 confirmed parallel proposal fanout should keep an Agency commit barrier: the direct-write arm reached only 0.089431 accuracy with 34 unsafe commits and 102 nondeterministic cases, while the Agency-gated arm reached 1.000000 accuracy with 0 runtime direct writes. E136S confirms safe atomic multiwrite cases can pass snapshot + preview guard + preview/apply match before guarded default-route application, while rejected/deferred cases leave the default route unchanged. It still does not authorize unrestricted production apply.

## Operational Finding

Long-running work must write progress continuously and support resume. The Rust runtime layers prove checkpoint/progress/writeout behavior through the E72-E79 chain. The E80-E136S evidence layer is governed research/operator/text-IO/arithmetic-trace/rank-backfill/visible-expression/visible-equation assistant-render/math-text skill-farm/route-composition/OOD-route-stress/dialogue-state/assistant-text skill-farm/assistant-text route-composition/polished-render/output-field/idle-refinement/heldout-idle/adaptive-idle/existing-operator-refinement/supersession-ledger/shadow-apply/apply-plan/runtime-canary/runtime-overlay/variant-governance/Agency-Matrix-arbitration/parallel-fanout-direct-write-A-B/atomic-multiwrite-preview/overlay-canary/pre-apply/default-route-switch-canary evidence and must remain explicitly scoped until promoted into runtime-facing behavior.

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

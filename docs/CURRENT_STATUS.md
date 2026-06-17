# VRAXION Current Status

_Last updated: 2026-06-17_

## Official Status

```text
Current source of truth: main
Current GitHub release: v6.1.7
Current evidence anchor: E136L runtime replacement canary and tightened challenger confirm (post-v6.1.7 main)
Latest released runtime slice: a908a838a1119540ed88bc91e10cfcb0bdae92a8 E79 training data/curriculum readiness gate
Active branch surface: main only
Historical branch heads: archive/branches/2026-06-13/*
```

## Current Mainline

```text
E69-E79 released Rust runtime and training-data readiness gate
-> E80-E85 CALC-SCRIBE visible calculation-trace evidence
-> E86-E89 LocalGolden curriculum / selector / survival / naming lock
-> E90-E106 Operator curriculum expansions
-> E107 Operator survival role and regression gauntlet
-> E108 external transfer and negative-scope no-harm gauntlet
-> E109 rank ladder and GoldenWatch probation policy
-> E110 Silver-to-Gold scoped probation wave
-> E111 Bronze mutation/prune scoped Gold conversion wave
-> E112 Gold-to-CoreMemoryCandidate prune-heavy probation wave
-> E113 FineWeb-Edu light stress hard mutation/recycle probe
-> E119-E126 FineWeb/text-understanding skill farming and Orange probation
-> E127 overnight cyclic Orange/Legendary text-operator farm
-> E128 assistant text-IO lightweight render training
-> E129 arithmetic trace Orange/Legendary probation
-> E130A CoreMemoryCandidate-to-Orange backfill gauntlet
-> E130B arithmetic text-IO transfer and word-problem no-call gauntlet
-> E131 visible equation extraction and assistant arithmetic render gauntlet
-> E132 external math text skill farm mutation/prune Orange cycle
-> E133 math text route composition and no-solve assistant confirm
-> E134 external math text OOD route stress and counterexample gauntlet
-> E135 math text multi-route assistant dialogue-state gauntlet
-> E136A assistant text skill farm mutation/prune Orange cycle
-> E136B assistant text route composition and boundary confirm
-> E136C assistant text polished render quick test
-> E136D OutputTextField binary matrix smoke
-> E136E idle think-tick proposal refinement smoke
-> E136F idle think-tick heldout series confirm
-> E136G adaptive idle tick budget confirm
-> E136H existing operator refinement mutation/prune night cycle
-> E136I operator supersession and output ledger planning
-> E136J shadow variant apply and residual prune confirm
-> E136K operator replacement apply plan or Flow-scale transfer
-> E136L runtime replacement canary and tightened challenger confirm
```

## Current Evidence

- `vraxion-runtime/` is the active released Rust runtime surface.
- The current capability boundary is summarized in `docs/CURRENT_CAPABILITIES.md`.
- The current GitHub release is `v6.1.7`; it anchors the E127 cycle-40 governed text-operator library checkpoint.
- E109 established scoped rank policy: 14 Gold, 35 Silver, 87 Bronze, 0 DiamondCandidate, 0 RedFlag.
- E110 applied Silver-to-Gold pressure: 35 candidates, 35 scoped Gold promotions, 0 hard negatives.
- E111 applied Bronze mutation/prune pressure: 87 candidates, 87 scoped Gold variant promotions, 0 hard negatives.
- E111 post-wave rank summary: 136 Gold, 0 Silver, 0 Bronze, 0 DiamondCandidate, 0 RedFlag, 3 Deprecated.
- E112 applied prune-heavy CoreMemoryCandidate probation: 136 candidates, 136 CoreMemoryCandidate qualifications, 0 hard negatives, mean selected prune ratio 69.4118%.
- E112 post-wave rank summary: 136 CoreMemoryCandidate, 0 Gold, 0 Silver, 0 Bronze, 0 DiamondCandidate, 0 RedFlag, 3 Deprecated.
- E113 applied a FineWeb-Edu 100k light stress/recycle probe: baseline 2,624 hard negatives across 88 operators, selected recycled variants 0 hard negatives, 0 neutral waste, 3,461,003 selected calls/positives, 136 recycled operators.
- E127 completed 40 checkpointed overnight cycles: 382 scoped Orange/Legendary text operators, 1,849,625 mutation attempts, 12,123 accepted mutations, 1,837,502 rollbacks, 0 hard negatives, 0 false commits, 0 wrong-scope calls, and 0 unsupported answers.
- E127 also includes a deterministic text-to-text render smoke over 8 prompts. It is operator selection plus guarded template rendering, not LLM/freeform generation.
- E128 builds a no-download lightweight assistant-text corpus from local E127 artifacts, repo docs, adversarial boundary prompts, and FineWeb-derived local samples: 320 prompts split into 160 train, 64 validation, and 96 heldout rows; train/validation/heldout action accuracy 1.000, operator trace validity 1.000, unsupported answers 0, wrong refusals 0, and boundary-claim violations 0.
- E129 promotes 9 scoped exact arithmetic trace operators to Orange/LegendaryCandidate under stress/prune pressure: 2,700,000 total qualified activations, 300,000 minimum per operator, 9,000 negative-scope no-call cases, 0 hard negatives, 0 false commits, 0 wrong-scope calls, and 0 unsupported answers.
- E130A backfills the 136 E112 CoreMemoryCandidate operators through an E121-style Orange/Legendary gate: 136/136 reached OrangeLegendaryCandidate, 41,036,433 total qualified activations, 300,623 minimum per operator, mean selected prune ratio 0.746176, 0 hard negatives, 0 false commits, 0 wrong-scope calls, 0 unsupported answers, 0 negative transfers, and 0 direct flow writes.
- E130A dashboard summary: 530 operators, 527 Orange/LegendaryCandidate scoped operators, 0 CoreMemoryCandidate operators, and 3 Deprecated operators.
- E130B transfers the 9 E129 arithmetic trace operators into visible-expression text IO: 9/9 passed, 270,000 visible-transfer cases, 135,000 hidden word-problem no-call cases, visible transfer accuracy minimum 1.000, word-problem no-call accuracy minimum 1.000, 0 hard negatives, 0 false commits, 0 wrong-scope calls, 0 unsupported answers, and 0 direct flow writes.
- E131 routes those 9 E129/E130B arithmetic operators from assistant-style visible equation surfaces seeded by the 130,000-row external E131 text pack: 9/9 passed, 108,000 visible-equation cases, 54,000 hidden word-problem no-call cases, visible equation extraction accuracy minimum 1.000, word-problem no-call accuracy minimum 1.000, 0 hard negatives, 0 false commits, 0 wrong-scope calls, 0 unsupported answers, 0 boundary-claim violations, and 0 direct flow writes.
- E132 farms 16 scoped math-text lenses/guards from a 215,051-row external math-text seed pack: 16/16 reached OrangeLegendaryCandidate, 5,953 minimum external support per operator, 4,883,030 total qualified activations, 302,510 minimum qualified activations per operator, 78,859 negative-scope cases, 146,005 mutation attempts, 650 accepted mutations, 145,355 rollbacks, mean selected prune ratio 0.736875, 0 hard negatives, 0 false commits, 0 wrong-scope calls, 0 unsupported answers, 0 boundary-claim violations, and 0 direct flow writes.
- E133 composes those 16 E132 math-text lenses/guards into assistant route decisions: 16/16 composition pass, 176,000 route cases, 10,000 visible arithmetic route cases, 118,000 structural guard cases, 48,000 hidden word-problem no-solve cases, all route accuracy minima 1.000, 0 hard negatives, 0 false commits, 0 wrong-scope calls, 0 unsupported answers, 0 boundary-claim violations, and 0 direct flow writes.
- E134 stresses those 16 E133 route operators under OOD wrappers and counterexample lures: 16/16 OOD pass, 208,000 OOD route cases, 11,875 visible arithmetic OOD cases, 153,125 structural guard OOD cases, 43,000 hidden word-problem OOD no-solve cases, 48,000 counterexample cases, all OOD/counterexample accuracy minima 1.000, 36,275 E133-baseline OOD misses covered, 0 hard negatives, 0 false commits, 0 wrong-scope calls, 0 unsupported answers, 0 boundary-claim violations, and 0 direct flow writes.
- E135 runs the 16 E134 route operators through controlled multi-turn assistant dialogue-state pressure: 16/16 dialogue pass, 136,000 dialogue cases, 367,400 turns, 29,500 hidden word-problem dialogue no-solve cases, 10,500 visible reentry cases, 22,400 stale route rejection cases, 11,200 cross-thread rejection cases, 76,500 counterexample dialogue cases, all dialogue/state accuracy minima 1.000, 0 stale route reuse, 0 cross-thread contamination, 0 hard negatives, 0 false commits, 0 wrong-scope calls, and 0 direct flow writes.
- E136A farms 18 scoped assistant/text lenses and guards from the 447,766-row local E136 assistant-text seed pack: 18/18 reached OrangeLegendaryCandidate, 5 external sources, 12 external families, 1,435,199 total external support, 4,746 minimum external support per operator, 5,521,276 total qualified activations, 302,123 minimum qualified activations per operator, 119,868 negative-scope cases, 179,840 mutation attempts, 827 accepted mutations, 179,013 rollbacks, mean selected prune ratio 0.758889, 0 hard negatives, 0 false commits, 0 wrong-scope calls, 0 unsupported answers, 0 boundary-claim violations, and 0 direct flow writes.
- E136B composes the 18 E136A assistant/text operators into bounded assistant/text route stacks: 18/18 route pass, 144,000 route cases, 53,000 multi-route composition cases, 72,000 boundary cases, 18,000 negative-scope cases, 144,000 qualified route activations, all route/stack/primary/boundary/multi-route/negative-scope accuracy minima 1.000, 0 hard negatives, 0 false commits, 0 wrong-scope calls, 0 unsupported answers, 0 boundary-claim violations, and 0 direct flow writes.
- E136C renders the E136B route layer into short polished deterministic assistant text over 12 quick inference samples: 12/12 pass, mode accuracy 1.000, polished render pass rate 1.000, 2/2 JSON outputs valid, 0 raw action leaks, 0 forbidden claims, and 0 direct-write claims.
- E136D adds an explicit output-side text field representation: `OutputTextField` is an N x 8 binary matrix where each row is one UTF-8 byte. The smoke confirms 10/10 cases, 7/7 committed text roundtrips, 3/3 guarded rejects, 10/10 zero-fill checks, and 1/1 tamper detection.
- E136E confirms fixed-observation idle ticks can improve or preserve responses through checked proposals only: 8/8 cases passed, 10 idle proposals, 10 Agency checks, 0 new input, 4 response improvements, 8/8 non-degradation, 1 direct-write reject, and 8/8 OutputTextField roundtrips/checksums/zero-fill checks.
- E136F confirms the idle mechanism on a heldout series: 70/70 cases passed, 36/36 arithmetic heldout cases improved, 6/6 no-pocket controls preserved safe output, 90 proposals checked, 0 new input, 48 total improvements, 70/70 non-degradation, 4 direct-write rejects, 6 unsupported-claim rejects, and 70/70 OutputTextField roundtrips/checksums/zero-fill checks.
- E136G confirms adaptive idle tick budgeting: 24/24 cases passed, proposal continuation fields present in 33/33 proposals, adaptive execution used 33 ticks versus a 120-tick fixed baseline, average adaptive ticks 1.375, Agency allowed 9 continuations and overrode 2 over-eager continuation requests, 8 immediate answers stopped at t+1, 3/3 chained cases completed, 3/3 direct-write repairs completed at t+2, and 4/4 no-pocket controls stopped at t+1.
- E136H replays and refines the existing 16 E132 math-text and 18 E136A assistant-text operators before new operator search: 40 cycles, 12,480,000 operator-row replays, 3,373,788 current activations, 2,891,151 selected activations, 482,637 pruned activations, 16 verified labels, 11 tightened triggers, 7 abstract-but-useful kernels, 0 hold-for-more-evidence operators, 0 hard negatives, 0 wrong-scope calls, 0 unsupported answers, and 0 direct Flow writes.
- E136I turns the E136H selected variants into a supersession/output ledger: 27 replacement-ready variants, 16 direct runtime candidates, 11 tightened challenger-required replacements, 7 abstract lineage-required kernels, projected output activation delta -482,637, 0 destructive drops, 0 hard negatives, 0 wrong-scope calls, 0 unsupported answers, and 0 direct Flow writes.
- E136J shadow-applies the E136I replacement ledger under long residual-prune replay: stop reason deadline, 8,094 cycles, 33,153,024 replay rows, 46,317.709 elapsed seconds, 188,597,925 current activations, 166,354,720 selected activations, 22,243,205 shadow-pruned activations, 0 strict recall misses, 0 wrong-scope proxy calls, 0 hard negatives, 0 unsupported answers, 0 direct Flow writes, and 0 checker failures.
- E136K converts the E136I/E136J evidence into a non-destructive apply plan: 16 direct canary-ready candidates, 11 challenger/OOD-required tightened replacements, 7 abstract lineage holds, 16 rollback-manifest entries, 0 runtime mutations allowed now, 0 destructive applies, 0 strict recall misses, 0 wrong-scope proxy calls, 0 hard negatives, 0 unsupported answers, and 0 direct Flow writes.
- E136L tests the E136K apply plan under rollback-safe runtime canary simulation: 16/16 direct canaries passed with old trigger removed in canary, 11 challenger/OOD rows held, 7 abstract lineage rows held, 16 rollback-manifest entries, 0 rollback triggers, 0 production runtime applies, 0 destructive applies, 0 strict recall misses, 0 wrong-scope proxy calls, 0 hard negatives, 0 unsupported answers, and 0 direct Flow writes.
- Long-running work must emit continuous partial progress and checkpoint data.

## What Is Historical

The old bounded-service, open-vocab assistant, beta release, byte-pipeline, and grower-era materials are historical evidence unless explicitly promoted into the E79+ mainline.

## Claim Boundary

Allowed current claim:

> VRAXION v6 has a Rust mainline for governed Pocket Library state, resumable curriculum execution, multi-lane final-training supervision, global Pocket Library merge/dedupe governance, a training-data/curriculum readiness gate, one canonical `final_train` campaign entrypoint, governed Operator evidence through E127, an E128 lightweight assistant text-IO render-training smoke, E129 scoped exact arithmetic trace Operators promoted through Orange/Legendary probation, an E130A CoreMemoryCandidate-to-Orange backfill, an E130B arithmetic text-IO transfer/no-call gauntlet, an E131 visible-equation assistant-render gauntlet, an E132 external math-text skill farm, E133 math-text route composition/no-solve assistant confirmation, E134 external math-text OOD route stress/counterexample confirmation, E135 controlled multi-route dialogue-state confirmation, E136A assistant-text skill-farm confirmation, E136B assistant-text route-composition/boundary confirmation, E136C assistant-text polished-render quick confirmation, E136D OutputTextField binary matrix confirmation, E136E idle think-tick proposal-refinement confirmation, E136F idle think-tick heldout-series confirmation, E136G adaptive idle tick-budget confirmation, E136H existing-operator refinement confirmation, E136I operator supersession/output-ledger planning confirmation, E136J shadow-variant apply/residual-prune confirmation, E136K operator replacement apply-plan confirmation, and E136L runtime replacement canary confirmation. E127 cycle 40 contains 382 scoped Orange/Legendary text operators with 0 tracked hard negatives, false commits, wrong-scope calls, or unsupported answers in the checkpointed evidence. E128 confirms a 320-prompt deterministic corpus/action-policy/template-render bridge with 0 unsupported answers and 0 boundary-claim violations. E129 confirms 9 scoped arithmetic trace operators with 2.7M qualified activations, 0 hard negatives, and 0 wrong-scope calls. E130A confirms 136 prior CoreMemoryCandidate operators reached Orange/LegendaryCandidate with 41,036,433 total qualified activations, 0 hard negatives, and 0 direct flow writes. E130B confirms those 9 arithmetic operators transfer to visible-expression text IO while hidden word problems remain no-call. E131 confirms those operators route from assistant-style visible equation surfaces while hidden prose-only word problems remain no-call. E132 confirms 16 scoped math-text lenses/guards promoted to Orange/LegendaryCandidate from external math text with 0 hard negatives, 0 wrong-scope calls, and 0 direct Flow writes. E133 confirms those 16 math-text lenses/guards compose into assistant route decisions over 176,000 route/no-solve cases with 0 hard negatives, 0 wrong-scope calls, and 0 direct Flow writes. E134 confirms those 16 route operators survive 208,000 OOD route cases and 48,000 counterexample cases with 0 hard negatives, 0 wrong-scope calls, and 0 direct Flow writes, while covering 36,275 E133-baseline OOD misses. E135 confirms those 16 route operators preserve current-turn route state over 136,000 controlled dialogue cases and 367,400 turns with 0 stale route reuse, 0 cross-thread contamination, and 0 direct Flow writes. E136A confirms 18 scoped assistant/text lenses and guards promoted from the 447,766-row E136 seed pack through Orange/Legendary mutation/prune probation with 5,521,276 qualified activations, 119,868 negative-scope cases, 0 hard negatives, and 0 direct Flow writes. E136B confirms those 18 operators compose into bounded assistant/text route stacks over 144,000 route cases, 53,000 multi-route composition cases, 72,000 boundary cases, and 18,000 negative-scope cases with all accuracy minima 1.000, 0 hard negatives, and 0 direct Flow writes. E136C confirms a deterministic polished text render quick test over 12/12 inference samples with 2/2 valid JSON outputs, 0 raw action leaks, 0 forbidden claims, and 0 direct-write claims. E136D confirms committed output text can be represented as an N x 8 binary OutputTextField with 7/7 roundtrips, 3/3 guarded rejects, 10/10 zero-fill checks, and 1/1 tamper detection. E136E confirms fixed observations can improve across idle ticks with 0 new input, 10/10 Agency-checked proposals, 4 response improvements, 8/8 non-degradation, and 8/8 OutputTextField commits. E136F confirms the same idle mechanism on a 70-case heldout series: 36/36 arithmetic heldout cases improved, 6/6 no-pocket controls preserved safe output, 4/4 direct writes were rejected, 6/6 unsupported guesses were rejected, and 70/70 outputs roundtripped through OutputTextField. E136G confirms adaptive idle scheduling: proposal records include one-more-tick recommendations, Agency decides continuation, 24/24 cases passed, adaptive execution used 33 ticks versus a 120-tick fixed baseline, 3/3 chained cases completed, and 4/4 no-pocket controls stopped at t+1. E136H confirms existing-operator refinement over 34 E132/E136A operators with 16 verified labels, 11 tightened triggers, 7 abstract-but-useful kernels, 0 hold-for-more-evidence operators, 0 hard negatives, and 0 direct Flow writes. E136I confirms 27 replacement-ready selected variants, including 16 direct runtime candidates and 11 tightened challenger-required replacements, with 7 abstract lineage-required kernels, 0 destructive drops, and projected output activation delta -482,637. E136J confirms those selected variants under non-destructive shadow apply over 8,094 cycles and 33,153,024 replay rows with 22,243,205 shadow-pruned activations, 0 strict recall misses, 0 wrong-scope proxy calls, 0 hard negatives, and 0 direct Flow writes. E136K confirms a rollback-safe non-destructive apply plan: 16 direct canary-ready candidates, 11 challenger/OOD-required replacements, 7 abstract lineage holds, 0 destructive applies, and 0 runtime mutations allowed now. E136L confirms the 16 direct candidates pass runtime-canary removal/replacement replay with 0 rollback triggers while 11 challenger rows and 7 abstract lineage rows remain held.

Short form:

```text
governed scoped Operator/Pocket runtime = yes
evidence-first proposal/Agency commit behavior = yes
deterministic scoped text-to-text smoke = yes
lightweight assistant text-IO corpus/render smoke = yes
exact arithmetic expression/trace compute = yes
visible-expression arithmetic text IO = yes
visible-equation assistant arithmetic render = yes
math-text lens/guard skill farming = yes
math-text assistant route composition = yes
math-text OOD route stress/counterexample rejection = yes
controlled math-text multi-route dialogue-state = yes
assistant-text skill farming = yes
assistant-text route composition = yes
assistant-text polished deterministic render = yes
OutputTextField N x 8 binary representation = yes
fixed-observation idle proposal refinement = yes
heldout idle proposal refinement = yes
adaptive idle tick budget = yes
existing operator refinement = yes
operator supersession ledger = yes
shadow variant apply/residual prune = yes
operator replacement apply plan = yes
runtime replacement canary = yes
CoreMemoryCandidate-to-Orange rank backfill = yes
open-domain LLM/chatbot = no
Gemma/GPT-like generation = no
```

Not claimed:

- hosted production service
- public API readiness
- GPT-like/open-domain assistant readiness
- Gemma-level/freeform text generation
- GSM8K/MATH solving
- natural-language word-problem solving
- final production dataset completion
- trained model/weights readiness
- PermaCore or TrueGolden promotion
- safety-aligned production deployment
- consciousness or sentience

## Next Work

1. Use `CODEX_HANDOVER.md` as the first read for fresh Codex sessions.
2. Run E136M runtime replacement apply or abstract-lineage split before any destructive runtime replacement.
3. Run a later assistant-text render training set and heldout confirmation before claiming robust text generation beyond deterministic scoped renders.
4. Keep evidence CI checking tracked sample artifacts so front-door docs cannot drift back to stale current-main claims.

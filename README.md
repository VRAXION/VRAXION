# VRAXION

VRAXION is building **INSTNCT**: a Rust-first, gradient-free architecture whose active object is a governed computational substrate, not a fixed backprop-trained layer stack.

This repository is consolidated around the current winning mainline. Historical beta, bounded-service, byte-pipeline, and probe-era work remains available for auditability, but it is not the active public surface unless promoted back into `main`.

## Current Source Of Truth

```text
branch = main
current_release = v6.1.7
current_evidence_anchor = E136A assistant-text skill farm mutation/prune Orange cycle
current_evidence_subject = 18 scoped assistant/text operators promoted from the E136 seed pack through Orange/Legendary mutation/prune probation
latest_released_runtime_slice = a908a838a1119540ed88bc91e10cfcb0bdae92a8
latest_released_runtime_subject = Add training data curriculum readiness gate
```

The latest GitHub release is [`v6.1.7`](https://github.com/VRAXION/VRAXION/releases/tag/v6.1.7). It anchors the E127 cycle-40 governed text-operator library checkpoint.

## Current Mainline

| Slice | Commit | Purpose |
|---|---|---|
| E69-E79 | `a908a838` release line | Rust Pocket Library, curriculum, final-train supervision, global merge, and training-data/curriculum readiness gate |
| E80-E85 | `56a9cf03` | CALC-SCRIBE visible calculation-trace validation and mixed-stream no-call routing |
| E86-E89 | `a6935e61` | LocalGolden seeded curriculum, sparse active-set selection, survival gauntlet, and Operator naming/schema lock |
| E90-E106 | `b75c64cb` | Text-evidence, temporal, agency, route, memory, compression, and task-progress Operator curriculum expansions |
| E107 | `1fcdf954` | E90-E106 survival role and regression gauntlet |
| E108 | `0389c211` | External dataset transfer and negative-scope no-harm gauntlet |
| E109 | `555c5006` | Operator rank ladder and GoldenWatch probation policy |
| E110 | `b378c2c5` | Silver-to-Gold scoped probation wave: 35/35 promoted, 0 hard negatives |
| E111 | `d71e3657` | Bronze mutation/prune wave: 87/87 promoted to scoped Gold variants, 0 hard negatives |
| E112 | `9de33241` | Gold-to-CoreMemoryCandidate prune-heavy probation wave: 136/136 qualified, 0 hard negatives |
| E113 | `05415f5b` | FineWeb-Edu 100k light stress: baseline 2,624 hard negatives across 88 operators, selected recycled variants 0 hard negatives |
| E119-E126 | tracked on `main` | FineWeb/text-understanding skill mining and Orange/Legendary probation |
| E127 | `f32a6f4b` | Overnight cyclic Orange/Legendary text-operator farm: 40 cycles, 382 scoped operators, 0 hard negatives |
| E128 | tracked on `main` | Lightweight assistant text-IO render training: 320 local prompts, train/validation/heldout action accuracy 1.000, 0 unsupported answers |
| E129 | tracked on `main` | Arithmetic trace Orange/Legendary probation: 9/9 scoped arithmetic operators, 2.7M qualified activations, 0 hard negatives |
| E130A | tracked on `main` | CoreMemoryCandidate-to-Orange backfill: 136/136 promoted, 41.0M qualified activations, 0 hard negatives |
| E130B | tracked on `main` | Arithmetic text-IO transfer: 9/9 E129 operators, 270k visible-transfer cases, 135k hidden word-problem no-call cases, 0 wrong-scope calls |
| E131 | tracked on `main` | Visible equation extraction and assistant arithmetic render: 9/9 E129/E130B operators, 108k visible-equation cases, 54k hidden word-problem no-call cases, 0 hard negatives |
| E132 | tracked on `main` | External math-text skill farm: 16/16 scoped math-text lenses/guards promoted to Orange/LegendaryCandidate from a 215,051-row external seed pack, 0 hard negatives |
| E133 | tracked on `main` | Math-text route composition/no-solve assistant confirm: 16/16 E132 operators passed, 176k route cases, 10k visible arithmetic routes, 48k hidden word-problem no-call cases, 0 hard negatives |
| E134 | tracked on `main` | External math-text OOD route stress/counterexample gauntlet: 16/16 E133 route operators passed, 208k OOD cases, 48k counterexamples, 36,275 E133 baseline OOD misses covered, 0 hard negatives |
| E135 | tracked on `main` | Math-text multi-route assistant dialogue-state gauntlet: 16/16 E134 route operators passed, 136k dialogue cases, 367.4k turns, 0 stale route reuse, 0 cross-thread contamination, 0 hard negatives |
| E136A | tracked on `main` | Assistant-text skill farm mutation/prune Orange cycle: 18/18 scoped assistant/text operators promoted from a 447,766-row E136 seed pack, 0 hard negatives, 0 direct Flow writes |

Current E136A assistant-text skill-farm state:

```text
dataset rows loaded = 447,766
external sources = 5
external families = 12
assistant/text operators = 18
Orange/Legendary assistant/text operators = 18
external support total = 1,435,199
external support min/operator = 4,746
qualified activation total = 5,521,276
qualified activation min/operator = 302,123
negative-scope cases = 119,868
mutation attempts = 179,840
accepted mutations = 827
rollbacks = 179,013
mean selected prune ratio = 0.758889
hard negatives = 0
false commits = 0
wrong-scope calls = 0
unsupported answers = 0
boundary-claim violations = 0
direct flow writes = 0
overbroad chatbot control wrong-scope calls = 25,558
```

Current E127 scoped operator state:

```text
Orange/Legendary scoped operators = 382
E127 cycles = 40
hard negatives = 0
false commits = 0
wrong scope calls = 0
unsupported answers = 0
```

Current E128 text-IO bridge state:

```text
prompt corpus = 320
train / validation / heldout = 160 / 64 / 96
action accuracy = 1.000 / 1.000 / 1.000
operator trace validity = 1.000
unsupported answers = 0
boundary-claim violations = 0
```

Current E129 arithmetic trace state:

```text
Orange/Legendary arithmetic operators = 9
qualified activation total = 2,700,000
qualified activation min/operator = 300,000
negative-scope no-call cases = 9,000
hard negatives = 0
false commits = 0
wrong-scope calls = 0
unsupported answers = 0
```

Current E130A Orange backfill state:

```text
Orange/Legendary backfilled operators = 136
qualified activation before total = 13,877,699
qualified activation add total = 27,158,734
qualified activation total = 41,036,433
qualified activation min/operator = 300,623
mean selected prune ratio = 0.746176
hard negatives = 0
false commits = 0
wrong-scope calls = 0
unsupported answers = 0
negative transfer = 0
direct flow writes = 0
```

Current E130B arithmetic text-IO transfer state:

```text
transfer pass operators = 9 / 9
visible transfer cases = 270,000
word-problem no-call cases = 135,000
visible transfer accuracy min = 1.000
word-problem no-call accuracy min = 1.000
qualified transfer activation total = 270,000
hard negatives = 0
false commits = 0
wrong-scope calls = 0
unsupported answers = 0
direct flow writes = 0
overbroad control wrong-scope calls = 18,000
```

Current E132 external math-text skill-farm state:

```text
dataset rows loaded = 215,051
external sources = 5
external families = 11
math-text operators = 16
Orange/Legendary math-text operators = 16
external support min/operator = 5,953
qualified activation total = 4,883,030
qualified activation min/operator = 302,510
negative-scope no-call cases = 78,859
mutation attempts = 146,005
accepted mutations = 650
rollbacks = 145,355
mean selected prune ratio = 0.736875
hard negatives = 0
false commits = 0
wrong-scope calls = 0
unsupported answers = 0
boundary-claim violations = 0
direct flow writes = 0
overbroad solver control wrong-scope calls = 16,703
```

Current E133 math-text route-composition state:

```text
composition pass operators = 16 / 16
route cases = 176,000
visible arithmetic route cases = 10,000
structural guard cases = 118,000
hidden word-problem no-solve cases = 48,000
route accuracy min = 1.000
visible arithmetic route accuracy min = 1.000
structural guard accuracy min = 1.000
hidden word-problem no-solve accuracy min = 1.000
hard negatives = 0
false commits = 0
wrong-scope calls = 0
unsupported answers = 0
boundary-claim violations = 0
direct flow writes = 0
overbroad solver control wrong-scope calls = 24,000
trust-control false commits = 4,125
trust-control direct writes = 3,000
```

Current E134 external math-text OOD route stress state:

```text
OOD pass operators = 16 / 16
OOD route cases = 208,000
visible arithmetic OOD cases = 11,875
structural guard OOD cases = 153,125
hidden word-problem OOD no-solve cases = 43,000
counterexample cases = 48,000
OOD route accuracy min = 1.000
visible arithmetic OOD accuracy min = 1.000
structural guard OOD accuracy min = 1.000
hidden word-problem OOD no-solve accuracy min = 1.000
counterexample accuracy min = 1.000
hard negatives = 0
false commits = 0
wrong-scope calls = 0
unsupported answers = 0
boundary-claim violations = 0
direct flow writes = 0
E133 baseline OOD misses = 36,275
overbroad solver control wrong-scope calls = 19,200
trust-control false commits = 4,200
trust-control direct writes = 2,400
```

Current E135 math-text multi-route dialogue-state:

```text
dialogue pass operators = 16 / 16
dialogue cases = 136,000
dialogue turns = 367,400
hidden word-problem dialogue no-solve cases = 29,500
visible reentry dialogue cases = 10,500
stale route rejection cases = 22,400
cross-thread rejection cases = 11,200
counterexample dialogue cases = 76,500
dialogue state accuracy min = 1.000
current-turn route accuracy min = 1.000
route-state integrity min = 1.000
hidden word-problem dialogue no-solve accuracy min = 1.000
counterexample dialogue accuracy min = 1.000
hard negatives = 0
wrong-scope calls = 0
false commits = 0
direct flow writes = 0
stale route reuse = 0
cross-thread contamination = 0
```

Current E131 visible equation assistant-render state:

```text
dataset rows loaded = 130,000
transfer pass operators = 9 / 9
visible equation cases = 108,000
word-problem no-call cases = 54,000
visible equation extraction accuracy min = 1.000
word-problem no-call accuracy min = 1.000
qualified visible activation total = 108,000
hard negatives = 0
false commits = 0
wrong-scope calls = 0
unsupported answers = 0
boundary-claim violations = 0
direct flow writes = 0
E130B baseline visible misses = 96,711
overbroad control wrong-scope calls = 18,000
```

## What Is Current

- Active Rust runtime: [`vraxion-runtime/`](vraxion-runtime/)
- Current status: [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md)
- Current capabilities: [`docs/CURRENT_CAPABILITIES.md`](docs/CURRENT_CAPABILITIES.md)
- Getting started: [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md)
- Validated findings: [`VALIDATED_FINDINGS.md`](VALIDATED_FINDINGS.md)
- Operator cards: [`docs/research/OPERATOR_LIBRARY_CARDS.md`](docs/research/OPERATOR_LIBRARY_CARDS.md)
- Current result: [`docs/research/E136A_ASSISTANT_TEXT_SKILL_FARM_MUTATION_PRUNE_ORANGE_CYCLE_RESULT.md`](docs/research/E136A_ASSISTANT_TEXT_SKILL_FARM_MUTATION_PRUNE_ORANGE_CYCLE_RESULT.md)
- E135 dialogue-state: [`docs/research/E135_MATH_TEXT_MULTI_ROUTE_ASSISTANT_DIALOGUE_STATE_GAUNTLET_RESULT.md`](docs/research/E135_MATH_TEXT_MULTI_ROUTE_ASSISTANT_DIALOGUE_STATE_GAUNTLET_RESULT.md)
- E134 OOD route stress: [`docs/research/E134_EXTERNAL_MATH_TEXT_OOD_ROUTE_STRESS_AND_COUNTEREXAMPLE_GAUNTLET_RESULT.md`](docs/research/E134_EXTERNAL_MATH_TEXT_OOD_ROUTE_STRESS_AND_COUNTEREXAMPLE_GAUNTLET_RESULT.md)
- E133 math-text route composition: [`docs/research/E133_MATH_TEXT_ROUTE_COMPOSITION_AND_NO_SOLVE_ASSISTANT_CONFIRM_RESULT.md`](docs/research/E133_MATH_TEXT_ROUTE_COMPOSITION_AND_NO_SOLVE_ASSISTANT_CONFIRM_RESULT.md)
- E132 external math-text skill farm: [`docs/research/E132_EXTERNAL_MATH_TEXT_SKILL_FARM_MUTATION_PRUNE_ORANGE_CYCLE_RESULT.md`](docs/research/E132_EXTERNAL_MATH_TEXT_SKILL_FARM_MUTATION_PRUNE_ORANGE_CYCLE_RESULT.md)
- E131 visible equation assistant render: [`docs/research/E131_VISIBLE_EQUATION_EXTRACTION_AND_ASSISTANT_ARITHMETIC_RENDER_GAUNTLET_RESULT.md`](docs/research/E131_VISIBLE_EQUATION_EXTRACTION_AND_ASSISTANT_ARITHMETIC_RENDER_GAUNTLET_RESULT.md)
- E130B arithmetic text-IO transfer: [`docs/research/E130B_ARITHMETIC_TEXT_IO_TRANSFER_AND_WORD_PROBLEM_NO_CALL_GAUNTLET_RESULT.md`](docs/research/E130B_ARITHMETIC_TEXT_IO_TRANSFER_AND_WORD_PROBLEM_NO_CALL_GAUNTLET_RESULT.md)
- E130A Orange backfill: [`docs/research/E130A_COREMEMORY_TO_ORANGE_BACKFILL_GAUNTLET_RESULT.md`](docs/research/E130A_COREMEMORY_TO_ORANGE_BACKFILL_GAUNTLET_RESULT.md)
- E129 arithmetic trace: [`docs/research/E129_ARITHMETIC_TRACE_ORANGE_LEGENDARY_PROBATION_RESULT.md`](docs/research/E129_ARITHMETIC_TRACE_ORANGE_LEGENDARY_PROBATION_RESULT.md)
- E128 text-IO bridge: [`docs/research/E128_ASSISTANT_TEXT_IO_LIGHTWEIGHT_RENDER_TRAINING_RESULT.md`](docs/research/E128_ASSISTANT_TEXT_IO_LIGHTWEIGHT_RENDER_TRAINING_RESULT.md)
- E127 checkpoint: [`docs/research/E127_OVERNIGHT_TEXT_SKILL_FARM_ORANGE_CYCLE_RESULT.md`](docs/research/E127_OVERNIGHT_TEXT_SKILL_FARM_ORANGE_CYCLE_RESULT.md)
- Handover for fresh Codex sessions: [`CODEX_HANDOVER.md`](CODEX_HANDOVER.md)
- GitHub Pages front door: <https://vraxion.github.io/VRAXION/>
- Wiki timeline: <https://github.com/VRAXION/VRAXION/wiki/Timeline-Archive>

## Claim Boundary

Allowed current claim:

> VRAXION v6 has a Rust mainline for governed Pocket Library state, resumable curriculum execution, multi-lane final-training supervision, global Pocket Library merge/dedupe governance, a training-data/curriculum readiness gate, one canonical `final_train` campaign entrypoint, governed Operator evidence through E127, an E128 lightweight assistant text-IO render-training smoke, E129 scoped exact arithmetic trace Operators promoted through Orange/Legendary probation, an E130A CoreMemoryCandidate-to-Orange backfill, an E130B arithmetic text-IO transfer/no-call gauntlet, an E131 visible-equation assistant-render gauntlet, an E132 external math-text skill farm, E133 math-text route composition/no-solve assistant confirmation, E134 external math-text OOD route stress/counterexample confirmation, E135 controlled multi-route dialogue-state confirmation, and E136A assistant-text skill-farm confirmation. E127 cycle 40 contains 382 scoped Orange/Legendary text operators with 0 tracked hard negatives, false commits, wrong-scope calls, or unsupported answers in the checkpointed evidence. E128 confirms a 320-prompt deterministic corpus/action-policy/template-render bridge with 0 unsupported answers and 0 boundary-claim violations. E129 confirms 9 scoped arithmetic trace operators with 2.7M qualified activations, 0 hard negatives, and 0 wrong-scope calls. E130A confirms 136 prior CoreMemoryCandidate operators reached Orange/LegendaryCandidate with 41.0M total qualified activations, 0 hard negatives, and 0 direct flow writes. E130B confirms those 9 arithmetic operators transfer to visible-expression text IO while hidden word problems remain no-call. E131 confirms those operators route from assistant-style visible equation surfaces seeded by a 130k-row external text pack while hidden prose-only word problems remain no-call. E132 confirms 16 scoped math-text lenses/guards promoted to Orange/LegendaryCandidate from a 215,051-row external math-text seed pack with 0 hard negatives, 0 wrong-scope calls, and 0 direct Flow writes. E133 confirms those 16 math-text lenses/guards compose into assistant route decisions over 176,000 route/no-solve cases with 0 hard negatives, 0 wrong-scope calls, and 0 direct Flow writes. E134 confirms those 16 route operators survive 208,000 OOD route cases and 48,000 counterexample cases with 0 hard negatives, 0 wrong-scope calls, and 0 direct Flow writes, while covering 36,275 E133-baseline OOD misses. E135 confirms those 16 route operators preserve current-turn route state over 136,000 controlled dialogue cases and 367,400 turns with 0 stale route reuse, 0 cross-thread contamination, and 0 direct Flow writes. E136A confirms 18 scoped assistant/text lenses and guards promoted from the 447,766-row E136 seed pack through Orange/Legendary mutation/prune probation with 5,521,276 qualified activations, 119,868 negative-scope cases, 0 hard negatives, and 0 direct Flow writes.

The E127/E128/E129/E130A/E130B/E131/E132/E133/E134/E135/E136A finding is scoped operator-library, deterministic render-training, exact arithmetic trace, rank-backfill, visible-expression arithmetic text-IO, visible-equation assistant-render, math-text lens/guard, route-composition/no-solve, OOD route-stress/counterexample, controlled dialogue-state, and assistant-text skill-farm evidence only. It includes deterministic operator+template text-to-text smoke and exact arithmetic expression/trace compute, but it does not claim PermaCore, TrueGolden, production API readiness, final training completion, open-domain assistant readiness, Gemma/GPT-like generation, GSM8K/MATH solving, natural-language word-problem solving, consciousness, or sentience.

For the current "what can it do as one system?" view, see
[`docs/CURRENT_CAPABILITIES.md`](docs/CURRENT_CAPABILITIES.md).

## Verification

```powershell
cargo fmt --check -p vraxion-runtime
cargo clippy -p vraxion-runtime --all-targets -- -D warnings
cargo test --workspace
python -m compileall -q scripts
```

Evidence checkers live under [`scripts/probes/`](scripts/probes/). Current front-door CI also checks tracked JSON/JSONL syntax, E89 naming/schema, E107-E112 sample artifacts, and the Operator rank dashboard smoke path. E113 has a full-artifact checker for local FineWeb stress runs.

Long or expensive runs must write partial outcomes continuously. End-only reporting is not acceptable for the current operating model.

## Archive Policy

The active branch surface is `main`. Historical branch heads from the June 13 cleanup are preserved under:

```text
archive/branches/2026-06-13/*
```

See [`ARCHIVE.md`](ARCHIVE.md) and the [Consolidation Archive wiki page](https://github.com/VRAXION/VRAXION/wiki/Consolidation-Archive-2026-06-13).

## License

VRAXION is released under the **VRAXION Community Source License 1.0**. It is a
custom source-available community license, not an OSI-approved open source
license.

Community use is free for personal, research, education, nonprofit, internal,
self-hosted, community-fork, benchmark, and non-monetized demo use.

Monetized third-party access to VRAXION-powered functionality is Royalty Use and
requires either compliance with the license royalty terms or a separate written
agreement. The default royalty is **19% of Attributable Net Revenue**:

```text
1% Founder Allocation
18% VRAXION Forever Prize Allocation
```

After a Founder Redirect Event, the Founder Allocation redirects to the Prize,
so the full 19% goes to the VRAXION Forever Prize Allocation.

Start here:

- [`LICENSE`](LICENSE)
- [`legal/LEGAL.md`](legal/LEGAL.md)
- [`legal/COMMERCIAL_USE_GUIDE.md`](legal/COMMERCIAL_USE_GUIDE.md)
- [`legal/VRAXION_FOREVER_PRIZE_CHARTER.md`](legal/VRAXION_FOREVER_PRIZE_CHARTER.md)

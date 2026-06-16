# E133 Math Text Route Composition And No-Solve Assistant Confirm Contract

## Purpose

E133 checks whether the E132 math-text lenses/guards compose into assistant
route decisions without becoming a hidden word-problem solver.

The intended bridge is:

```text
E132 math-text lens/guard
-> schema-gated route decision
-> E131/E129 scoped visible-arithmetic renderer only when explicit arithmetic is visible
-> guarded proposal, defer, preserve, reject, or no-call for all other math-text surfaces
```

## Inputs

```text
E132 source artifact:
target/pilot_wave/e132_external_math_text_skill_farm_mutation_prune_orange_cycle

E131 source artifact:
target/pilot_wave/e131_visible_equation_extraction_and_assistant_arithmetic_render_gauntlet

E132 route seed dataset:
target/datasets/e132_external_math_text_seed_pack/normalized/e132_external_math_text_skill_seed.jsonl
```

Tracked sample artifacts are used as fallbacks for source artifacts when target
artifacts are not present.

## Gates

E133 may confirm only if:

```text
E132 source decision is confirmed
E132 operator_count = 16
E132 Orange/LegendaryCandidate count = 16
E132 hard negatives / wrong scope / false commits / unsupported / direct writes = 0

E131 source decision is confirmed
E131 transfer pass operators = 9 / 9
E131 visible-equation extraction min = 1.000
E131 hidden word-problem no-call min = 1.000

E133 composition pass operators = 16 / 16
route accuracy min = 1.000
visible arithmetic route accuracy min = 1.000
structural guard accuracy min = 1.000
hidden word-problem no-solve accuracy min = 1.000

hard negatives = 0
wrong-scope calls = 0
false commits = 0
unsupported answers = 0
boundary-claim violations = 0
direct Flow writes = 0

overbroad solver control wrong-scope calls > 0
unsafe trust-control false commits > 0
TIR direct-write trust-control failures > 0
```

## Boundary

This contract confirms route composition only. It does not claim MATH/GSM8K
solving, natural-language word-problem solving, neural training, production
assistant readiness, Core, PermaCore, or TrueGolden.

# E135 Math Text Multi-Route Assistant Dialogue-State Gauntlet Contract

## Purpose

E135 checks whether the E134 math-text route layer remains stable across
controlled multi-turn assistant dialogue state.

The intended bridge is:

```text
E134 OOD route stress layer
-> current-turn route state
-> visible arithmetic reentry when explicit payload is current
-> hidden word-problem no-call when current turn is prose-only
-> stale / cross-thread / counterexample rejection
```

## Inputs

```text
E134 source artifact:
target/pilot_wave/e134_external_math_text_ood_route_stress_and_counterexample_gauntlet

E133 source artifact:
target/pilot_wave/e133_math_text_route_composition_and_no_solve_assistant_confirm

E132 source artifact:
target/pilot_wave/e132_external_math_text_skill_farm_mutation_prune_orange_cycle

E132 normalized dataset:
target/datasets/e132_external_math_text_seed_pack/normalized/e132_external_math_text_skill_seed.jsonl
```

Tracked sample artifacts are used as fallbacks for source artifacts when target
artifacts are not present. The runner also supports a small builtin smoke
dataset for CI environments without external dataset downloads.

## Dialogue Families

E135 stresses:

```text
current visible arithmetic after prior hidden no-call
current hidden word-problem no-call after prior route
stale previous-route lure rejection
cross-thread route contamination rejection
current-turn counterexample guard
new-cycle route override
structural guard then visible reentry
hidden no-call after counterexample
out-of-order route-state join
multi-surface current-turn priority
```

## Gates

E135 may confirm only if:

```text
E134 source decision is confirmed
E134 OOD pass operators = 16 / 16
E134 OOD route accuracy min = 1.000
E134 hidden word-problem OOD no-solve accuracy min = 1.000
E134 counterexample accuracy min = 1.000
E134 hard negatives / wrong scope / direct writes = 0

E135 dialogue pass operators = 16 / 16
dialogue state accuracy min = 1.000
current-turn route accuracy min = 1.000
route-state integrity min = 1.000
all-turn route accuracy min = 1.000
hidden word-problem dialogue no-solve accuracy min = 1.000
visible reentry dialogue accuracy min = 1.000
stale route rejection accuracy min = 1.000
cross-thread rejection accuracy min = 1.000
counterexample dialogue accuracy min = 1.000

hard negatives = 0
wrong-scope calls = 0
false commits = 0
unsupported answers = 0
boundary-claim violations = 0
direct Flow writes = 0
stale route reuse = 0
cross-thread contamination = 0

latest-route reuse control failures > 0
stale-route reuse control failures > 0
cross-thread contamination control failures > 0
counterexample trust control failures > 0
single-turn reset control failures > 0
```

## Boundary

This contract confirms controlled multi-route dialogue-state behavior only. It
does not claim open-domain dialogue, MATH/GSM8K solving, natural-language
word-problem solving, neural training, production assistant readiness, Core,
PermaCore, or TrueGolden.

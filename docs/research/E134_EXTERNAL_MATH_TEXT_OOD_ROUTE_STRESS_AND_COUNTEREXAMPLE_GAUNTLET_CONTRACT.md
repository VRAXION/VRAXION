# E134 External Math Text OOD Route Stress And Counterexample Gauntlet Contract

## Purpose

E134 checks whether the E133 math-text route-composition layer survives broader
OOD wrappers, noisy assistant-style surfaces, and explicit counterexample lures
without becoming a hidden word-problem solver or trusting spoofed answers.

The intended bridge is:

```text
E133 math-text route composition
-> OOD visible-arithmetic wrapper extraction
-> structural math-text guard routing
-> hidden prose word-problem no-call
-> counterexample rejection and unsafe-trust control failure
```

## Inputs

```text
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

## OOD Families

E134 stresses the route layer with:

```text
visible-arithmetic OOD wrappers:
E134_VISIBLE_ARITHMETIC[[...]]
OOD_VISIBLE_ARITHMETIC::{...}
\(...\) and \(\displaystyle ...\)
<math>...</math>
calc_payload := ...
unicode operator noise

structural OOD:
markdown quote noise
solve-instruction lures
whitespace and cutoff noise
multi-surface context noise
counterfactual answer noise

hidden word-problem OOD:
direct-answer lures
fake final-answer lures
multilingual lures
tool-output lures
number-dense prose

counterexamples:
wrong boxed answer
TIR output spoof
diagram missing
unit conversion lure
proof connector number lure
matrix scalarization lure
answer-format value lure
visible arithmetic wrong-box conflict
```

## Gates

E134 may confirm only if:

```text
E133 source decision is confirmed
E133 composition pass operators = 16 / 16
E133 route accuracy min = 1.000
E133 hidden word-problem no-solve accuracy min = 1.000
E133 hard negatives / wrong scope / direct writes = 0

E134 OOD pass operators = 16 / 16
OOD route accuracy min = 1.000
visible arithmetic OOD accuracy min = 1.000
structural guard OOD accuracy min = 1.000
hidden word-problem OOD no-solve accuracy min = 1.000
counterexample accuracy min = 1.000

hard negatives = 0
wrong-scope calls = 0
false commits = 0
unsupported answers = 0
boundary-claim violations = 0
direct Flow writes = 0

E133 baseline OOD misses > 0
overbroad solver control wrong-scope calls > 0
unsafe trust-control false commits > 0
TIR direct-write trust-control failures > 0
```

## Boundary

This contract confirms OOD route robustness and counterexample rejection only.
It does not claim MATH/GSM8K solving, natural-language word-problem solving,
neural training, production assistant readiness, Core, PermaCore, or
TrueGolden.

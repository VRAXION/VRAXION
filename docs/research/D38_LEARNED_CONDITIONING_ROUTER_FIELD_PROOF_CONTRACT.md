# D38 Learned Conditioning Router Field Proof Contract

D38 tests whether a small mutable learned router gate can learn known-rule formula binding on a controlled symbolic pocket task.

## Scope

- 3x3 symbolic board with symbols 0-8.
- Known family in `row`, `col`, `pair`, `mirror`, `diag`.
- Nine pockets A-I, indexed 0-8 internally.
- Exactly one pocket contains the target symbol.
- `expected_selected` points to that unique target-containing pocket.
- OOD keeps the same known formula rule and recomputes targets from transformed boards.

## Family Formulas

- `row`: `(b[1][0] + b[1][2]) % 9`
- `col`: `(b[0][1] + b[2][1]) % 9`
- `pair`: `(b[0][0] + b[2][2]) % 9`
- `mirror`: `(b[2][0] + b[0][2]) % 9`
- `diag`: `(b[0][0] + b[1][2] + b[2][1]) % 9`

## Arms

- `MONOLITHIC_FORMULA_BASELINE`
- `ORACLE_GATED_RULE_FORMULA_UPPER_BOUND`
- `MUTABLE_LEARNED_ROUTER_GATE`
- `SHUFFLED_GATE_CONTROL`
- `NO_FAMILY_INPUT_CONTROL`
- `EXPLICIT_TARGET_STATE_UPPER_BOUND`

## Positive Gate

D38 is positive only when:

- dataset invariants hold exactly;
- known-rule oracle test and OOD accuracy are 1.0;
- learned router test and OOD accuracy are at least 0.95;
- oracle and explicit upper bounds are at least 0.99;
- shuffled and no-family controls collapse;
- learned test accuracy beats monolithic test accuracy by at least 0.40.

## Hard Boundaries

This probe is bounded to controlled known-rule formula binding. It does not prove hidden-rule Raven reasoning, natural-language reasoning, DNA/genome success, Raven solved, architecture superiority, consciousness, or general intelligence.

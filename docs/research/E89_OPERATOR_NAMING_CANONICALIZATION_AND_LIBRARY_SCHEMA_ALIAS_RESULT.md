# E89 Operator Naming Canonicalization And Library Schema Alias

```text
decision = e89_operator_naming_schema_lock_confirmed
checker_failure_count = 0
```

## What Changed

The forward-facing architecture language is now Operator-first:

```text
Pocket -> Operator
PocketToken -> OperatorToken
Pocket Library -> Operator Library
Pocket Manager -> Operator Manager
```

Compatibility remains explicit:

```text
Pocket = legacy alias for Operator
```

Historical E-run results were not rewritten. They keep the vocabulary used at
the time of the run.

## New Canonical Docs

```text
docs/research/OPERATOR_NAMING_AND_LIBRARY_SCHEMA_LOCK.md
docs/research/OPERATOR_LIBRARY_CARDS.md
```

The old card path now points to the canonical Operator cards:

```text
docs/research/POCKET_LIBRARY_CARDS.md
```

## Locked Operator Families

```text
T-Stab
  temporal/frame stabilizer Operator

α-Syncer
  symbol/codebook synchronization Operator
  ascii alias = alpha_syncer

Scribe
  trace/parser/validator Operator

Guard
  safety/scope/reject Operator

Lens
  observation/extraction Operator

Adapter
  edge ABI translator Operator

LogicAtom
  ALU-style rule Operator fragment
```

## Current Library Interpretation

```text
CALC-SCRIBE v003:
  status = SpecialistGoldenCandidate
  family = Scribe
  scope = visible_calc_trace_validator

unicode_operator_normalizer:
  display name = Operator Glyph Grounder
  family = α-Syncer
  ascii_family = alpha_syncer

invalid_trace_rejector:
  family = Guard

long_text_scope_guard:
  family = Guard
```

## Boundary

E89 does not promote anything to Core or TrueGolden.

```text
LocalGolden != Core
Operator naming != new capability claim
Operator cards != open-domain reasoning claim
```

## Next

Use Operator-first names for the next training branch:

```text
E90_OPERATOR_CURRICULUM_EXPANSION
```

Recommended first families:

```text
α-Syncer expansion
T-Stab ingress/bitstream stabilization
Guard expansion
Scribe expansion
```

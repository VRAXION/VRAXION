# E89 Operator Naming Canonicalization And Library Schema Alias Contract

## Purpose

Lock the post-E88 naming layer before the next training branch.

E89 is not a capability benchmark. It is a canonicalization pass:

```text
old generic term = Pocket
new generic term = Operator
```

Historical E-run result files may keep their original wording. Forward-facing
architecture, library cards, and schema docs must use Operator-first language.

## Required Changes

```text
docs/research/OPERATOR_NAMING_AND_LIBRARY_SCHEMA_LOCK.md
docs/research/OPERATOR_LIBRARY_CARDS.md
docs/research/POCKET_LIBRARY_CARDS.md
scripts/probes/run_e89_operator_naming_canonicalization_check.py
docs/research/E89_OPERATOR_NAMING_CANONICALIZATION_AND_LIBRARY_SCHEMA_ALIAS_RESULT.md
```

## Canonical Terms

```text
Operator
OperatorToken
Operator Library
Operator Manager
Operator Registry
```

Allowed compatibility aliases:

```text
Pocket = legacy alias for Operator
PocketToken = legacy alias for OperatorToken
Pocket Library = legacy alias for Operator Library
Pocket Manager = legacy alias for Operator Manager
```

## Family Names

```text
T-Stab
α-Syncer / alpha_syncer
Scribe
Guard
Lens
Adapter
LogicAtom
```

Code and filenames must use ASCII aliases where needed:

```text
alpha_syncer
operator_library
operator_token
```

## Checker Requirements

The checker must fail if:

```text
canonical naming lock doc is missing
Operator Library Cards are missing
legacy Pocket card file does not point to the Operator cards
OperatorToken / Registry / Manager terms are missing
Pocket compatibility alias is missing
α-Syncer and alpha_syncer aliases are missing
direct stable Flow/Ground write boundary is missing
LocalGolden is promoted to Core/TrueGolden by naming alone
```

## Decision Labels

```text
e89_operator_naming_schema_lock_confirmed
e89_operator_naming_schema_lock_failed
```

## Boundary

E89 does not change historical metrics, model behavior, training data, or
artifact scores. It only locks the forward naming/schema layer.

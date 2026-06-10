# E16B Text-Flow Operator Composition Discovery Confirm Contract

## Purpose

`E16B_TEXT_FLOW_OPERATOR_COMPOSITION_DISCOVERY_CONFIRM` tests whether controlled
text-flow transform behavior can be recovered after direct macro operators are
removed from the primary runtime. The primary receives randomized support/query
token streams plus low-level evidence flags, searches bounded primitive chains,
prunes a library, and evaluates on heldout vocab/codebook episodes.

This milestone is intentionally scoped to deterministic synthetic text-flow
operator composition. It does not add E15 long-horizon memory, region-aware
parallel scheduling, external dependencies, or a neural benchmark.

## Search-First Result

Before adding E16B, the repo and fetched refs were searched for:

```text
E16B
OPERATOR_COMPOSITION_DISCOVERY
operator composition discovery
missing macro
missing primitive
primitive ablation
macro ablation
operator grammar
composed operator
mutation discovered operator
text flow operator discovery
text flow composition discovery
support disambiguation
ambiguity abstain
chain length ablation
primitive grammar
E16
E15_TEXT_STREAM_LONG_HORIZON
E14_TEXT_STREAM_COMPOSITION
```

No equivalent E16B implementation was found. Existing hits were adjacent E14/E15
pointers or older non-equivalent primitive inventory references.

## Primary Grammar

Forbidden direct macro operators in primary discovery:

```text
REVERSE
MAP_THEN_REVERSE
REVERSE_THEN_MAP
FILTER_THEN_REVERSE
ROTATE_THEN_MAP
MAP_THEN_ROTATE
SWAP_OUTER_THEN_MAP
```

Allowed lower-level primitives:

```text
SWAP01
SWAP12
SWAP23
ROTL
ROTR
MAP
FILTER_VALID
COPY
COMMIT_OUTPUT
```

`MAP` may execute only when anonymized mapping-table evidence exists.
`FILTER_VALID` may execute only when invalid/decoy evidence exists.

## Candidate Program Schema

```text
program_id
primitive_sequence
chain_len
evidence_fit_score
support_coverage
heldout_coverage
conflict_count
cost
trace_validity
reason_code
```

## Task Families

```text
REVERSE_FROM_SWAPS
MAP_THEN_REVERSE_FROM_PRIMITIVES
REVERSE_THEN_MAP_FROM_PRIMITIVES
FILTER_THEN_REVERSE
ROTATE_THEN_MAP
MAP_THEN_ROTATE
SWAP_OUTER_THEN_MAP
SUPPORT_AMBIGUITY_ABSTAIN_OR_REPAIR
SUPPORT_DISAMBIGUATION
HELDOUT_VOCAB_CODEBOOK
DECOY_HEAVY_COMPOSITION
```

## Systems

```text
RANDOM_LIBRARY_SMALL
RANDOM_LIBRARY_MATCHED_BUDGET
RANDOM_LIBRARY_BEST_OF_N_CONTROL
TRUE_MACRO_LIBRARY_CONTROL
TRUE_PRIMITIVE_HAND_AUTHORED_CONTROL
COMPOSITION_DISCOVERY_NO_GATE
COMPOSITION_DISCOVERY_PRIMARY
COMPOSITION_DISCOVERY_PRUNED_PRIMARY
INSUFFICIENT_CHAIN_LEN_ABLATION
MISSING_ORDER_PRIMITIVES_ABLATION
MISSING_MAP_PRIMITIVE_ABLATION
MISSING_FILTER_PRIMITIVE_ABLATION
AMBIGUOUS_SUPPORT_NO_ABSTAIN_ABLATION
```

The primary candidate is:

```text
COMPOSITION_DISCOVERY_PRUNED_PRIMARY
```

The true macro library is a privileged invalid control and cannot be selected as
primary.

## Positive Gate

The primary passes only if macro removal is confirmed, no direct macro leak is
detected, discovery/composition/heldout/support/order metrics meet the E16B
thresholds, trace/writeback safety holds, deterministic replay passes, random
controls are beaten, and the no-gate control is worse on trace/writeback.

Required ablations must show that chain length, order primitives, MAP,
FILTER_VALID, and ambiguity abstain are functional dependencies rather than
decorative additions.

## Boundary

This confirms grammar-level operator composition discovery from lower-level primitives in a deterministic synthetic controlled text-flow proxy. It does not confirm unconstrained operator invention or general natural-language AI.

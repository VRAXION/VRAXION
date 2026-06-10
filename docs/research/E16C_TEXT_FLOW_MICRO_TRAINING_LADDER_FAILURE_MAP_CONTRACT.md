# E16C Text-Flow Micro-Training Ladder Failure Map Contract

## Purpose

`E16C_TEXT_FLOW_MICRO_TRAINING_LADDER_FAILURE_MAP` is a deterministic
failure-first probe. It starts from a minimal micro-VM and maps how far bounded
micro-program discovery gets across a controlled text-flow curriculum.

The goal is not to optimize only for a full confirm. A useful run must still
write the best stage reached, first failing stage, failure signature, missing
capacity or primitive class, best-so-far program library, and next repair
recommendation.

## Search-First Result

Before adding E16C, local files and fetched refs were searched for:

```text
E16C
MICRO_TRAINING_LADDER
training ladder
failure map
micro primitive
micro VM
operator invention
from micro primitives
text flow training
token boundary discovery
character stream recovery
sentence template
curriculum
mutation trained text flow
failure stage
first failing stage
best stage reached
```

No equivalent E16C implementation was found. Existing hits were the E16B next
pointer and older, non-equivalent curriculum or failure-map references.

## Micro-VM

The primary may search programs composed from:

```text
READ_POS
WRITE_POS
COPY_POS
COMPARE_EQ
IF_EQ
IF_VALID_EVIDENCE
IF_REWRITE_EVIDENCE
ROUTE_TOKEN
KEEP_TOKEN
DROP_TOKEN
COMMIT_OUTPUT
OPEN_MEMORY_SLOT
WRITE_MEMORY_SLOT
READ_MEMORY_SLOT
CLEAR_MEMORY_SLOT
TRACE_CHECK
GATED_COMMIT
```

Direct macro operators are forbidden in the primary library:

```text
REVERSE
ROTATE
SWAP01
SWAP12
SWAP23
MAP
FILTER
BIND
QUERY
MAP_THEN_REVERSE
REVERSE_THEN_MAP
FILTER_THEN_REVERSE
```

## Curriculum

```text
0 RAW_CHAR_STREAM_RECOVERY
1 TOKEN_BOUNDARY_DISCOVERY
2 TOKEN_COPY_AND_ORDER
3 WORD_LEVEL_REWRITE_EVIDENCE
4 FILTER_AND_DECOY_HANDLING
5 PHRASE_COMPOSITION
6 CONTROLLED_SENTENCE_TEMPLATE
7 MULTI_SENTENCE_BINDING_MEMORY
8 NOISY_MULTI_SENTENCE_REPAIR
```

Each stage has a train split, heldout split, randomized codebook, leak audit,
learning curve, and deterministic replay record.

## Systems

```text
RANDOM_MICRO_PROGRAM_BASELINE
GREEDY_SUPPORT_FIT_BASELINE
HAND_MICRO_REFERENCE_CONTROL
MICRO_TRAINING_NO_GATE
MICRO_TRAINING_PRIMARY
MICRO_TRAINING_PRUNED_PRIMARY
NO_REWRITE_MICRO_ABLATION
NO_VALIDITY_MICRO_ABLATION
NO_MEMORY_MICRO_ABLATION
NO_CONDITIONAL_MICRO_ABLATION
TOO_SHORT_PROGRAM_BUDGET_ABLATION
```

The primary candidate is:

```text
MICRO_TRAINING_PRUNED_PRIMARY
```

The hand micro reference is privileged and cannot be selected as primary.

## Decision Logic

Full pass:

```text
decision = e16c_text_flow_micro_training_ladder_confirmed
```

Partial informative pass:

```text
decision = e16c_text_flow_micro_training_ladder_partial_confirmed
```

This requires at least stages 0 through 5 to pass, a coherent first failure
record, complete failure-map artifacts, and clean leak/safety audits.

Early failure:

```text
decision = e16c_text_flow_micro_training_ladder_failed_at_stage_N
```

Invalid:

```text
decision = e16c_invalid_or_incomplete_run
```

## Boundary

This is a deterministic synthetic controlled text-flow micro-training ladder. It maps how far micro-program discovery gets from a minimal micro-VM. It does not prove general natural language AI or unconstrained invention from absolute nothing.

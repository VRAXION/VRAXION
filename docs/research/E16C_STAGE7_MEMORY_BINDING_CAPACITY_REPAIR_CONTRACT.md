# E16C Stage 7 Memory Binding Capacity Repair Contract

## Purpose

`E16C_STAGE7_MEMORY_BINDING_CAPACITY_REPAIR` targets the first bottleneck found
by the E16C micro-training ladder:

```text
best_stage_passed = 6
first_failing_stage = 7
first_failing_stage_name = MULTI_SENTENCE_BINDING_MEMORY
failure_signature = stage_7_memory_binding_capacity_shortfall
failure_reason_code = finite_memory_slots_and_delayed_binding_policy_insufficient
```

This probe focuses only on Stage 7 memory binding policy repair. It does not
start a new full ladder, add a neural baseline, add scheduler work, or use
external datasets.

## Search-First Result

Before adding this repair probe, local files and fetched refs were searched for:

```text
E16C_STAGE7_MEMORY_BINDING_CAPACITY_REPAIR
STAGE7_MEMORY_BINDING
memory binding capacity repair
delayed binding policy
multi sentence binding
finite memory slots
long horizon recall
key value addressing
memory slot capacity
ambiguity repair
stale update rejection
E16C failure map
memory repair policy
```

No equivalent implementation was found. Existing hits were the prior E16C
failure-map recommendation pointing to this repair.

## Runtime Scope

The runtime receives controlled synthetic text-flow episodes with nonce tokens,
multiple statement-like frames, delayed query frames, heldout vocab/codebooks,
and optional decoy or corrupt evidence. It may search over memory policies
composed from micro-ops, but it must not receive oracle answers, task-family
routes, or direct macro routes.

Allowed micro-ops:

```text
READ_POS
WRITE_POS
COPY_POS
COMPARE_EQ
IF_EQ
ROUTE_TOKEN
OPEN_MEMORY_SLOT
WRITE_MEMORY_SLOT
READ_MEMORY_SLOT
CLEAR_MEMORY_SLOT
MEMORY_SLOT_SCORE
TRACE_CHECK
GATED_COMMIT
ABSTAIN_OUTPUT
REPAIR_COMMIT
```

Forbidden macro shortcuts:

```text
BIND
QUERY
MEMORY_LOOKUP_MACRO
KEY_VALUE_BIND_MACRO
REVERSE
MAP
FILTER
```

## Task Families

```text
SINGLE_BIND_DELAYED_QUERY
MULTI_BIND_DELAYED_QUERY
NESTED_BINDING_DEPTH2
NESTED_BINDING_DEPTH3
CAPACITY_PRESSURE
STALE_UPDATE_REJECTION
CORRUPT_THEN_REPAIR
AMBIGUOUS_EVIDENCE_ABSTAIN_OR_REPAIR
DISTRACTOR_GAP
MIXED_MEMORY_AND_TEMPLATE
```

## Systems

```text
E16C_BASELINE_STAGE7_POLICY
LAST_WRITE_MEMORY_NO_GATE
VALID_LAST_MEMORY
MAJORITY_MEMORY_NO_ABSTAIN
FIXED_SLOT_FIFO_MEMORY
FIXED_SLOT_LRU_MEMORY
KEY_ADDRESSED_MEMORY_POLICY
MUTATION_TRAINED_MEMORY_POLICY_PRIMARY
MUTATION_TRAINED_PRUNED_MEMORY_POLICY_PRIMARY
NO_MEMORY_SLOTS_ABLATION
LOW_MEMORY_CAPACITY_ABLATION
NO_STALE_REJECTION_ABLATION
NO_REPAIR_EVIDENCE_ABLATION
NO_AMBIGUITY_ABSTAIN_ABLATION
NO_NESTED_RESOLUTION_ABLATION
```

The primary candidate is:

```text
MUTATION_TRAINED_PRUNED_MEMORY_POLICY_PRIMARY
```

## Repair Gate

The repaired primary passes when binding, recall, ambiguity handling, nested
depth, capacity pressure, stale rejection, corrupt-then-repair, distractor gap,
trace validity, and writeback safety all clear the Stage 7 repair thresholds.

The run also reports a memory capacity sweep over:

```text
1, 2, 3, 4, 6, 8, 12
```

Stage 8 downstream metrics are stretch-only and do not decide this milestone.

## Boundary

This is a deterministic synthetic controlled text-flow Stage 7 memory binding repair probe. It tests targeted mutation/search over memory policies. It does not prove general natural-language AI.

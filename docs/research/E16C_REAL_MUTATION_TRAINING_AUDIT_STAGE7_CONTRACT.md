# E16C Real Mutation Training Audit Stage 7 Contract

## Purpose

`E16C_REAL_MUTATION_TRAINING_AUDIT_STAGE7` audits the earlier Stage 7 memory
binding repair result by replacing static metric fixtures with real
episode-driven candidate evaluation.

The probe must answer whether mutation/search over memory policies genuinely
improves the Stage 7 bottleneck when metrics are computed from generated
train/validation/heldout episodes and actual policy execution.

## Search-First Result

Before adding this audit, local files and fetched refs were searched for:

```text
E16C_REAL_MUTATION_TRAINING_AUDIT_STAGE7
real mutation training
mutation training audit
Stage7 real training
candidate policy population
episode driven training
static metrics audit
hardcoded metrics
base_metrics
capacity_sweep_rows
training_curve interpolation
memory binding capacity repair
E16C_STAGE7_MEMORY_BINDING_CAPACITY_REPAIR
```

No equivalent real-audit implementation was found. The previous Stage 7 repair
probe exists, but this contract treats it as a fixture/spec reference rather
than proof of automated mutation training.

## Non-Fixture Requirement

The runner must not generate primary metrics through:

```text
static base_metrics final rows
static capacity sweep accuracy tables
interpolated training curves from target metrics
aggregate metrics without per-episode evaluation
hardcoded final primary metric values
oracle answer routing during inference
```

The checker must recompute aggregate metrics from:

```text
e16c_real_per_episode_eval_report.json
e16c_real_generation_score_report.json
e16c_real_capacity_sweep_report.json
e16c_real_ablation_report.json
```

## Stage 7 Families

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

## Policy Actions

Allowed policy actions:

```text
READ_TOKEN
COMPARE_TOKEN
WRITE_MEMORY_SLOT
READ_MEMORY_SLOT
CLEAR_MEMORY_SLOT
SCORE_MEMORY_SLOT
ROUTE_KEY
ROUTE_VALUE
UPDATE_CONFIDENCE
REJECT_STALE
APPLY_REPAIR_EVIDENCE
ABSTAIN_IF_AMBIGUOUS
RESOLVE_NESTED
GATED_COMMIT
EMIT_OUTPUT
```

Forbidden shortcuts:

```text
BIND
QUERY
MEMORY_LOOKUP_MACRO
KEY_VALUE_BIND_MACRO
ORACLE_LOOKUP
task_family routing
expected_answer routing
```

## Decision Logic

Full pass:

```text
decision = e16c_real_mutation_training_stage7_confirmed
next = E16C_STAGE8_REAL_MUTATION_REPAIR_CONFIRM
```

Partial:

```text
decision = e16c_real_mutation_training_stage7_partial
next = E16C_STAGE7_REAL_TRAINING_REPAIR_CONTINUE
```

Fail:

```text
decision = e16c_real_mutation_training_stage7_failed
next = E16C_STAGE7_POLICY_SEARCH_REDESIGN
```

Invalid:

```text
decision = e16c_real_mutation_training_stage7_invalid_or_incomplete
next = E16C_REAL_TRAINING_AUDIT_RETRY
```

## Boundary

This confirms real deterministic mutation/search training over Stage 7 memory policies in a controlled synthetic text-flow proxy. It does not prove general natural-language AI or production training readiness.

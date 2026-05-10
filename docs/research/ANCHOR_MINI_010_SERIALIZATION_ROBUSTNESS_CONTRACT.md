# ANCHOR-MINI-010 Serialization Robustness Contract

## Purpose

ANCHOR-MINI-010 tests whether the MINI-009 byte PLAN route depends on one rigid
template, or whether it remains shortcut-resistant across held-out
serialization families.

The narrow claim is:

```text
A schema-aware byte-serialized PLAN-first route remains shortcut-resistant when
the task is the same but the serialization format changes.
```

This is not a learned natural-language parser. It is a schema-aware byte
robustness test.

## Dataset

Reuse the MINI-009 shortcut-flip setup:

```text
candidate_count: 4
category_count: 4
train shortcut: highest surface_prior points to gold with probability 0.90
eval shortcut: highest surface_prior points to wrong candidate with probability 0.90
seeds: 2026-2125
```

Training serialization is always:

```text
canonical_fixed
```

Eval serializations:

```text
canonical_fixed
field_order_swap
slot_order_perm
alias_long
noise_fields
```

Examples:

```text
canonical_fixed:
G=2;A=E0:S9;B=E2:S1;C=E1:S0;D=E3:S2

field_order_swap:
G=2;A=S9:E0;B=S1:E2;C=S0:E1;D=S2:E3

slot_order_perm:
G=2;C=E1:S0;A=E0:S9;D=E3:S2;B=E2:S1

alias_long:
GOAL=2|A(EFFECT=0,SURFACE=9)|B(EFFECT=2,SURFACE=1)|C(EFFECT=1,SURFACE=0)|D(EFFECT=3,SURFACE=2)

noise_fields:
N=7;G=2;A=E0:S9:X4;B=E2:S1:X8;C=E1:S0:X2;D=E3:S2:X5
```

Rules:

```text
No answer label in task_bytes
No match bit in task_bytes
No policy bit in task_bytes
No oracle PLAN bit in task_bytes for non-oracle carriers
```

## Decoder Modes

```text
schema_aware
  Main mode. Parses all locked serialization families from task_bytes.

fixed_template_control
  Diagnostic mode. Uses the MINI-009 fixed-template parser.
  Expected to pass canonical_fixed and lose decode_integrity on held-out formats.
```

Both modes must derive non-oracle carrier features from `task_bytes`, not from
precomputed match/policy bits.

## Carrier Arms

```text
BYTE_DIRECT_ANSWER
BYTE_AUX_PLAN_DIRECT_ANSWER
BYTE_PLAN_FIRST
BYTE_PLAN_FIRST_HYBRID
BYTE_SHUFFLED_TEACHER
BYTE_SHORTCUT_TEACHER
BYTE_ORACLE_PLAN_VISIBLE
```

For `BYTE_PLAN_FIRST`, the final answer must not read oracle answer, oracle
match bits, or the direct surface shortcut as final decision input. It may read
only the predicted PLAN/policy state derived from byte-local decoded features.

## Metrics

Report per carrier, seed, decoder mode, and eval serialization family:

```text
answer_eval_accuracy
shortcut_trap_rate
observed_shortcut_accuracy
shortcut_validity_accuracy
invalid_shortcut_rejection_accuracy
policy_bit_accuracy
plan_exact_row_accuracy
answer_from_plan_consistency
byte_input_integrity
decode_integrity
train_surface_alignment
eval_surface_flip_rate
edge_count
```

Valid main stress requires, under `schema_aware`:

```text
train_surface_alignment >= 0.85
eval_surface_flip_rate >= 0.85
byte_input_integrity == true
decode_integrity == true
valid_seeds >= 80 / 100
```

## Status Rules

Strong positive requires every held-out eval family under `schema_aware` to pass:

```text
BYTE_DIRECT_ANSWER shortcut_trap_rate >= 0.45
BYTE_PLAN_FIRST answer_eval_accuracy >= BYTE_DIRECT_ANSWER + 0.25
BYTE_PLAN_FIRST answer_eval_accuracy >= BYTE_SHUFFLED_TEACHER + 0.25
BYTE_PLAN_FIRST answer_eval_accuracy >= BYTE_SHORTCUT_TEACHER + 0.25
BYTE_PLAN_FIRST shortcut_trap_rate <= 0.25
BYTE_PLAN_FIRST policy_bit_accuracy >= 0.90
BYTE_PLAN_FIRST plan_exact_row_accuracy >= 0.85
BYTE_PLAN_FIRST answer_from_plan_consistency >= 0.95
BYTE_PLAN_FIRST_HYBRID remains directionally positive
BYTE_ORACLE_PLAN_VISIBLE remains upper bound or is matched by learned PLAN
```

Required diagnostic:

```text
fixed_template_control passes canonical_fixed
fixed_template_control loses decode_integrity on held-out variants
```

Statuses:

```text
ANCHOR_MINI_010_SERIALIZATION_STRONG_POSITIVE
ANCHOR_MINI_010_SERIALIZATION_WEAK_POSITIVE
ANCHOR_MINI_010_TEMPLATE_ONLY
ANCHOR_MINI_010_ORACLE_ONLY
ANCHOR_MINI_010_NEGATIVE
ANCHOR_MINI_010_INVALID_STRESS
ANCHOR_MINI_010_RESOURCE_BLOCKED
```

## Claim Boundary

MINI-010 tests schema-aware serialization robustness only. It does not prove a
learned parser, natural-language AnchorCells, Qwen behavior, full INSTNCT
recurrent behavior, or symbol grounding at scale.

Generated outputs stay under:

```text
target/anchorweave/anchor_mini010_serialization_robustness/
```

Do not write under:

```text
data/anchorweave/cells/
```

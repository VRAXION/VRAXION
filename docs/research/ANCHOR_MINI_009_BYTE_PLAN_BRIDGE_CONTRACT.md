# ANCHOR-MINI-009 Byte-Serialized PLAN Bridge Contract

## Purpose

ANCHOR-MINI-009 tests whether the MINI-008 PLAN-first mechanism survives when
the situation is read from a fixed byte-serialized record instead of built-in
symbolic situation features.

The narrow claim is:

```text
A byte-serialized task input can train a PLAN-first sparse carrier to make
shortcut-resistant decisions better than direct answer training, aux-only PLAN
training, shuffled/wrong teacher controls, and shortcut-teacher controls.
```

This is a bridge test:

```text
symbolic process routing -> serialized model-facing carrier
```

It is not a full natural-language AnchorCell or symbol-grounding proof.

## Dataset

Reuse the MINI-008 shortcut-flip setup:

```text
candidate_count: 4
category_count: 4
train shortcut: highest surface_prior points to gold with probability 0.90
eval shortcut: highest surface_prior points to wrong candidate with probability 0.90
seeds: 2026-2125
```

Each example includes a fixed ASCII byte record:

```text
G=2;A=E0:S9;B=E2:S1;C=E1:S0;D=E3:S2
```

Rules:

```text
G = goal category byte
E = candidate effect category byte
S = surface-prior bucket byte
No answer label in task_bytes
No match bit in task_bytes
No policy bit in task_bytes
No oracle PLAN bit in task_bytes for non-oracle carriers
```

The dataset may store raw fields for labels, audit, and metrics. Non-oracle
carrier features must be derived from `task_bytes`, not from precomputed
match/policy bits.

## Deterministic Teacher Labels

For each example:

```text
observed_shortcut_slot = argmax(surface_prior)
shortcut_valid = effect_category(observed_shortcut_slot) == goal_category
shortcut_reject = observed_shortcut_slot if shortcut_valid == false
candidate_match_bits = effect_category(candidate) == goal_category
candidate_policy_bits = choose matching candidate, reject nonmatching candidates
answer_label = candidate with policy choose
```

Wrong-teacher controls:

```text
BYTE_SHUFFLED_TEACHER: policy bits use goal_category + 1 mod 4
BYTE_SHORTCUT_TEACHER: policy bits choose the observed surface shortcut
```

## Carrier Arms

```text
BYTE_DIRECT_ANSWER
  Final answer from direct/surface byte route.

BYTE_AUX_PLAN_DIRECT_ANSWER
  PLAN labels are trained as auxiliary outputs, but final answer remains direct.

BYTE_PLAN_FIRST
  Final answer routes through predicted PLAN/policy state inferred from bytes.

BYTE_PLAN_FIRST_HYBRID
  PLAN route plus direct surface bypass.

BYTE_SHUFFLED_TEACHER
  PLAN route trained against shifted wrong teacher labels.

BYTE_SHORTCUT_TEACHER
  PLAN route trained to praise the surface shortcut.

BYTE_ORACLE_PLAN_VISIBLE
  Oracle policy bits visible to the final route. Upper-bound diagnostic only.
```

For `BYTE_PLAN_FIRST`, the final answer must not read oracle answer, oracle
match bits, or the direct surface shortcut as final decision input. It may read
only the predicted PLAN/policy state derived from byte-local features.

## Metrics

Report per carrier and seed:

```text
answer_eval_accuracy
shortcut_trap_rate
observed_shortcut_accuracy
shortcut_validity_accuracy
invalid_shortcut_rejection_accuracy
policy_bit_accuracy
plan_exact_row_accuracy
answer_from_plan_consistency
train_surface_alignment
eval_surface_flip_rate
byte_input_integrity
edge_count
```

Valid stress requires:

```text
train_surface_alignment >= 0.85
eval_surface_flip_rate >= 0.85
byte_input_integrity == true
valid_seeds >= 80 / 100
```

## Status Rules

Strong positive requires:

```text
BYTE_DIRECT_ANSWER shortcut_trap_rate >= 0.45
BYTE_PLAN_FIRST answer_eval_accuracy >= BYTE_DIRECT_ANSWER + 0.25
BYTE_PLAN_FIRST answer_eval_accuracy >= BYTE_SHUFFLED_TEACHER + 0.25
BYTE_PLAN_FIRST answer_eval_accuracy >= BYTE_SHORTCUT_TEACHER + 0.25
BYTE_PLAN_FIRST shortcut_trap_rate <= 0.25
BYTE_PLAN_FIRST invalid_shortcut_rejection_accuracy >= 0.90
BYTE_PLAN_FIRST policy_bit_accuracy >= 0.90
BYTE_PLAN_FIRST plan_exact_row_accuracy >= 0.85
BYTE_PLAN_FIRST answer_from_plan_consistency >= 0.95
BYTE_PLAN_FIRST_HYBRID remains directionally positive
BYTE_ORACLE_PLAN_VISIBLE is an upper bound or matched by learned PLAN
```

Statuses:

```text
ANCHOR_MINI_009_BYTE_PLAN_STRONG_POSITIVE
ANCHOR_MINI_009_BYTE_PLAN_WEAK_POSITIVE
ANCHOR_MINI_009_ORACLE_ONLY
ANCHOR_MINI_009_NEGATIVE
ANCHOR_MINI_009_INVALID_STRESS
ANCHOR_MINI_009_RESOURCE_BLOCKED
```

## Claim Boundary

MINI-009 proves or falsifies the byte-serialized bridge only in the sparse audit
carrier. It does not prove natural-language AnchorCells, Qwen behavior, full
INSTNCT recurrent behavior, or symbol grounding at scale.

Generated outputs stay under:

```text
target/anchorweave/anchor_mini009_byte_plan_bridge/
```

Do not write under:

```text
data/anchorweave/cells/
```

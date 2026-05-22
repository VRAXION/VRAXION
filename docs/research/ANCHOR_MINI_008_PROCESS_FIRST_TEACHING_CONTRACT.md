# ANCHOR-MINI-008 Process-First Teaching Contract

## Purpose

ANCHOR-MINI-008 tests whether a sparse VRAXION-style audit carrier can learn a
scored PLAN route before answering.

The narrow claim is:

```text
Process-first teaching improves shortcut-resistant eval behavior over direct
answer training and over wrong teacher feedback.
```

This is not free-form chain-of-thought. The "inner voice" is represented as
scored process fields:

```text
fast shortcut detected
shortcut validity checked
invalid shortcut rejected
policy selected
answer emitted from policy
```

## Dataset

Reuse the MINI-005/006 shortcut-flip setup:

```text
candidate_count: 4
category_count: 4
train shortcut: highest surface_prior points to gold with probability 0.90
eval shortcut: highest surface_prior points to wrong candidate with probability 0.90
seeds: 2026-2125
```

The true answer is the candidate whose effect category matches the goal
category.

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
SPARSE_SHUFFLED_TEACHER: policy bits use goal_category + 1 mod 4
SPARSE_SHORTCUT_TEACHER: policy bits choose the observed surface shortcut
```

## Carrier Arms

```text
SPARSE_DIRECT_ANSWER
  Final answer from direct surface path only.

SPARSE_AUX_PLAN_DIRECT_ANSWER
  PLAN labels are trained as auxiliary outputs, but final answer remains direct.

SPARSE_PLAN_FIRST
  Final answer routes through predicted PLAN/policy state.

SPARSE_PLAN_FIRST_HYBRID
  PLAN route plus direct surface bypass.

SPARSE_SHUFFLED_TEACHER
  PLAN route trained against shifted wrong teacher labels.

SPARSE_SHORTCUT_TEACHER
  PLAN route trained to praise the surface shortcut.

SPARSE_ORACLE_PLAN_VISIBLE
  Oracle policy bits visible to the final route. Upper-bound diagnostic only.
```

For `SPARSE_PLAN_FIRST`, the final answer must not read oracle answer, oracle
match bits, or direct surface shortcut as final decision input. It may read only
the predicted PLAN/policy state.

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
edge_count
```

Valid stress requires:

```text
train_surface_alignment >= 0.85
eval_surface_flip_rate >= 0.85
valid_seeds >= 80 / 100
```

## Status Rules

Strong positive requires:

```text
SPARSE_DIRECT_ANSWER shortcut_trap_rate >= 0.45
SPARSE_PLAN_FIRST answer_eval_accuracy >= SPARSE_DIRECT_ANSWER + 0.25
SPARSE_PLAN_FIRST answer_eval_accuracy >= SPARSE_SHUFFLED_TEACHER + 0.25
SPARSE_PLAN_FIRST answer_eval_accuracy >= SPARSE_SHORTCUT_TEACHER + 0.25
SPARSE_PLAN_FIRST shortcut_trap_rate <= 0.25
SPARSE_PLAN_FIRST invalid_shortcut_rejection_accuracy >= 0.90
SPARSE_PLAN_FIRST policy_bit_accuracy >= 0.90
SPARSE_PLAN_FIRST plan_exact_row_accuracy >= 0.85
SPARSE_PLAN_FIRST answer_from_plan_consistency >= 0.95
SPARSE_PLAN_FIRST_HYBRID remains directionally positive
SPARSE_ORACLE_PLAN_VISIBLE is an upper bound or matched by learned plan
```

Statuses:

```text
ANCHOR_MINI_008_PROCESS_FIRST_STRONG_POSITIVE
ANCHOR_MINI_008_PROCESS_FIRST_WEAK_POSITIVE
ANCHOR_MINI_008_ORACLE_ONLY
ANCHOR_MINI_008_NEGATIVE
ANCHOR_MINI_008_INVALID_STRESS
ANCHOR_MINI_008_RESOURCE_BLOCKED
```

## Claim Boundary

MINI-008 proves or falsifies process-first teaching only in the sparse audit
carrier. It does not prove natural-language AnchorCells, Qwen behavior, full
INSTNCT recurrent behavior, or symbol grounding at scale.

Generated outputs stay under:

```text
target/anchorweave/anchor_mini008_process_teacher/
```

Do not write under:

```text
data/anchorweave/cells/
```

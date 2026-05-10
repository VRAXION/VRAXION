# ANCHOR-MINI-005 Internalization Contract

## Purpose

ANCHOR-MINI-005 tests whether AnchorCell-style process supervision can be
internalized by a carrier, rather than merely used when the process answer is
handed to the final decision route.

Narrow claim under test:

```text
Process supervision improves shortcut-flip OOD behavior when the process signal
is used during training, then hidden from the final eval input.
```

This is not a natural-language AnchorCell test, not a full INSTNCT recurrent
proof, and not a grounding-at-scale claim.

## Research Frame

MINI-005 is closest to:

- Learning Using Privileged Information: process labels are available during
  training but not as oracle inputs at eval.
- Generalized distillation: teacher/process structure is transferred into a
  student that later runs from ordinary inputs.
- Concept/process bottleneck learning: the decision path may route through a
  learned process state, but oracle concept leakage must be guarded against.

## Dataset

Use the MINI-003 shortcut-flip task:

```text
candidate_count: 4
category_count: 4
goal/effect inputs: symbolic ids
train shortcut: highest surface_prior points to gold with probability 0.90
eval shortcut: highest surface_prior points to wrong candidate with probability 0.90
```

The true answer is the candidate whose effect category matches the goal
category. `match_bits` are allowed as labels and oracle upper-bound features,
but are not allowed as eval inputs for learned-process carriers.

## Visibility Regimes

```text
ORACLE_PROCESS_VISIBLE
  match/process bits are visible to the decision route.
  Upper-bound sanity only.

PROCESS_TRAIN_ONLY
  process labels are training targets/fitness terms.
  They are not exposed as decision inputs at eval.

SITUATION_ONLY
  only raw goal/effect ids and surface priors are visible.
  No match bits and no oracle process inputs.
```

Primary evidence can only come from `PROCESS_TRAIN_ONLY` / `SITUATION_ONLY`.
Oracle runs are upper bounds.

## Sparse Carriers

```text
SPARSE_DIRECT
  direct surface route only; expected shortcut failure.

SPARSE_ORACLE_ROUTED
  final answer routes through visible match bits.
  Upper bound only.

SPARSE_LEARNED_PROCESS
  process branch infers match from raw goal/effect category indicators.
  Final answer routes through the learned process state.

SPARSE_LEARNED_HYBRID
  learned process route plus direct surface bypass.

SPARSE_SHUFFLED_PROCESS
  learned process route trained against intentionally wrong process labels.
  In this audit carrier the wrong label is the candidate whose effect category
  is `goal_category + 1 mod 4`, so the negative control is learnable but
  semantically misaligned.
  Negative control.
```

Allowed non-oracle eval features:

```text
surface_prior(candidate)
goal_category_indicator(k)
effect_category_indicator(candidate,k)
goal_and_effect_category_conjunction(candidate,k)
goal_and_shifted_effect_category_conjunction(candidate,k)
bias
```

`goal_and_effect_category_conjunction` is a raw category gate. It is allowed
because it is computed from ordinary goal/effect inputs, not from precomputed
`match_bits`.

Forbidden for non-oracle carriers:

```text
match_bits(candidate)
shuffled_match_bits(candidate)
answer_label
surface_shortcut_label
```

## Metrics

Report per carrier and per seed:

```text
answer_eval_ood_accuracy
shortcut_trap_rate
process_bit_accuracy
process_exact_row_accuracy
surface_shortcut_train_alignment
surface_shortcut_eval_flip_rate
oracle_gap
internalization_gap
shuffled_control_gap
```

Valid stress requires:

```text
surface_shortcut_train_alignment >= 0.85
surface_shortcut_eval_flip_rate >= 0.85
```

Process fitness must be balanced across positive and negative process bits. A
carrier must not get high process fitness by predicting all sparse process bits
as absent while still using relative logits to answer correctly.

Default process auxiliary weight:

```text
aux_weight: 4.0
```

## Status Rules

```text
ANCHOR_MINI_005_INTERNALIZATION_STRONG_POSITIVE
ANCHOR_MINI_005_INTERNALIZATION_WEAK_POSITIVE
ANCHOR_MINI_005_ORACLE_ONLY
ANCHOR_MINI_005_NEGATIVE
ANCHOR_MINI_005_INVALID_STRESS
ANCHOR_MINI_005_RESOURCE_BLOCKED
```

Strong positive requires:

```text
valid seeds >= 80 / 100
SPARSE_LEARNED_PROCESS OOD accuracy >= SPARSE_DIRECT + 0.25
SPARSE_LEARNED_PROCESS OOD accuracy >= SPARSE_SHUFFLED_PROCESS + 0.25
SPARSE_LEARNED_PROCESS shortcut_trap_rate <= 0.25
SPARSE_ORACLE_ROUTED OOD accuracy >= 0.90
SPARSE_ORACLE_ROUTED shortcut_trap_rate <= 0.10
SPARSE_LEARNED_HYBRID remains directionally positive
SPARSE_SHUFFLED_PROCESS does not reproduce the improvement
```

Interpretation:

```text
STRONG_POSITIVE:
  process has been partially internalized.

ORACLE_ONLY:
  carrier works only when handed process bits.

NEGATIVE:
  process supervision did not survive masking.
```

## Hygiene

Generated outputs stay under:

```text
target/anchorweave/anchor_mini005_internalization/
```

Do not commit raw target outputs. Do not write under:

```text
data/anchorweave/cells/
```

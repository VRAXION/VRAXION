# ANCHOR-MINI-003 / Surface-Biased Training Signal Stress Test

Status: `preregistered_toy_training_view_stress_test`

This probe tests a stricter AnchorCell training-view claim than
`ANCHOR-MINI-002`:

```text
Decomposed AnchorCell-style auxiliary supervision improves OOD trap resistance
over answer-only training when train/eval input is identical and surface
shortcuts flip.
```

This is not an LLM prompt test, not an `AnchorWeave-v1.0` export, and not a
natural-language AnchorCell test. It is a deterministic toy
representation-learning stress test.

## Task

Each example contains:

```text
goal_id
four candidate effect_ids
surface_prior for each candidate
```

Every `goal_id` belongs to one hidden goal category. Every `effect_id` belongs
to one hidden effect category.

Gold answer:

```text
choose the candidate whose effect category matches the goal category
```

The surface-prior shortcut deliberately flips:

```text
train: highest surface_prior points to gold with probability 0.90
eval:  highest surface_prior points to a wrong candidate with probability 0.90
```

## Training Arms

All arms receive the same input features at train and eval time.

```text
ANSWER_ONLY
ANCHOR_MULTI_TASK
SHUFFLED_ANCHOR_MULTI_TASK
```

Targets:

```text
ANSWER_ONLY:
  answer label only

ANCHOR_MULTI_TASK:
  answer label
  goal category
  candidate effect categories
  candidate_matches_goal bits

SHUFFLED_ANCHOR_MULTI_TASK:
  answer label
  shuffled auxiliary labels
```

The shuffled arm tests whether improvement comes from correct auxiliary
structure, not merely extra loss terms.

## Model

```text
tiny_mlp
```

The model is CPU-only and deterministic. It uses one shared nonlinear trunk.
The answer logits are category-compatibility scores between the predicted goal
category and each predicted candidate effect category. This keeps the
architecture identical across arms while making the stress fair: answer-only can
learn the compatibility path implicitly from answer labels, while the anchor arm
receives explicit supervision for the latent categories that the answer path
uses.

## Metrics

```text
answer_train_accuracy
answer_eval_ood_accuracy
goal_category_eval_accuracy
effect_category_eval_accuracy
match_bit_accuracy
match_positive_accuracy
match_exact_row_accuracy
shortcut_trap_rate
surface_shortcut_train_alignment
surface_shortcut_eval_flip_rate
```

## Verdict

Statuses:

```text
ANCHOR_MINI_003_STRONG_POSITIVE
ANCHOR_MINI_003_WEAK_POSITIVE
ANCHOR_MINI_003_NEGATIVE
ANCHOR_MINI_003_INVALID_STRESS
ANCHOR_MINI_003_RESOURCE_BLOCKED
```

`INVALID_STRESS` if any seed fails:

```text
ANSWER_ONLY shortcut_trap_rate >= 0.45
surface_shortcut_eval_flip_rate >= 0.85
```

`STRONG_POSITIVE` requires all seeds to pass:

```text
ANCHOR_MULTI_TASK answer_eval_ood_accuracy >= ANSWER_ONLY + 0.25
ANCHOR_MULTI_TASK answer_eval_ood_accuracy >= SHUFFLED_ANCHOR_MULTI_TASK + 0.25
ANCHOR_MULTI_TASK answer_eval_ood_accuracy >= 0.65
ANCHOR_MULTI_TASK shortcut_trap_rate <= 0.25
ANCHOR_MULTI_TASK goal_category_eval_accuracy >= 0.90
ANCHOR_MULTI_TASK effect_category_eval_accuracy >= 0.90
SHUFFLED_ANCHOR_MULTI_TASK goal_category_eval_accuracy <= 0.50
SHUFFLED_ANCHOR_MULTI_TASK effect_category_eval_accuracy <= 0.50
```

`WEAK_POSITIVE` requires at least 4/5 seeds to pass and no seed may reverse the
main answer effect:

```text
ANCHOR_MULTI_TASK answer_eval_ood_accuracy > ANSWER_ONLY
ANCHOR_MULTI_TASK answer_eval_ood_accuracy > SHUFFLED_ANCHOR_MULTI_TASK
```

Otherwise:

```text
ANCHOR_MINI_003_NEGATIVE
```

## Claim Boundary

Positive evidence means decomposed anchor supervision can improve
shortcut-resistant OOD generalization in a toy representation-learning setting
while keeping eval input identical. It does not prove Qwen LoRA behavior,
VRAXION architecture advantage, natural-language AnchorCells, or grounding at
scale.

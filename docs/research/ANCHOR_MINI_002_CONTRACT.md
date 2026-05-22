# ANCHOR-MINI-002 / Training View Effect Toy A/B

Status: `preregistered_toy_training_view_ab`

This probe tests the smallest direct AnchorCell training-view claim:

```text
Does decomposed anchor supervision improve held-out answer generalization over
answer-only training when train/eval inputs are identical?
```

This is not an LLM prompt test, not an `AnchorWeave-v1.0` export, and not a
natural-language AnchorCell test. It is a toy representation-learning A/B.

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

Surface-prior features are included as shortcut traps. They can make a wrong
candidate look attractive without changing the gold rule.

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

The model uses one shared nonlinear trunk. The answer logits are the same
candidate-match logits used by the auxiliary `candidate_matches_goal` target.
This keeps the architecture identical across arms while making the decomposition
test fair: answer-only training learns the match logits implicitly through
answer cross-entropy, while the anchor arm receives explicit per-candidate match
supervision.

A linear model is intentionally not a primary gate here:
category equality over independent one-hot goal/effect IDs requires interaction
features or nonlinear representation learning. This probe tests whether
anchor-style auxiliary training shapes that shared representation.

## Metrics

```text
answer_train_accuracy
answer_eval_accuracy
goal_category_eval_accuracy
effect_category_eval_accuracy
match_eval_accuracy
match_positive_eval_accuracy
match_exact_row_eval_accuracy
shortcut_trap_rate
```

`answer_eval_accuracy` is the primary metric. Auxiliary accuracies are
diagnostic: they show whether the anchor arm learned the intended latent
structure. `match_eval_accuracy` is reported only as an imbalanced diagnostic
because each row has one positive match and three negative matches.

## Verdict

`ANCHOR_MINI_002_POSITIVE` only if the model passes:

```text
ANCHOR_MULTI_TASK answer_eval_accuracy >= ANSWER_ONLY + 0.15
ANCHOR_MULTI_TASK answer_eval_accuracy >= SHUFFLED_ANCHOR_MULTI_TASK + 0.20
ANCHOR_MULTI_TASK goal_category_eval_accuracy >= 0.90
ANCHOR_MULTI_TASK effect_category_eval_accuracy >= 0.90
ANCHOR_MULTI_TASK shortcut_trap_rate <= ANSWER_ONLY
SHUFFLED_ANCHOR_MULTI_TASK goal_category_eval_accuracy <= 0.50
SHUFFLED_ANCHOR_MULTI_TASK effect_category_eval_accuracy <= 0.50
```

Otherwise:

```text
ANCHOR_MINI_002_NEGATIVE
```

## Claim Boundary

Positive evidence means decomposed anchor supervision can improve answer
generalization in a toy representation-learning setting while keeping eval
input identical. It does not prove natural-language AnchorCells, LLM prompting,
or grounding at scale.

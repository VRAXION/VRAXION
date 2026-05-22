# STABLE_LOOP_PHASE_LOCK_029_HARD_NEGATIVE_GRAMMAR_DISTILLATION Result

Status: complete.

## Verdicts

```text
EXTERNAL_GRAMMAR_TEACHER_STILL_REQUIRED
HAND_GRAMMAR_SUPERVISION_REFERENCE_REPRODUCED
RANDOM_HARD_NEGATIVE_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
WEAK_LABELS_STILL_INSUFFICIENT
PRODUCTION_API_NOT_READY
```

## Run

Quick selector:

```text
seeds = 2026
eval_examples = 512
widths = 8,12
path_lengths = 4,8,16,24
ticks = 8,16,24,32
completed rows = 3584
```

Smoke was skipped because no non-hand hard-negative arm reached the positive gate.

## Reference

The explicit grammar reference still reproduces the 027 upper bound:

| Arm | Suff final | Long path | Family min | Wrong-if-delivered | Retained successor | Route order |
|---|---:|---:|---:|---:|---:|---:|
| HAND_GRAMMAR_SUPERVISION_REFERENCE | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 |

## Hard-Negative Result

No targeted hard-negative arm closed the 028 gap. The best hard-negative variants reproduced the same high-aggregate / low-grammar failure mode:

| Arm | Suff final | Long path | Family min | Wrong-if-delivered | Retained successor | Route order |
|---|---:|---:|---:|---:|---:|---:|
| COUNTERFACTUAL_CORRUPTION_028_BASELINE | 0.984 | 0.957 | 0.000 | 0.018 | 0.779 | 0.769 |
| HARD_NEGATIVE_MIXED_DISTILLATION | 0.984 | 0.957 | 0.000 | 0.018 | 0.779 | 0.769 |
| HARD_NEGATIVE_CURRICULUM_SHORT_TO_LONG | 0.984 | 0.957 | 0.000 | 0.018 | 0.779 | 0.769 |
| HARD_NEGATIVE_TEACHER_STUDENT | 0.984 | 0.957 | 0.000 | 0.018 | 0.779 | 0.769 |
| HIGH_AGGREGATE_LOW_FAMILY_MIN_NEGATIVES | 0.984 | 0.957 | 0.000 | 0.018 | 0.779 | 0.769 |
| SHORTCUT_DELIVERS_WRONG_ORDER_NEGATIVES | 0.984 | 0.957 | 0.000 | 0.018 | 0.779 | 0.769 |

The branch/cycle/duplicate-delivery hard negatives were weaker and did not improve route-order recovery:

| Arm | Suff final | Long path | Family min | Wrong-if-delivered | Retained successor | Route order |
|---|---:|---:|---:|---:|---:|---:|
| CYCLE_REACHES_TARGET_NEGATIVES | 0.909 | 0.850 | 0.000 | 0.054 | 0.501 | 0.489 |
| BRANCH_REACHES_TARGET_NEGATIVES | 0.909 | 0.850 | 0.000 | 0.098 | 0.500 | 0.488 |
| DUPLICATE_DELIVERY_NEGATIVES | 0.909 | 0.850 | 0.000 | 0.098 | 0.500 | 0.488 |

## Controls

The controls failed as required:

| Arm | Suff final | Long path | Wrong-if-delivered | Gate shuffle collapse |
|---|---:|---:|---:|---:|
| RANDOM_HARD_NEGATIVE_CONTROL | 0.692 | 0.400 | 0.342 | 0.261 |
| RANDOM_PHASE_RULE_CONTROL | 0.497 | 0.521 | 0.408 | 0.074 |

## Interpretation

029 confirms that generic hard negatives are not enough. They preserve the 028 partial signal:

```text
aggregate delivery improves
wrong-if-delivered stays low
gate-shuffle control still collapses
```

But they do not recover the full route grammar:

```text
family_min_accuracy = 0.000
route_order_accuracy ~= 0.769
retained_successor_accuracy ~= 0.779
missing_successor_count ~= 5.3
```

The current blocker is narrower than 028:

```text
FAMILY-MIN-SPECIFIC / ORDER-COMPLETE GRAMMAR TEACHER STILL REQUIRED
```

Hard negatives must target the exact family-min collapse and missing-successor route-order failures, not just high-aggregate traps.

## Claim Boundary

029 is a hard-negative grammar distillation diagnostic. This negative result means the tested weak/hard-negative label sources do not replace the explicit route-grammar teacher. It does not prove production routing, full VRAXION, language grounding, consciousness, Prismion uniqueness, biological equivalence, or physical quantum behavior.

# STABLE_LOOP_PHASE_LOCK_143V_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_PROTOTYPE Result

143V records the expected prototype repair for duplicate selected marker conflict rejection.

Boundary: constrained helper/backend evidence only. This is prompt-visible selected-pocket binding only, not rule metadata reasoning, not open-ended arbitration, not GPT-like/open-domain/broad assistant capability, not production/public API/deployment/safety readiness, and not architecture superiority.

## Expected Result

If positive:

```text
decision = selected_marker_occurrence_count_rejection_prototype_positive
next = 143W_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_SCALE_CONFIRM
```

## What It Proves

143V positive would prove only a narrow helper primitive repair:

```text
selected marker candidate-line count == 1 -> same-line extraction
selected marker candidate-line count != 1 -> fallback
```

It does not prove rule metadata reasoning, open-ended arbitration, GPT-like behavior, production readiness, or architecture superiority.

## Required Controls

143V must demonstrate:

```text
duplicate selected marker conflict -> fallback
duplicate selected marker same value -> fallback for now
duplicate non-selected marker conflict -> selected marker binding still succeeds
selected marker mention in prose -> not counted
selected marker prose line start with non-value text -> not counted
blank selected marker followed by another value line -> fallback, no following-line leak
single selected marker -> binding still succeeds
zero selected marker -> fallback
legacy decoder behavior -> unchanged
```

The helper source diff audit must show that only `_instnct_select_rule_selected_pocket_value` changed, while request validation, allowed and forbidden request keys, old decoder behavior, raw generation, and non-INSTNCT paths remain unchanged.

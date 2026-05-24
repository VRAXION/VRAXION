# STABLE_LOOP_PHASE_LOCK_143U_DUPLICATE_SELECTED_MARKER_CONFLICT_REJECTION_HELPER_PRIMITIVE_PLAN Result

143U records the expected planning-only repair recommendation after 143R. It is not a helper patch, not helper generation, not training, not checkpoint mutation, and not a request-key change.

Boundary: planning-only constrained helper/backend evidence only. This is prompt-visible selected-pocket binding only, not rule metadata reasoning, not open-ended arbitration, not GPT-like/open-domain/broad assistant capability, not production/public API/deployment/safety readiness, and not architecture superiority.

## Expected Decision

```text
decision = duplicate_selected_marker_conflict_rejection_primitive_plan_recommended
selected_option = selected_marker_occurrence_count_must_equal_one
next = 143V_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_PROTOTYPE
```

## Repair Recommendation

The selected policy is conservative:

```text
0 selected marker candidate-lines -> fallback
1 selected marker candidate-line -> extract value
2+ selected marker candidate-lines -> fallback
```

Same-value duplicate selected marker acceptance is deferred. The claim is not that same-value duplicates can never be accepted; the claim is that the next safe invariant is exactly one selected marker candidate-line.

143V should count selected marker candidate-lines by line prefix, not by raw substring occurrence:

```text
line.strip().startswith(selected_marker)
```

This avoids false duplicate rejection when the marker text is mentioned in prose or instructions.

## Target 143V Controls

The 143V prototype must prove that the repair is narrow:

```text
duplicate selected marker conflict -> fallback
duplicate selected marker same value -> fallback for now
duplicate non-selected marker -> selected marker binding still succeeds
selected marker mention in prose -> no false duplicate fallback
single selected marker -> binding still succeeds
zero selected marker -> fallback
```

143V must only change `_instnct_select_rule_selected_pocket_value`; request validation, allowed request keys, forbidden request keys, old decoder paths, and non-INSTNCT generation paths must remain unchanged.

If 143V passes, the next route is:

```text
143W_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_SCALE_CONFIRM
```

If it is overbroad and rejects non-selected duplicates, route to:

```text
143Y_NON_SELECTED_MARKER_DUPLICATE_REGRESSION_ANALYSIS
```

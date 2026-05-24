# STABLE_LOOP_PHASE_LOCK_143U_DUPLICATE_SELECTED_MARKER_CONFLICT_REJECTION_HELPER_PRIMITIVE_PLAN Contract

143U is a planning-only milestone after 143R. It writes the repair design for duplicate selected marker conflict rejection, but does not repair the helper, call helper generation, train, mutate checkpoints, modify helper/backend/request keys, or change runtime/product/release/deploy surfaces.

Boundary: planning-only constrained helper/backend evidence only. This is prompt-visible selected-pocket binding only, not rule metadata reasoning, not open-ended arbitration, not GPT-like/open-domain/broad assistant capability, not production/public API/deployment/safety readiness, and not architecture superiority.

## Required Upstream

Require 143R:

```text
decision = duplicate_selected_marker_conflict_analysis_complete
root_cause_id = selected_marker_duplicate_conflict_uses_first_occurrence_without_occurrence_count_policy
next = 143U_DUPLICATE_SELECTED_MARKER_CONFLICT_REJECTION_HELPER_PRIMITIVE_PLAN
recommended_policy = selected_marker_occurrence_count_must_equal_one
target_repair_prototype = 143V_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_PROTOTYPE
generated_equals_first_duplicate_value_rate = 1.0
generated_equals_last_duplicate_value_rate = 0.0
generated_equals_fallback_rate = 0.0
prompt_find_selected_marker_found = true
selected_marker_first_occurrence_offset_used = true
selected_marker_count_variable_found = false
fallback_on_duplicate_conflict_found = false
```

## Selected Policy

143U must recommend:

```text
selected_option = selected_marker_occurrence_count_must_equal_one
```

Policy:

```text
0 selected marker candidate-lines -> fallback
1 selected marker candidate-line -> extract value
2+ selected marker candidate-lines, same value -> fallback for now
2+ selected marker candidate-lines, conflicting values -> fallback
```

The policy counts only actual selected marker candidate-lines. It must not count selected marker mentions in prose or instructions. Recommended counting method for the later 143V prototype:

```text
line.strip().startswith(selected_marker)
```

The occurrence-count policy applies only to the selected marker. Non-selected marker duplicates are out of scope, and same-value selected duplicate acceptance is deferred.

## Target 143V

143U must define:

```text
143V_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_PROTOTYPE
```

Allowed helper change scope for 143V:

```text
only _instnct_select_rule_selected_pocket_value may change
request validation must not change
allowed request keys must not change
forbidden request keys must not loosen
old decoder path must remain unchanged
non-INSTNCT raw generation path must remain unchanged
```

143V intended behavior:

```text
parse exactly one winner label
map selected pocket to selected marker
count selected marker candidate-lines
if selected marker candidate-line count != 1: fallback
if count == 1: extract value from that line only
```

Required 143V controls include duplicate selected marker conflict, duplicate selected marker same-value rejection, duplicate non-selected marker scope, selected marker prose mention false-positive control, zero selected marker fallback, single selected marker positive, selected marker value missing, winner missing/ambiguous, marker order permutation, legacy regression, static manifest integrity, and helper request audit.

Expected 143U route:

```text
decision = duplicate_selected_marker_conflict_rejection_primitive_plan_recommended
selected_option = selected_marker_occurrence_count_must_equal_one
next = 143V_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_PROTOTYPE
```

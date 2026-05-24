# STABLE_LOOP_PHASE_LOCK_143R_DUPLICATE_SELECTED_MARKER_CONFLICT_ANALYSIS Result

143R records the expected artifact-only duplicate selected marker conflict analysis after 143P.

Boundary: constrained helper/backend evidence only. This is prompt-visible selected-pocket binding only, not rule metadata reasoning, not open-ended arbitration, not GPT-like/open-domain/broad assistant capability, not production/public API/deployment/safety readiness, and not architecture superiority.

## Expected Result

143R should complete with:

```text
decision = duplicate_selected_marker_conflict_analysis_complete
root_cause_id = selected_marker_duplicate_conflict_uses_first_occurrence_without_occurrence_count_policy
next = 143U_DUPLICATE_SELECTED_MARKER_CONFLICT_REJECTION_HELPER_PRIMITIVE_PLAN
```

The result is not a new capability win and not a repair. It is a machine-checkable diagnosis that the 143P main selected-pocket binding path works, while duplicate selected marker conflicts currently use first occurrence extraction.

## Evidence Chain

143P showed:

```text
winner_label_parse_accuracy = 1.0
selected_pocket_to_marker_binding_accuracy = 1.0
pocket_marker_order_permutation_accuracy = 1.0
duplicate_selected_marker_conflict_rejection_rate = 0.0
duplicate_selected_marker_first_value_rate = 1.0
duplicate_selected_marker_last_value_rate = 0.0
```

143R must add source-level analysis:

```text
selected_function_name = _instnct_select_rule_selected_pocket_value
prompt_find_selected_marker_found = true
selected_marker_first_occurrence_offset_used = true
all_occurrence_scan_found = false
selected_marker_count_variable_found = false
duplicate_selected_marker_conflict_rejection_found = false
fallback_on_duplicate_conflict_found = false
```

## Interpretation

The failure is a narrow helper edge-policy gap:

```text
selected marker duplicate conflict -> first occurrence selected
```

It is not a winner-label parser failure, static marker map failure, marker-order shortcut, request metadata oracle, per-row manifest switch, legacy regression, selected-marker-missing gap, or broad reasoning/architecture failure.

The next safe policy is:

```text
selected_marker_occurrence_count_must_equal_one
```

That means:

```text
0 selected marker occurrences -> fallback
1 selected marker occurrence -> extract value
2+ selected marker occurrences -> fallback
```

143U should plan that repair without modifying the helper. The first helper-changing prototype should be a later milestone such as `143V_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_PROTOTYPE`.

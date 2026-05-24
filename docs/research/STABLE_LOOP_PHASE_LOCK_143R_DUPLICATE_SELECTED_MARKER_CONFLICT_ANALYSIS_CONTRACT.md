# STABLE_LOOP_PHASE_LOCK_143R_DUPLICATE_SELECTED_MARKER_CONFLICT_ANALYSIS Contract

143R is artifact-only duplicate selected marker conflict analysis after the 143P selected-pocket binding scale edge gap. It reads 143P artifacts and helper source text only. It does not repair the helper, call helper generation, train, mutate checkpoints, modify helper/backend/request keys, or change runtime/product/release/deploy surfaces.

Boundary: constrained helper/backend evidence. This is prompt-visible selected-pocket binding only, not rule metadata reasoning, not open-ended arbitration, not GPT-like/open-domain/broad assistant capability, not production/public API/deployment/safety readiness, and not architecture superiority.

## Required Upstream

Require 143P:

```text
decision = duplicate_selected_marker_conflict_not_rejected
next = 143R_DUPLICATE_SELECTED_MARKER_CONFLICT_ANALYSIS
winner_label_parse_accuracy = 1.0
selected_pocket_to_marker_binding_accuracy = 1.0
pocket_marker_order_permutation_accuracy = 1.0
main_pocket_writeback_rate = 1.0
duplicate_selected_marker_conflict_rejection_rate = 0.0
duplicate_selected_marker_first_value_rate = 1.0
duplicate_selected_marker_last_value_rate = 0.0
legacy_manifest_regression_passed = true
deterministic_replay_passed = true
```

## Required Evidence

143R must write:

```text
duplicate_conflict_trace_report.json
duplicate_conflict_failure_mode_report.json
helper_duplicate_marker_semantics_audit.json
selected_marker_occurrence_policy_matrix.json
alternative_hypothesis_matrix.json
root_cause_report.json
repair_options_matrix.json
target_143u_milestone_plan.json
decision.json
summary.json
report.md
```

The trace report must show:

```text
duplicate_rows_count > 0
selected_marker_occurrence_count_min >= 2
selected_marker_occurrence_count_max >= 2
generated_equals_first_duplicate_value_rate = 1.0
generated_equals_last_duplicate_value_rate = 0.0
generated_equals_fallback_rate = 0.0
generated_equals_unexpected_value_rate = 0.0
```

The helper source audit must be function-level and identify `_instnct_select_rule_selected_pocket_value` using `prompt.find(selected_marker)` as the selected marker extraction offset, with no all-occurrence scan, selected-marker count policy, duplicate conflict rejection branch, or fallback-on-duplicate branch.

## Decision

Expected decision:

```text
decision = duplicate_selected_marker_conflict_analysis_complete
root_cause_id = selected_marker_duplicate_conflict_uses_first_occurrence_without_occurrence_count_policy
next = 143U_DUPLICATE_SELECTED_MARKER_CONFLICT_REJECTION_HELPER_PRIMITIVE_PLAN
```

The conservative repair policy for the next plan is:

```text
0 selected marker occurrences -> fallback
1 selected marker occurrence -> extract value
2+ selected marker occurrences, same value -> fallback for now
2+ selected marker occurrences, conflicting values -> fallback
recommended_policy = selected_marker_occurrence_count_must_equal_one
```

The occurrence-count policy applies only to the selected marker. Non-selected marker duplicates are out of scope for this failure. Same-value duplicate acceptance is deferred and is not declared permanently unsafe.

## Target 143U

143U must remain planning-only. It must not modify `shared_raw_generation_helper.py`. It should route the first repair prototype to a later milestone:

```text
143V_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_PROTOTYPE
```

143U must forbid helper request key changes, per-row selected-pocket metadata, per-row manifest switching, narrowed payload marker lists, hidden final/winner-value/gold/answer markers, and broad architecture claims.

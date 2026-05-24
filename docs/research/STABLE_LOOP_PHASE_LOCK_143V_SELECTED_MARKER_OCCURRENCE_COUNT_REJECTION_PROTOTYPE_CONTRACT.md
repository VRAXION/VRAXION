# STABLE_LOOP_PHASE_LOCK_143V_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_PROTOTYPE Contract

143V is a helper-changing prototype after 143U. It repairs the duplicate selected marker conflict edge only inside `_instnct_select_rule_selected_pocket_value`, behind the existing manifest-gated selected-pocket binding decoder.

Boundary: constrained helper/backend evidence only. This is prompt-visible selected-pocket binding only, not rule metadata reasoning, not open-ended arbitration, not GPT-like/open-domain/broad assistant capability, not production/public API/deployment/safety readiness, and not architecture superiority.

## Required Upstream

Require 143U:

```text
decision = duplicate_selected_marker_conflict_rejection_primitive_plan_recommended
selected_option = selected_marker_occurrence_count_must_equal_one
next = 143V_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_PROTOTYPE
```

## Helper Scope

143V may change only `_instnct_select_rule_selected_pocket_value`.

It must not change request validation, allowed request keys, forbidden request keys, the old decoder path, non-INSTNCT raw generation, deployment/runtime/product/release surfaces, or root license files.

The selected marker candidate-line grammar is:

```text
^\s*<escaped selected_marker>\s*((EV|VAL|SYM)[A-Za-z0-9_+\-]*)?\s*$
```

Rules:

```text
0 selected marker candidate-lines -> fallback
1 selected marker candidate-line with valid value -> extract that same-line value
1 selected marker candidate-line without valid value -> fallback selected_marker_value_missing
2+ selected marker candidate-lines -> fallback selected_marker_duplicate_conflict
```

The helper must not count raw substring occurrences, quoted marker mentions, prose or instruction lines that start with the marker but contain non-value text, or non-selected duplicate marker lines. Extraction must be same-line only and must not scan following lines or a 128-character segment after the marker.

## Required Evidence

143V must write:

```text
shared_helper_diff_audit.json
selected_marker_occurrence_count_report.json
selected_marker_candidate_line_parser_report.json
duplicate_selected_marker_conflict_report.json
duplicate_selected_marker_same_value_report.json
duplicate_non_selected_marker_scope_report.json
duplicate_non_selected_marker_conflict_report.json
selected_marker_prose_mention_report.json
selected_marker_prose_line_start_report.json
following_line_value_leak_report.json
legacy_manifest_regression_report.json
static_manifest_integrity_report.json
helper_request_audit.json
aggregate_metrics.json
decision.json
summary.json
report.md
```

Expected positive route:

```text
decision = selected_marker_occurrence_count_rejection_prototype_positive
next = 143W_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_SCALE_CONFIRM
```

Clean negative routes include duplicate conflict still not rejected, overbroad duplicate rejection, candidate-line parser false positive, following-line value leak, single-marker binding regression, and helper integrity failure.

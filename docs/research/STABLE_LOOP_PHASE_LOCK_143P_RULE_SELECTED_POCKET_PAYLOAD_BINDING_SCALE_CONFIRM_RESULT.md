# STABLE_LOOP_PHASE_LOCK_143P_RULE_SELECTED_POCKET_PAYLOAD_BINDING_SCALE_CONFIRM Result

143P records the scale confirm or clean edge-case negative for the 143K selected-pocket binding primitive.

Boundary: constrained helper/backend evidence only. It tests prompt-visible selected-pocket binding only, not rule metadata reasoning, not open-ended arbitration, not GPT-like/open-domain/broad assistant capability, not production/public API/deployment/safety readiness, and not architecture superiority.

## Expected Outcomes

Positive if all scale and edge gates pass:

```text
decision = rule_selected_pocket_payload_binding_scale_confirmed
verdict = INSTNCT_RULE_SELECTED_POCKET_BINDING_SCALE_CONFIRMED
next = 143Z_RULE_SELECTED_POCKET_BINDING_NEXT_DECISION_PLAN
```

Clean negative if duplicate selected marker conflicts are not rejected:

```text
decision = duplicate_selected_marker_conflict_not_rejected
verdict = INSTNCT_RULE_SELECTED_POCKET_BINDING_SCALE_EDGE_GAP
next = 143R_DUPLICATE_SELECTED_MARKER_CONFLICT_ANALYSIS
```

## Required Evidence

143P writes:

```text
duplicate_selected_marker_conflict_report.json
malformed_winner_label_report.json
duplicate_same_winner_label_report.json
selected_marker_missing_report.json
selected_marker_value_missing_report.json
legacy_manifest_regression_report.json
shared_helper_no_change_audit.json
helper_request_audit.json
static_manifest_integrity_report.json
pocket_marker_order_permutation_report.json
prompt_scanner_report.json
aggregate_metrics.json
decision.json
summary.json
report.md
```

Positive gates:

```text
winner_label_parse_accuracy >= 0.95
selected_pocket_to_marker_binding_accuracy >= 0.95
pocket_marker_order_permutation_accuracy >= 0.95
all_6_marker_orders_covered = true
all_3_winner_labels_covered = true
all_winner_positions_covered = true
first_prompt_marker_shortcut_rate = 0.0
duplicate_selected_marker_conflict_rejection_rate >= 0.95
duplicate_selected_marker_first_value_rate = 0.0
duplicate_selected_marker_last_value_rate = 0.0
legacy_manifest_regression_passed = true
shared_helper_no_change_since_143k = true
```

If the duplicate conflict report shows first/last selected-marker value selection instead of fallback, this is a helper edge-policy gap, not a broad architecture failure.


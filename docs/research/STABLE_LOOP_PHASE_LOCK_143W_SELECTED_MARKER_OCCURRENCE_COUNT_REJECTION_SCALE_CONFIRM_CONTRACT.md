# STABLE_LOOP_PHASE_LOCK_143W_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_SCALE_CONFIRM Contract

143W is the scale-confirm milestone after the 143V selected-marker occurrence-count rejection prototype. It does not repair the helper and must not modify `scripts/probes/shared_raw_generation_helper.py`; it uses the existing manifest-gated decoder:

```text
deterministic_pocket_gated_rule_selected_pocket_binding_decoder
```

Boundary: constrained helper/backend evidence only. This is prompt-visible selected-pocket binding only, not rule metadata reasoning, not open-ended arbitration, not GPT-like/open-domain/broad assistant capability, not production/public API/deployment/safety readiness, and not architecture superiority.

## Required Upstream

143W requires 143V exactly:

```text
decision = selected_marker_occurrence_count_rejection_prototype_positive
next = 143W_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_SCALE_CONFIRM
single_selected_marker_binding_accuracy = 1.0
duplicate_selected_marker_conflict_rejection_rate = 1.0
duplicate_selected_marker_same_value_rejection_rate = 1.0
duplicate_non_selected_marker_conflict_binding_accuracy = 1.0
selected_marker_candidate_line_parse_accuracy = 1.0
selected_marker_prose_line_false_positive_rate = 0.0
following_line_value_leak_rate = 0.0
legacy_manifest_regression_passed = true
deterministic_replay_passed = true
```

## Scale Shape

Default scale:

```text
seeds = 4901,4902,4903,4904
families = 8
groups_per_family = 24
group_size = 4
main_eval_rows = 3072
max_new_tokens = 96
heartbeat_sec = 20
```

Required families:

```text
SINGLE_SELECTED_MARKER_POSITIVE
WINNER_LABEL_POSITION_INVARIANCE
POCKET_MARKER_ORDER_PERMUTATION
DUPLICATE_SELECTED_MARKER_CONFLICT
DUPLICATE_SELECTED_MARKER_SAME_VALUE
DUPLICATE_NON_SELECTED_MARKER_SCOPE
SELECTED_MARKER_PROSE_AND_LINE_PARSER
FOLLOWING_LINE_VALUE_LEAK_TRAP
```

Metric denominators must stay explicit: `positive_binding_subset_writeback_rate` is computed only over rows where writeback is expected. Expected-fallback edge rows are scored by their own fallback-policy rates and must not lower the positive binding denominator.

## Required Evidence

143W must write:

```text
shared_helper_no_change_audit.json
helper_repair_semantics_audit.json
selected_marker_occurrence_count_report.json
selected_marker_candidate_line_parser_report.json
duplicate_selected_marker_conflict_report.json
duplicate_selected_marker_same_value_report.json
duplicate_non_selected_marker_scope_report.json
duplicate_non_selected_marker_conflict_report.json
selected_marker_prose_mention_report.json
selected_marker_prose_line_start_report.json
selected_marker_prose_plus_one_valid_line_report.json
selected_marker_invalid_value_report.json
selected_marker_multi_value_same_line_report.json
following_line_value_leak_report.json
per_seed_gate_report.json
per_family_gate_report.json
legacy_manifest_regression_report.json
static_manifest_integrity_report.json
helper_request_audit.json
aggregate_metrics.json
decision.json
summary.json
report.md
```

`shared_helper_no_change_audit.json` must compare the current helper hash with the 143V post-repair helper hash and prove that 143W did not modify the helper.

`helper_repair_semantics_audit.json` must verify the repaired semantics remain present: `_instnct_select_rule_selected_pocket_value`, `candidate_line_re`, `prompt.splitlines()`, no `prompt.find(selected_marker)`, same-line extraction, fallback when selected marker candidate-line count is not one, fallback on selected-marker value missing, and the old decoder path still present.

## Positive Route

If all gates pass:

```text
decision = selected_marker_occurrence_count_rejection_scale_confirmed
verdict = INSTNCT_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_SCALE_CONFIRMED
next = 143Z_RULE_SELECTED_POCKET_BINDING_NEXT_DECISION_PLAN
```

Clean negative routes include duplicate conflict scale failure, overbroad duplicate rejection, candidate-line parser scale failure, following-line value leak, single-marker binding regression, and helper integrity failure.

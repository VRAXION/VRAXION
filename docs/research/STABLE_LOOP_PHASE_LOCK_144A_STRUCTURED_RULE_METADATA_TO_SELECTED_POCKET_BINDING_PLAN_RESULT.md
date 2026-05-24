# STABLE_LOOP_PHASE_LOCK_144A_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_PLAN Result

144A records the expected implementation-ready plan for the first structured rule metadata to selected-pocket binding prototype.

Boundary: constrained helper/backend evidence only, structured rule metadata to selected-pocket binding only, not natural-language rule reasoning, not open-ended arbitration, not GPT-like/open-domain/broad assistant capability, not production/public API/deployment/safety readiness, and not architecture superiority.

## Expected Route

```text
decision = structured_rule_metadata_to_selected_pocket_binding_prototype_plan_recommended
selected_option = canonical_structured_rule_metadata_parser_plus_existing_selected_pocket_binding
next = 144B_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_PROTOTYPE
```

144A is planning-only and artifact-only. It does not call helper generation, import `shared_raw_generation_helper.py`, train, mutate checkpoints, modify helper/backend/request-key/runtime/product surfaces, or change broad capability claims.

## Implementation-Ready Target

144A must make `target_144b_milestone_plan.json` complete enough to implement 144B without another planning milestone.

144B decoder:

```text
deterministic_pocket_gated_structured_rule_metadata_binding_decoder
```

Existing selected-pocket binding decoder behavior must remain unchanged:

```text
deterministic_pocket_gated_rule_selected_pocket_binding_decoder
```

144B should parse canonical structured rule metadata, derive selected pocket identity, then reuse the confirmed selected-pocket binding layer for static marker lookup and same-line value extraction.

## Required 144B Coverage

144B subsets:

```text
EXPLICIT_WINNER_LABEL_BASELINE
RULE_METADATA_DERIVED_NO_WINNER_LABEL
QUORUM_RULE_DERIVED
RECENCY_RULE_DERIVED
TIE_BREAK_RULE_DERIVED
HIERARCHY_RULE_DERIVED
SAME_VALUES_DIFFERENT_RULE
SAME_RULE_DIFFERENT_VALUES
SAME_TEMPLATE_OPPOSITE_RULE_WINNER
RULE_METADATA_CORRUPTION_CONTROL
MISSING_RULE_METADATA_CONTROL
AMBIGUOUS_RULE_METADATA_CONTROL
LEGACY_SELECTED_POCKET_BINDING_REGRESSION_CONTROL
```

144B metrics:

```text
rule_metadata_parse_accuracy
derived_selected_pocket_accuracy
selected_pocket_to_marker_binding_accuracy
same_line_value_extraction_accuracy
end_to_end_answer_accuracy
rule_derived_no_winner_label_accuracy
explicit_winner_baseline_accuracy
rule_metadata_ablation_accuracy
corrupt_rule_metadata_rejection_rate
missing_rule_metadata_fallback_rate
ambiguous_rule_metadata_rejection_rate
helper_request_forbidden_metadata_count
per_row_manifest_switch_rate
per_row_payload_marker_switch_rate
legacy_selected_pocket_binding_regression_passed
deterministic_replay_passed
```

Clean negatives route to parse failure, derived selected-pocket failure, oracle shortcut, ambiguity, hierarchy policy, selected-pocket binding regression, or helper integrity analysis.

A future 144B positive would prove only constrained structured rule metadata -> selected pocket binding. It would not prove natural-language rule reasoning, open-ended arbitration, GPT-like capability, production readiness, or architecture superiority.

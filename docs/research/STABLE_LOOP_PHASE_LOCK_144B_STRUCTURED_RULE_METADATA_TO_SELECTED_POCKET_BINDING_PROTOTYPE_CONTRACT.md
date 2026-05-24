# STABLE_LOOP_PHASE_LOCK_144B_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_PROTOTYPE CONTRACT

144B is the first executable prototype after the 144A implementation-ready plan.

The milestone adds one helper primitive behind a new manifest-gated decoder:

```text
deterministic_pocket_gated_structured_rule_metadata_binding_decoder
```

The expected positive route is:

```text
decision = structured_rule_metadata_to_selected_pocket_binding_prototype_positive
verdict = INSTNCT_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_PROTOTYPE_POSITIVE
next = 144H_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_SCALE_CONFIRM
```

Boundary: 144B is constrained helper/backend evidence only and structured rule metadata to selected-pocket binding only; it is not natural-language rule reasoning, not open-ended arbitration, not GPT-like/open-domain/broad assistant capability, not production/public API/deployment/safety readiness, and not architecture superiority.

## Helper Contract

144B may modify `scripts/probes/shared_raw_generation_helper.py`, but only by adding the new manifest-gated structured rule metadata decoder path. The existing selected-pocket binding decoder must remain unchanged:

```text
deterministic_pocket_gated_rule_selected_pocket_binding_decoder
```

The helper request keys remain exactly:

```text
prompt
checkpoint_path
checkpoint_hash
seed
max_new_tokens
generation_config
```

The new decoder parses canonical structured `key=value` metadata, derives a selected pocket id, maps that pocket through the static pocket marker map, applies the 143V/143W selected marker candidate-line policy, and emits `ANSWER=E<value>` only when every layer succeeds.

Failure must fallback when metadata parsing fails, selected pocket derivation fails, the open gate is absent, the selected marker is absent, the selected marker appears more than once as a valid candidate line, or the selected marker line lacks a valid same-line value.

The new path must expose trace fields for the new decoder only:

```text
parsed_rule_type
parsed_rule_fields
parse_success
derived_selected_pocket_id
binding_marker
extracted_value
generated_answer
failure_reason
```

## Prototype Contract

The runner must use the 144A artifacts as upstream source of truth and verify:

```text
decision = structured_rule_metadata_to_selected_pocket_binding_prototype_plan_recommended
selected_option = canonical_structured_rule_metadata_parser_plus_existing_selected_pocket_binding
next = 144B_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_PROTOTYPE
```

Rule-derived rows must not include `winner=pocket_*`, `selected_pocket_id`, final/winner/answer/gold/target/resolved/expected output shortcuts, per-row selected pocket request metadata, per-row manifest switching, narrowed payload marker lists, or post-generation repair.

The prototype must score parser, derivation, binding, same-line extraction, and end-to-end answer behavior separately. Explicit winner-label rows are allowed only as a baseline and must use the existing selected-pocket decoder, not the new structured metadata decoder.

Positive gates:

```text
rule_metadata_parse_accuracy >= 0.90
derived_selected_pocket_accuracy >= 0.90
selected_pocket_to_marker_binding_accuracy >= 0.95
same_line_value_extraction_accuracy >= 0.95
end_to_end_answer_accuracy >= 0.90
rule_derived_no_winner_label_accuracy >= 0.90
explicit_winner_baseline_accuracy >= 0.95
rule_metadata_ablation_accuracy <= 0.15
corrupt_rule_metadata_rejection_rate >= 0.90
missing_rule_metadata_fallback_rate >= 0.90
ambiguous_rule_metadata_rejection_rate >= 0.90
helper_request_forbidden_metadata_count = 0
per_row_manifest_switch_rate = 0.0
per_row_payload_marker_switch_rate = 0.0
legacy_selected_pocket_binding_regression_passed = true
deterministic_replay_passed = true
```

Clean negative routes:

```text
structured_rule_metadata_parse_failure -> 144C_STRUCTURED_RULE_METADATA_PARSE_FAILURE_ANALYSIS
derived_selected_pocket_failure -> 144D_DERIVED_SELECTED_POCKET_FAILURE_ANALYSIS
rule_metadata_oracle_shortcut_detected -> 144E_RULE_METADATA_ORACLE_SHORTCUT_ANALYSIS
rule_metadata_ambiguity_not_rejected -> 144F_RULE_METADATA_AMBIGUITY_ANALYSIS
hierarchy_priority_policy_failure -> 144G_HIERARCHY_RULE_POLICY_ANALYSIS
selected_pocket_binding_regression -> 143L_WINNER_LABEL_BINDING_FAILURE_ANALYSIS
helper_integrity_failure -> 135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL
```

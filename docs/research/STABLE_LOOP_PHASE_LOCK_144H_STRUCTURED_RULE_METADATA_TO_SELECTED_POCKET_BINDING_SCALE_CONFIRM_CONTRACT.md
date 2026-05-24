# STABLE_LOOP_PHASE_LOCK_144H_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_SCALE_CONFIRM CONTRACT

144H is the scale confirm after the positive 144B structured rule metadata prototype. It does not repair or extend the helper. It uses the existing manifest-gated decoder:

```text
deterministic_pocket_gated_structured_rule_metadata_binding_decoder
```

Expected positive route:

```text
decision = structured_rule_metadata_to_selected_pocket_binding_scale_confirmed
verdict = INSTNCT_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_SCALE_CONFIRMED
next = 144Z_STRUCTURED_RULE_METADATA_BINDING_NEXT_DECISION_PLAN
```

Boundary: 144H is constrained helper/backend evidence only and structured rule metadata to selected-pocket binding only; it is not natural-language rule reasoning, not open-ended arbitration, not GPT-like/open-domain/broad assistant capability, not production/public API/deployment/safety readiness, and not architecture superiority.

## Requirements

144H must verify the 144B positive result exactly before running scale confirmation. The upstream requirements include parse, derivation, selected-pocket binding, same-line extraction, and end-to-end answer metrics at `1.0`, rule metadata ablation at `0.0`, corrupt/missing/ambiguous rejection at `1.0`, legacy selected-pocket binding regression passing, and deterministic replay passing.

The runner must not modify `scripts/probes/shared_raw_generation_helper.py`. It must audit the current helper hash against the upstream 144B helper hash, confirm that the structured rule metadata decoder remains manifest-gated, and confirm that the old selected-pocket decoder remains present.

Scale coverage must include quorum, recency, tie-break, hierarchy, wrong-family extra key rejection, clear quorum winner ignoring irrelevant tie-break order, hierarchy stale rejection priority, same-template opposite rule winner, same values different rule, same rule different values, corruption, missing metadata, ambiguous metadata, explicit winner baseline, and 143W-style selected-pocket binding regression.

Rule-derived prompts must forbid `winner=pocket_*`, `selected_pocket_id`, final/winner/answer/gold/target/resolved/expected output shortcuts, per-row selected-pocket metadata, per-row manifest switching, narrowed payload marker lists, and post-generation repair.

Positive gates:

```text
rule_metadata_parse_accuracy >= 0.98
derived_selected_pocket_accuracy >= 0.98
selected_pocket_to_marker_binding_accuracy >= 0.98
same_line_value_extraction_accuracy >= 0.98
end_to_end_answer_accuracy >= 0.98
rule_derived_no_winner_label_accuracy >= 0.98
explicit_winner_baseline_accuracy >= 0.98
rule_metadata_ablation_accuracy <= 0.05
corrupt_rule_metadata_rejection_rate >= 0.98
missing_rule_metadata_fallback_rate >= 0.98
ambiguous_rule_metadata_rejection_rate >= 0.98
quorum_tie_break_accuracy >= 0.98
quorum_clear_winner_ignores_tie_break_accuracy >= 0.98
wrong_family_extra_key_rejection_rate >= 0.98
hierarchy_policy_accuracy >= 0.98
same_template_opposite_rule_winner_accuracy >= 0.98
legacy_selected_pocket_binding_regression_passed = true
legacy_143w_binding_regression_passed = true
helper_request_forbidden_metadata_count = 0
per_row_manifest_switch_rate = 0.0
per_row_payload_marker_switch_rate = 0.0
shared_helper_no_change_since_144b = true
deterministic_replay_passed = true
```

Clean negative routes:

```text
structured_rule_metadata_parse_scale_failure -> 144C_STRUCTURED_RULE_METADATA_PARSE_FAILURE_ANALYSIS
derived_selected_pocket_scale_failure -> 144D_DERIVED_SELECTED_POCKET_FAILURE_ANALYSIS
rule_metadata_oracle_shortcut_detected -> 144E_RULE_METADATA_ORACLE_SHORTCUT_ANALYSIS
rule_metadata_ambiguity_not_rejected -> 144F_RULE_METADATA_AMBIGUITY_ANALYSIS
hierarchy_priority_policy_scale_failure -> 144G_HIERARCHY_RULE_POLICY_ANALYSIS
wrong_family_extra_key_not_rejected -> 144I_RULE_FAMILY_EXACT_KEY_POLICY_ANALYSIS
quorum_tie_break_policy_failure -> 144J_QUORUM_TIE_BREAK_POLICY_ANALYSIS
quorum_irrelevant_tie_break_dominates -> 144K_QUORUM_CLEAR_WINNER_POLICY_ANALYSIS
selected_pocket_binding_regression -> 143L_WINNER_LABEL_BINDING_FAILURE_ANALYSIS
helper_integrity_failure -> 135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL
```

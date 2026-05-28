# D38 RULE_KNOWN_ROUTER_LAYER_PROTOTYPE CONTRACT

D38 tests known-rule formula routing only on controlled symbolic pocket tasks.

Non-claims: this is not hidden-rule Raven solving, not natural-language reasoning, not DNA/genome v2, and not architecture superiority.

Arms: MONOLITHIC_FORMULA_BASELINE, ORACLE_GATED_RULE_FORMULA_UPPER_BOUND, MUTABLE_LEARNED_ROUTER_GATE, SHUFFLED_GATE_CONTROL, NO_FAMILY_INPUT_CONTROL, EXPLICIT_TARGET_STATE_UPPER_BOUND, TARGET_GIVEN_ORACLE_CONTROL.

Dataset invariants required:
- duplicate_target_pocket_rate = 0.0
- missing_target_pocket_rate = 0.0
- expected_selected_points_to_target_rate = 1.0

OOD invariance required:
- known_rule_oracle_test_accuracy = 1.0
- known_rule_oracle_ood_accuracy = 1.0
- ood_label_rule_changed = false

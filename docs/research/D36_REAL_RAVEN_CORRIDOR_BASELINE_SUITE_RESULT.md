# D36 Real Raven Corridor Baseline Suite Result

## Upstream D35 audit
- D34 was scaffold/synthetic.
- D35 was a minimal real probe.
- D35 duplicate-target risk existed.
- D35 OOD shift risk (label-rule change risk) existed.
- D36 fixes these via invariant-enforced generator and OOD permutations that preserve label logic.

## Dataset invariant requirements
- duplicate_target_pocket_rate must be 0.0
- missing_target_pocket_rate must be 0.0
- expected_selected_points_to_target_rate must be 1.0
- ood_label_rule_changed must be false

Measured invariant outcomes are only valid after executing the D36 runner and D36 checker artifacts for the target run path.

## Boundary note
- RULE_HIDDEN_ROUTING uses precomputed rule-hypothesis features without family label.
- It is not raw visual Raven reasoning.

## Non-claims
- no solved claim
- no architecture superiority claim
- no natural-language reasoning claim

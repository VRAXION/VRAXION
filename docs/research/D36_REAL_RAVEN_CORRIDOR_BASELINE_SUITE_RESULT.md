# D36 Real Raven Corridor Baseline Suite Result

## Upstream D35 audit
- D34 was scaffold/synthetic.
- D35 was a minimal real probe.
- D35 duplicate-target risk existed.
- D35 OOD shift risk (label-rule change risk) existed.
- D36 fixes these via invariant-enforced generator and OOD permutations that preserve label logic.

## Dataset invariant outcome
- duplicate_target_pocket_rate = 0.0
- missing_target_pocket_rate = 0.0
- expected_selected_points_to_target_rate = 1.0
- ood_label_rule_changed = false

## Non-claims
- no solved claim
- no architecture superiority claim
- no natural-language reasoning claim

# STABLE_LOOP_PHASE_LOCK_143A_INSTNCT_POCKET_GATED_MULTI_POCKET_ARBITRATION_PROBE Contract

143A is an executable helper-only probe after 142Z. It tests constrained helper/backend multi-pocket arbitration final selection: three pocket-carried candidates are present, an arbitration rule selects one final value, and shortcut arms must fail.

This is constrained helper/backend evidence only. It is not open-ended reasoning, not general composition, not GPT-like readiness, not open-domain reasoning, not broad assistant capability, not production/public API/deployment/safety readiness, and not architecture superiority.

## Boundaries

- No training.
- No source checkpoint mutation.
- No helper/backend/request-key changes.
- No runtime/release/product/deploy/public API changes.
- No root `LICENSE` change.
- Runner may call `shared_raw_generation_helper.raw_generate` only through the shared helper.
- Checker must not call `raw_generate`.
- Helper requests may include only `prompt`, `checkpoint_path`, `checkpoint_hash`, `seed`, `max_new_tokens`, and `generation_config`.
- Expected/scorer/oracle metadata must not enter helper requests.

## Required Upstreams

Require 142Z:

```text
decision = multi_pocket_arbitration_probe_recommended
next = 143A_INSTNCT_POCKET_GATED_MULTI_POCKET_ARBITRATION_PROBE
```

Require hardened 142F scale evidence, including helper request audit, deterministic replay, per-seed/per-family/per-winner gates, and clean shortcut rates.

## Required Families

```text
THREE_POCKET_RULE_SELECT
QUORUM_TWO_OF_THREE
RECENCY_OVERRIDE
TIE_BREAK_ARBITRATION
STALE_POCKET_LOSES
SAME_TEMPLATE_ARBITRATION_INVERSION
RULE_HIERARCHY_CONFLICT
```

## Adversarial Controls

143A must include first-pocket, last-pocket, default-pocket, stale-pocket, visible-value, noisy-distractor, closed-pocket, quorum, recency, tie-break, and prefix controls.

It must also include these hard controls:

```text
RESOLVED_FINAL_MARKER_ECHO_CONTROL
POCKET_LABEL_PERMUTATION_CONTROL
RULE_HIERARCHY_CONFLICT_CONTROL
SAME_VALUES_DIFFERENT_RULE_CONTROL
SAME_RULE_DIFFERENT_VALUES_CONTROL
```

The resolved-final-marker echo control corrupts the resolved final marker while leaving the pocket/rule-selected value correct. If the helper path merely echoes the resolved-final-marker, the control passes and 143A is invalid. Positive requires:

```text
resolved_final_marker_echo_rate = 0.0
resolved_final_marker_echo_control_failed = true
```

## Positive Gates

```text
main_final_answer_accuracy >= 0.70
multi_pocket_arbitration_accuracy >= 0.70
quorum_rule_accuracy >= 0.65
recency_rule_accuracy >= 0.65
tie_break_accuracy >= 0.65
rule_hierarchy_conflict_accuracy >= 0.65
priority_inversion_accuracy >= 0.70
same_template_opposite_winner_accuracy >= 0.70
pocket_label_permutation_accuracy >= 0.70
same_values_different_rule_accuracy >= 0.70
same_rule_different_values_accuracy >= 0.70
main_pocket_writeback_rate >= 0.80
main_contrast_group_accuracy >= 0.70
ablation_final_answer_accuracy <= 0.15
pocket_ablation_delta_final_answer_accuracy >= 0.50
default_pocket_shortcut_rate = 0.0
first_pocket_shortcut_rate = 0.0
last_pocket_shortcut_rate = 0.0
stale_pocket_shortcut_rate = 0.0
visible_bypass_violation_rate = 0.0
noisy_distractor_violation_rate = 0.0
direct_pocket_value_marker_rate = 0.0
deterministic_replay_passed = true
```

## Positive Decision

```text
decision = instnct_pocket_gated_multi_pocket_arbitration_probe_positive
verdict = INSTNCT_POCKET_GATED_MULTI_POCKET_ARBITRATION_POSITIVE
next = 143F_INSTNCT_POCKET_GATED_MULTI_POCKET_ARBITRATION_SCALE_CONFIRM
```

## Clean Negatives

```text
resolved_final_marker_echo_detected -> 143B_RESOLVED_FINAL_MARKER_ECHO_ANALYSIS
default_pocket_shortcut_detected -> 143C_DEFAULT_POCKET_SHORTCUT_ANALYSIS
first_or_last_pocket_shortcut_detected -> 143D_POSITIONAL_POCKET_SHORTCUT_ANALYSIS
multi_pocket_arbitration_failure -> 143E_MULTI_POCKET_ARBITRATION_FAILURE_ANALYSIS
quorum_recency_tie_break_failure -> 143G_QUORUM_RECENCY_TIE_BREAK_FAILURE_ANALYSIS
rule_hierarchy_conflict_failure -> 143H_RULE_HIERARCHY_CONFLICT_ANALYSIS
pocket_ablation_not_decision_critical -> 141D_POCKET_CAUSALITY_FAILURE_ANALYSIS
helper_integrity_failure -> 135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL
```

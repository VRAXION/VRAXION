# STABLE_LOOP_PHASE_LOCK_143P_RULE_SELECTED_POCKET_PAYLOAD_BINDING_SCALE_CONFIRM Contract

143P is an executable helper-only scale confirm after the positive 143K prototype.

Boundary: constrained helper/backend evidence only. It tests prompt-visible selected-pocket binding only, not rule metadata reasoning, not open-ended arbitration, not GPT-like/open-domain/broad assistant capability, not production/public API/deployment/safety readiness, and not architecture superiority.

## Upstream

Require 143K:

```text
decision = rule_selected_pocket_payload_binding_prototype_positive
next = 143P_RULE_SELECTED_POCKET_PAYLOAD_BINDING_SCALE_CONFIRM
winner_label_parse_accuracy = 1.0
selected_pocket_to_marker_binding_accuracy = 1.0
pocket_marker_order_permutation_accuracy = 1.0
first_prompt_marker_shortcut_rate = 0.0
missing_winner_label_fallback_rate = 1.0
ambiguous_winner_label_rejection_rate = 1.0
closed_pocket_ablation_accuracy = 0.0
legacy_manifest_regression_passed = true
deterministic_replay_passed = true
```

## Scope

143P must not modify `shared_raw_generation_helper.py`. It uses the existing 143K manifest-gated decoder:

```text
deterministic_pocket_gated_rule_selected_pocket_binding_decoder
```

The runner may call raw generation only through the shared helper. The checker must not call `raw_generate`.

## Scale And Edge Policies

Defaults:

```text
seeds = 4701,4702,4703,4704
families = 8
groups_per_family = 24
group_size = 4
main_eval_rows = 3072
max_new_tokens = 96
```

Required families:

```text
EXPLICIT_WINNER_LABEL_BINDING
WINNER_LABEL_POSITION_INVARIANCE
POCKET_MARKER_ORDER_PERMUTATION
SAME_VALUES_DIFFERENT_WINNER
SAME_WINNER_DIFFERENT_VALUES
MISSING_AMBIGUOUS_WINNER_REJECTION
DUPLICATE_SELECTED_MARKER_CONFLICT
MALFORMED_WINNER_AND_MARKER_ABSENCE
```

Edge policy:

```text
missing winner label -> fallback
ambiguous conflicting winner labels -> fallback
duplicate same winner labels -> fallback
malformed winner labels -> fallback
selected marker missing -> fallback
selected marker value missing -> fallback
duplicate selected marker with conflicting values -> fallback
```

If duplicate selected marker conflict is not rejected, 143P must route cleanly:

```text
decision = duplicate_selected_marker_conflict_not_rejected
next = 143R_DUPLICATE_SELECTED_MARKER_CONFLICT_ANALYSIS
```

## Positive Route

If all gates pass:

```text
decision = rule_selected_pocket_payload_binding_scale_confirmed
verdict = INSTNCT_RULE_SELECTED_POCKET_BINDING_SCALE_CONFIRMED
next = 143Z_RULE_SELECTED_POCKET_BINDING_NEXT_DECISION_PLAN
```

Required gates include high scale accuracy, no positional shortcut, no helper request metadata oracle, static manifests, legacy regression preservation, deterministic replay, and all edge-case rejection policies.

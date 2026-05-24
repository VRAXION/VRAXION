# STABLE_LOOP_PHASE_LOCK_143J_RULE_SELECTED_POCKET_PAYLOAD_BINDING_HELPER_PRIMITIVE_PLAN Contract

143J is a planning-only milestone after 143I. It designs the next helper primitive but does not implement helper/backend behavior.

Boundary: constrained helper/backend planning-only evidence. It does not implement helper/backend behavior, call helper generation, train, mutate checkpoints, change helper request keys, and is not architecture superiority. It is not rule metadata reasoning, not open-ended reasoning, not open-domain or broad assistant capability, and not production/public API/deployment/safety readiness.

## Required Upstream

Require 143I:

```text
decision = no_resolved_final_marker_bridge_failure_analysis_complete
root_cause_id = helper_payload_marker_selection_lacks_rule_selected_pocket_binding
next = 143J_RULE_SELECTED_POCKET_PAYLOAD_BINDING_HELPER_PRIMITIVE_PLAN
supported_by_143f_clean_dependency = true
supported_by_helper_source_audit = true
hidden_marker_leak_rejected = true
per_row_manifest_oracle_rejected = true
shortcut_failure_rejected = true
abc_static_first_marker_behavior_confirmed = true
```

## Decision

143J compares and scores:

```text
prompt_level_explicit_winner_label_parser_plus_static_marker_map
static_pocket_marker_map_plus_prompt_selected_pocket_binding
rule_metadata_parser
keep_resolved_final_marker_only
```

Expected decision:

```text
decision = rule_selected_pocket_payload_binding_primitive_plan_recommended
selected_option = prompt_level_explicit_winner_label_parser_plus_static_marker_map
next = 143K_RULE_SELECTED_POCKET_PAYLOAD_BINDING_PROTOTYPE_PROBE
```

143K positive would prove prompt-visible selected-pocket binding only. It would not prove rule metadata reasoning. It would not prove open-ended arbitration.

## Static Map Constraint

The manifest may define only a stable map:

```text
pocket_a -> "pocket A candidate:"
pocket_b -> "pocket B candidate:"
pocket_c -> "pocket C candidate:"
```

It must not carry per-row selected pocket identity.

## Target 143K Requirements

143K must parse only prompt-visible `winner=pocket_a|pocket_b|pocket_c`, bind it to the static marker map, extract that value, and emit `ANSWER=E<value>`.

143K must keep helper request keys unchanged:

```text
prompt
checkpoint_path
checkpoint_hash
seed
max_new_tokens
generation_config
```

143K must forbid per-row selected pocket metadata, per-row manifest switching, payload marker narrowing to the correct pocket, hidden final/winner-value/gold/answer markers, resolved final markers in main no-resolved rows, post-generation repair, and broad architecture claims.

Required 143K controls include winner-label wrong/missing/ambiguous controls, label position invariance, pocket marker order permutation, same-values different-winner, same-winner different-values, first-prompt-marker shortcut, positional pocket shortcuts, visible/noisy controls, closed-pocket ablation, and static manifest integrity.

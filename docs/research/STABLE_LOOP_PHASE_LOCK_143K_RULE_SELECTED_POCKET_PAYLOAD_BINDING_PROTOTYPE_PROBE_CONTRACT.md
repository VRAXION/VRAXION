# STABLE_LOOP_PHASE_LOCK_143K_RULE_SELECTED_POCKET_PAYLOAD_BINDING_PROTOTYPE_PROBE Contract

143K is an executable helper-only prototype after 143J. It adds one manifest-gated shared helper primitive for prompt-visible selected-pocket binding:

```text
winner=pocket_b
-> static pocket marker map
-> "pocket B candidate:"
-> value extraction
-> ANSWER=E<value>
```

Boundary: constrained helper/backend evidence only. 143K positive proves prompt-visible selected-pocket binding only. It is not rule metadata reasoning, not open-ended arbitration, not GPT-like/open-domain/broad assistant capability, not production/public API/deployment/safety readiness, and not architecture superiority.

## Required Upstream

Require 143J:

```text
decision = rule_selected_pocket_payload_binding_primitive_plan_recommended
selected_option = prompt_level_explicit_winner_label_parser_plus_static_marker_map
next = 143K_RULE_SELECTED_POCKET_PAYLOAD_BINDING_PROTOTYPE_PROBE
```

## Helper Primitive

Modify `scripts/probes/shared_raw_generation_helper.py` only by adding a new manifest-gated INSTNCT decoder:

```text
deterministic_pocket_gated_rule_selected_pocket_binding_decoder
```

The new decoder must:

```text
open pocket gate present
parse exactly one selector:
  winner=pocket_a
  winner=pocket_b
  winner=pocket_c
map selector through static manifest map:
  pocket_a -> "pocket A candidate:"
  pocket_b -> "pocket B candidate:"
  pocket_c -> "pocket C candidate:"
extract the selected marker value
emit ANSWER=E<value>
```

Fallback if the selector is missing, ambiguous, conflicting, invalid, or if the selected marker has no value.

Old manifests without the new decoder must preserve old first-marker/fallback behavior exactly. Helper request keys remain unchanged:

```text
prompt
checkpoint_path
checkpoint_hash
seed
max_new_tokens
generation_config
```

## Required Controls

143K must include:

```text
WINNER_LABEL_BINDING_CONTROL
WINNER_LABEL_CORRUPTION_ORACLE_CONTROL
WINNER_LABEL_MISSING_CONTROL
WINNER_LABEL_AMBIGUOUS_CONTROL
WINNER_LABEL_POSITION_INVARIANCE_CONTROL
POCKET_MARKER_ORDER_PERMUTATION_CONTROL
SAME_VALUES_DIFFERENT_WINNER_CONTROL
SAME_WINNER_DIFFERENT_VALUES_CONTROL
FIRST_PROMPT_MARKER_SHORTCUT_CONTROL
CLOSED_POCKET_ABLATION_CONTROL
STATIC_MANIFEST_INTEGRITY_CONTROL
LEGACY_MANIFEST_REGRESSION_CONTROL
```

Main rows must not contain resolved final markers or hidden final/winner-value/gold/answer marker equivalents. The prompt scanner is case-insensitive regex. Allowed selector forms are only:

```text
winner=pocket_a
winner=pocket_b
winner=pocket_c
```

Forbidden marker/value equivalents include:

```text
winner[-_ ]?value
selected[-_ ]?value
final[-_ ]?winner
answer[-_ ]?pocket
answer[-_ ]?value
gold[-_ ]?pocket
gold[-_ ]?value
target[-_ ]?value
resolved[-_ ]?output
expected[-_ ]?output
arbitrated[-_ ]?final
selected[-_ ]?final
ANSWER\s*=
TARGET\s*=
GOLD\s*=
EXPECTED\s*=
```

## Positive Decision

If gates pass:

```text
decision = rule_selected_pocket_payload_binding_prototype_positive
verdict = INSTNCT_RULE_SELECTED_POCKET_PAYLOAD_BINDING_PROTOTYPE_POSITIVE
next = 143P_RULE_SELECTED_POCKET_PAYLOAD_BINDING_SCALE_CONFIRM
```

Clean negative routes:

```text
winner_label_binding_failure -> 143L_WINNER_LABEL_BINDING_FAILURE_ANALYSIS
oracle_manifest_shortcut_detected -> 143M_ORACLE_MANIFEST_SHORTCUT_ANALYSIS
positional_pocket_shortcut_detected -> 143D_POSITIONAL_POCKET_SHORTCUT_ANALYSIS
ambiguous_winner_not_rejected -> 143N_AMBIGUOUS_WINNER_LABEL_ANALYSIS
missing_winner_not_rejected -> 143O_MISSING_WINNER_LABEL_ANALYSIS
pocket_ablation_not_decision_critical -> 141D_POCKET_CAUSALITY_FAILURE_ANALYSIS
helper_integrity_failure -> 135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL
```


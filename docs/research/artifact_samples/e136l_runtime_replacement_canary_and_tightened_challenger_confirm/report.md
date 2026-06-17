# E136L Runtime Replacement Canary And Tightened Challenger Confirm

```text
decision = e136l_runtime_replacement_canary_and_tightened_challenger_confirmed
next     = E136M_RUNTIME_REPLACEMENT_APPLY_OR_ABSTRACT_LINEAGE_SPLIT
```

E136L tests the E136K apply plan in rollback-safe canary form. The direct
canary group is evaluated as if the legacy trigger was removed and the selected
variant was active in the canary runtime. Tightened challenger and abstract
lineage rows are replayed but not applied.

## Result

```text
operator_count = 34
direct_canary_tested_count = 16
direct_canary_pass_count = 16
old_operator_removed_in_canary_count = 16
runtime_replacement_canary_allowed_count = 16
production_runtime_apply_count = 0
destructive_apply_count = 0

challenger_ood_tested_count = 11
challenger_hold_count = 11
challenger_runtime_apply_allowed_count = 0
abstract_lineage_hold_count = 7

rollback_manifest_count = 16
rollback_trigger_count = 0

current_activation_total = 188597925
selected_activation_total = 166354720
shadow_pruned_activation_total = 22243205
shadow_prune_ratio = 0.11794

direct_canary_legacy_activation_total = 60362384
direct_canary_selected_activation_total = 56912127
direct_canary_removed_activation_total = 3450257
direct_canary_removed_activation_ratio = 0.057159

sample_rows_processed = 8345
sample_direct_legacy_activation_total = 10056
sample_direct_selected_activation_total = 9025
sample_direct_removed_activation_total = 1031

strict_recall_miss_total = 0
wrong_scope_proxy_total = 0
hard_negative_total = 0
unsupported_answer_total = 0
direct_flow_write_total = 0

sample_strict_recall_miss_total = 0
sample_wrong_scope_proxy_total = 0
sample_hard_negative_total = 0
sample_direct_flow_write_total = 0
```

## Boundary

This is a runtime-canary simulation and rollback audit only. It does not
destructively replace or prune runtime operators.

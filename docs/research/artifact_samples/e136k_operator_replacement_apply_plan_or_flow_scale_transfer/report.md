# E136K Operator Replacement Apply Plan Or Flow Scale Transfer

```text
decision = e136k_operator_replacement_apply_plan_confirmed
next     = E136L_RUNTIME_REPLACEMENT_CANARY_AND_TIGHTENED_CHALLENGER_CONFIRM
```

## Metrics

```text
operator_count = 34
direct_canary_ready_count = 16
challenger_ood_required_count = 11
abstract_lineage_required_count = 7
runtime_mutation_allowed_now_count = 0
destructive_apply_count = 0
rollback_manifest_count = 16

current_activation_total = 188597925
selected_activation_total = 166354720
shadow_pruned_activation_total = 22243205
shadow_prune_ratio = 0.11794

strict_recall_miss_total = 0
wrong_scope_proxy_total = 0
hard_negative_total = 0
unsupported_answer_total = 0
direct_flow_write_total = 0

accepted_mutation_total = 96
mutation_attempt_total = 43720
mutation_accept_rate = 0.002196
```

## Boundary

This is an apply plan and canary manifest only. It does not destructively replace
or prune runtime operators.

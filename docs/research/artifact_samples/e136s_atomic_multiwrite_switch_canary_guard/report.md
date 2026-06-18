# E136S Atomic Multiwrite Default-Route Switch Canary Guard

```text
decision = e136s_atomic_multiwrite_default_route_switch_canary_guard_confirmed
next     = E136T_ATOMIC_MULTIWRITE_DEFAULT_ROUTE_PROBATION_ROLLOUT_DECISION
```

```text
default_route_switch_candidate = true
production_apply_allowed_now = false
default_route_case_count = 1536
default_route_success_count = 1536
switch_case_count = 2048
switch_success_count = 2048
default_route_applied_count = 1024
blocked_no_apply_count = 1024
rollback_snapshot_count = 2048
preview_guard_passed_count = 1024
preview_match_count = 2048
guard_false_apply_count = 0
guard_missed_apply_count = 0
blocked_mutation_count = 0
partial_write_count = 0
runtime_direct_write_count = 0
held_variant_promoted_count = 0
oracle_plan_feature_use_count = 0
```

Boundary: default-route switch canary guard only. No production apply.

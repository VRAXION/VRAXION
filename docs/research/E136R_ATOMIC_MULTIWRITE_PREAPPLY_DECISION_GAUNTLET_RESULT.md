# E136R Atomic Multiwrite Pre-Apply Decision Gauntlet Result

```text
decision = e136r_atomic_multiwrite_default_route_candidate_confirmed
next     = E136S_ATOMIC_MULTIWRITE_DEFAULT_ROUTE_SWITCH_CANARY_GUARD
```

E136R stress-tests the Rust atomic multiwrite path before any default-route
application. It combines a default-route regression suite with seeded overlay
canary stress cases. The result can mark the path as a default-route candidate,
but it does not authorize production apply.

## Decision

```text
default_route_candidate = true
production_apply_allowed_now = false
recommended_next = E136S_ATOMIC_MULTIWRITE_DEFAULT_ROUTE_SWITCH_CANARY_GUARD
```

## Result

```text
default_route_case_count = 1536
default_route_success_count = 1536
default_false_commit_count = 0
default_missed_commit_count = 0

canary_case_count = 2048
canary_success_count = 2048
canary_overlay_active_count = 2048
default_route_unchanged_count = 2048
rollback_snapshot_count = 2048
seeded_default_state_preserved_count = 2048

commit_single_count = 614
commit_multi_count = 205
commit_chunk_count = 205
defer_count = 615
reject_count = 409
atomic_write_total = 1844
rollback_commit_count = 204

partial_write_count = 0
order_independence_failure_count = 0
runtime_direct_write_count = 0
held_variant_promoted_count = 0
oracle_plan_feature_use_count = 0
destructive_delete_count = 0
checker_failure_count = 0
```

## Covered Stress Families

```text
default_clean_valid_binary_commit
default_wrong_feature_reject
default_corrupt_crc_reject
canary_disjoint_atomic_multiwrite
canary_homogeneous_chunk_commit
canary_single_primary_commit
canary_stale_snapshot_reject
canary_checksum_tamper_reject
canary_direct_flow_write_reject
canary_ambiguous_same_region_reject
canary_held_challenger_hold
canary_rollback_fallback_commit
canary_proposal_capacity_reject
```

## Interpretation

The candidate survived both default-route regression and overlay canary stress:

```text
default route
-> unchanged and regression-clean

overlay canary route
-> atomic single/multi/chunk commits exercised
-> stale/checksum/direct/ambiguous/capacity rejects exercised
-> rollback snapshot preserved
-> no partial writes or destructive deletes
```

This confirms the atomic multiwrite route as an E136S switch-canary candidate.
It still remains outside production apply until the default-route switch guard
passes.

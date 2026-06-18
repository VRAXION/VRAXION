# E136S Atomic Multiwrite Default-Route Switch Canary Guard Result

```text
decision = e136s_atomic_multiwrite_default_route_switch_canary_guard_confirmed
next     = E136T_ATOMIC_MULTIWRITE_DEFAULT_ROUTE_PROBATION_ROLLOUT_DECISION
```

E136S moves the E136R default-route candidate into a guarded switch-canary
runtime path. The atomic route may write to the default runtime state only after
snapshot + preview guard + preview/apply match. Rejected/deferred cases must
leave the default route unchanged.

## Decision

```text
default_route_switch_candidate = true
production_apply_allowed_now = false
guarded_default_route_apply_count = 1024
guarded_blocked_no_apply_count = 1024
recommended_next = E136T_ATOMIC_MULTIWRITE_DEFAULT_ROUTE_PROBATION_ROLLOUT_DECISION
```

## Runtime Delta

```text
switch_canary_api = LockedBodyRuntime::process_atomic_default_route_switch_canary
switch_canary_config = AtomicDefaultRouteSwitchCanaryConfig::e136s_switch_canary
preview_api = LockedBodyRuntime::process_atomic_proposals_preview
agency_api = agency_decide_atomic_batch

switch_canary_active = true
default_route_apply_allowed = true
rollback_snapshot_required = true
preview_match_required = true
production_apply_allowed_now = false
```

## Result

```text
default_route_case_count = 1536
default_route_success_count = 1536
default_false_commit_count = 0
default_missed_commit_count = 0

switch_case_count = 2048
switch_success_count = 2048
switch_canary_active_count = 2048
rollback_snapshot_count = 2048
rollback_ready_count = 2048
preview_checked_count = 2048
preview_guard_passed_count = 1024
preview_match_count = 2048
default_route_applied_count = 1024
blocked_no_apply_count = 1024
default_route_unchanged_on_block_count = 1024

commit_single_count = 614
commit_multi_count = 205
commit_chunk_count = 205
defer_count = 615
reject_count = 409
atomic_write_total = 1844
rollback_commit_count = 204

guard_false_apply_count = 0
guard_missed_apply_count = 0
blocked_mutation_count = 0
preview_mismatch_count = 0
rollback_triggered_count = 0
partial_write_count = 0
runtime_direct_write_count = 0
held_variant_promoted_count = 0
oracle_plan_feature_use_count = 0
destructive_delete_count = 0
checker_failure_count = 0
```

## Covered Switch Families

```text
default_clean_valid_binary_commit
default_wrong_feature_reject
default_corrupt_crc_reject
switch_disjoint_atomic_multiwrite
switch_homogeneous_chunk_commit
switch_single_primary_commit
switch_stale_snapshot_reject
switch_checksum_tamper_reject
switch_direct_flow_write_reject
switch_ambiguous_same_region_reject
switch_held_challenger_hold
switch_rollback_fallback_commit
switch_proposal_capacity_reject
```

## Interpretation

The guarded switch path now performs real default-route writes for safe atomic
cases and preserves the default route for blocked cases:

```text
safe atomic cases
-> snapshot
-> preview
-> preview guard pass
-> apply to default runtime
-> preview/apply match

unsafe or unresolved cases
-> snapshot
-> preview
-> guard block
-> no default-route mutation
```

This confirms the route as an E136T probation-rollout candidate. It still does
not authorize unrestricted production apply.

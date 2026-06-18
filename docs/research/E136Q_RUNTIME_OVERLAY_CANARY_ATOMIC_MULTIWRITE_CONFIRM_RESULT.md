# E136Q Runtime Overlay Canary Atomic Multiwrite Confirm Result

```text
decision = e136q_runtime_overlay_canary_atomic_multiwrite_confirmed
next     = E136R_ATOMIC_MULTIWRITE_DEFAULT_ROUTE_OR_PRODUCTION_APPLY_DECISION
```

E136Q moves the E136P Rust implementation preview into a runtime overlay canary.
The canary route calls the new atomic multiwrite API on an overlay clone while
the default runtime route remains unchanged.

## Runtime Delta

```text
locked_body_canary_api = LockedBodyRuntime::process_atomic_overlay_canary
preview_api = LockedBodyRuntime::process_atomic_proposals_preview
agency_api = agency_decide_atomic_batch

canary_overlay_active = true
default_route_unchanged = true
rollback_ready = true
production_apply_allowed_now = false
```

## Result

```text
default_route_case_count = 1
default_route_success_count = 1

canary_case_count = 10
canary_success_count = 10

commit_single_count = 3
commit_multi_count = 1
commit_chunk_count = 1
defer_count = 3
reject_count = 2
atomic_write_total = 9

partial_write_count = 0
order_independence_failure_count = 0
runtime_direct_write_count = 0
held_variant_promoted_count = 0
oracle_plan_feature_use_count = 0
destructive_delete_count = 0

direct_flow_write_reject_count = 1
stale_snapshot_reject_count = 1
checksum_tamper_reject_count = 1
ambiguous_same_region_reject_count = 1
proposal_capacity_reject_count = 1

rollback_snapshot_count = 10
checker_failure_count = 0
```

## Covered Canary Cases

```text
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

The new Rust path is now behind a canary overlay boundary:

```text
default runtime route
-> unchanged

canary overlay route
-> atomic multiwrite/chunk/rollback/reject exercised
-> rollback snapshot taken
-> no production apply
```

The next decision is whether to keep this in canary, expand canary coverage, or
allow a default-route/prod-apply candidate under E136R. This result does not
authorize destructive deletion or default production replacement.

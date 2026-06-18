# E136P Runtime Atomic Multiwrite Implementation Preview Result

```text
decision = e136p_runtime_atomic_multiwrite_implementation_preview_confirmed
next     = E136Q_RUNTIME_OVERLAY_CANARY_ATOMIC_MULTIWRITE_CONFIRM
```

E136P moves the E136O challenger policy into the Rust runtime preview surface.
This adds a concrete Agency atomic batch API and a `LockedBodyRuntime` preview
entrypoint while keeping production apply disabled.

## Runtime Delta

```text
agency_atomic_batch_api = agency_decide_atomic_batch
locked_body_preview_api = LockedBodyRuntime::process_atomic_proposals_preview
rust_runtime_api_added = true
implementation_preview_ready = true
production_apply_allowed_now = false
```

The existing single-proposal Agency path remains intact. The new path accepts a
bounded proposal batch, filters unsafe proposals, resolves primary/rollback/held
roles, then either commits the full write-set atomically or commits nothing.

## Result

```text
case_count = 10
success_count = 10

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
checker_failure_count = 0
```

## Covered Cases

```text
disjoint_atomic_multiwrite
homogeneous_chunk_commit
single_primary_commit
stale_snapshot_reject
checksum_tamper_reject
direct_flow_write_reject
ambiguous_same_region_reject
held_challenger_hold
rollback_fallback_commit
proposal_capacity_reject
```

## Boundary

This is implementation preview, not production apply. The next step is a runtime
overlay/canary confirm that exercises the new Rust API from a canary route
before any default runtime replacement is allowed.

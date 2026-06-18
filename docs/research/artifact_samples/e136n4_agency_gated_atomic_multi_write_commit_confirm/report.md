# E136N4 Agency-Gated Atomic Multi-Write Commit Confirm

```text
decision = e136n4_agency_gated_atomic_multi_write_confirmed
next     = E136O_CHALLENGER_OOD_RUNTIME_REPLACEMENT_GAUNTLET
```

E136N4 checks whether Agency can approve multiple Flow writes in one atomic
batch after parallel proposal fanout.

## Result

```text
case_count = 225
accuracy = 1.000000

atomic_multi_region_commit_case_count = 37
atomic_write_total = 191
partial_write_count = 0
order_independence_failure_count = 0
runtime_direct_write_count = 0

direct_flow_write_reject_count = 34
stale_snapshot_reject_count = 34
checksum_tamper_reject_count = 34
ambiguous_same_region_reject_count = 34

expected_child_check_count = 36
agency_child_check_count = 36
expected_flow_chunk_count = 10
agency_flow_chunk_count = 10
held_variant_promoted_count = 0
destructive_delete_count = 0
```

## Interpretation

The supported shape is:

```text
parallel read/propose
-> Agency validates write sets
-> Agency builds one atomic batch
-> batch commits multiple regions or commits nothing
```

This is not parallel direct Flow write. Proposals do not mutate Flow directly;
Agency commits the approved batch after conflict, snapshot, checksum, held, and
direct-write checks.

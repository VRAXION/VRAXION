# E136O Prep Agency Atomic Multiwrite Train/Gauntlet Overnight Result

```text
decision = e136o_prep_agency_atomic_multiwrite_train_gauntlet_confirmed
next     = E136O_CHALLENGER_OOD_RUNTIME_REPLACEMENT_GAUNTLET
```

E136O runs a four-hour shadow/probation training gauntlet over the E136N,
E136N2, E136N3, and E136N4 agency/commit surface. It lets a policy mutate and
prune under train, heldout, and OOD/noisy checks, but does not apply production
replacement.

## Result

```text
elapsed_seconds = 14400.004
cycle_count = 2488677
mutation_attempts = 2488680

accepted_mutations = 6
accepted_prune_mutations = 4

train_accuracy = 1.000000
heldout_accuracy = 1.000000
ood_accuracy = 1.000000
full_accuracy = 1.000000

full_case_count = 7146
full_atomic_multi_region_commit_case_count = 765
full_atomic_write_total = 3667

full_direct_flow_write_reject_count = 1729
full_stale_snapshot_reject_count = 1008
full_checksum_tamper_reject_count = 2224
full_ambiguous_same_region_reject_count = 1214

full_partial_write_count = 0
full_order_independence_failure_count = 0
full_runtime_direct_write_count = 0
full_held_variant_promoted_count = 0
checker_failure_count = 0
```

## Accepted Shape

The best shadow policy kept the important commit behavior:

```text
enable_chunk_commit = true
enable_multi_write = true
max_multi_write = 3
reject_direct_flow_write = true
reject_stale_snapshot = true
reject_checksum_tamper = true
enable_rollback_audit_write = true
enable_rollback_fallback = true
```

The run converged quickly, then plateaued: all six accepted changes happened
near the start, and the remaining long run acted as a stability gauntlet.

## Boundary

This is not a production replacement proof. The policy evaluation is still a
proxy/shadow gauntlet and uses case metadata such as expected action/family
while scoring behavior. In particular, `reject_ambiguous_same_region = false`
in the final serialized policy should not be treated as permission to remove
the production ambiguous-region guard; the full artifact still rejected 1,214
ambiguous same-region cases through the proxy decision path.

The safe interpretation is:

```text
agency-gated atomic multiwrite is trainable, pruneable, and stable under the
current noisy/OOD shadow gauntlet; production replacement still needs a runtime
challenger gauntlet without oracle-style labels.
```

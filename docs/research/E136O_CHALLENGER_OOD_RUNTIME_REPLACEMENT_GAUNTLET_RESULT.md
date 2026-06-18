# E136O Challenger OOD Runtime Replacement Gauntlet Result

```text
decision = e136o_challenger_ood_runtime_replacement_gauntlet_confirmed
next     = E136P_RUNTIME_ATOMIC_MULTIWRITE_IMPLEMENTATION_PREVIEW
```

E136O challenger replays the E136O/E136N4 agency atomic-write surface with a
runtime-style commit policy. Unlike the previous shadow/proxy training run, the
challenger does not use oracle plan features such as `expected_action` or case
family while building its commit plan. Oracle labels are used only by the
checker.

## Result

```text
base_case_count = 225
full_case_count = 7146

train_accuracy = 1.000000
heldout_accuracy = 1.000000
ood_accuracy = 1.000000
full_accuracy = 1.000000

full_atomic_multi_region_commit_case_count = 765
full_atomic_write_total = 3667

full_partial_write_count = 0
full_order_independence_failure_count = 0
full_runtime_direct_write_count = 0
full_held_variant_promoted_count = 0
full_oracle_plan_feature_use_count = 0

full_direct_flow_write_reject_count = 1729
full_stale_snapshot_reject_count = 1008
full_checksum_tamper_reject_count = 2224
full_ambiguous_same_region_reject_count = 1214

implementation_ready = true
implementation_preview_allowed = true
production_apply_allowed_now = false
checker_failure_count = 0
```

## Runtime Manifest

The implementation-prep runtime manifest keeps the conservative guards that
the shadow policy was not allowed to remove:

```text
reject_direct_flow_write = true
reject_stale_snapshot = true
reject_checksum_tamper = true
reject_ambiguous_same_region = true
hold_held_variants = true
stable_write_order = true

enable_chunk_commit = true
enable_multi_write = true
enable_rollback_fallback = true
enable_rollback_audit_write = true
max_multi_write = 3
chunk_min_support = 3
require_whole_group_for_chunk = true
```

## Interpretation

This closes the main caveat from the E136O overnight run:

```text
shadow/proxy train result
-> oracle-free runtime challenger
-> implementation preview allowed
```

The safe next step is an implementation preview that wires this policy into a
runtime-facing path behind an overlay/canary boundary. It is still not a
production apply or destructive prune authorization.

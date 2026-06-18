# E136O Challenger OOD Runtime Replacement Gauntlet

```text
decision = e136o_challenger_ood_runtime_replacement_gauntlet_confirmed
next     = E136P_RUNTIME_ATOMIC_MULTIWRITE_IMPLEMENTATION_PREVIEW
```

E136O challenger replays the E136O/E136N4 surface with a runtime-style commit
policy that does not use oracle plan features such as expected action or case
family. Oracle labels are used only by the checker.

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

implementation_ready = True
production_apply_allowed_now = False
```

## Boundary

This prepares implementation. It does not apply production runtime replacement.
The conservative runtime manifest keeps ambiguous-region, held-variant, and
stable-order guards even where the previous shadow policy pruned them.

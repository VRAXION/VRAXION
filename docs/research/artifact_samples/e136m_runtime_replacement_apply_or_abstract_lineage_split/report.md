# E136M Runtime Replacement Apply Or Abstract Lineage Split

```text
decision = e136m_runtime_replacement_overlay_and_lineage_split_confirmed
next     = E136N_CHALLENGER_OOD_RUNTIME_REPLACEMENT_GAUNTLET
```

E136M materializes the first runtime-facing replacement overlay from E136L.
The direct canary-passed rows become active selected variants in the overlay.
Legacy triggers are disabled only inside the overlay and retained for rollback.
Challenger and abstract rows remain held.

## Result

```text
operator_count = 34
runtime_overlay_active_count = 16
runtime_overlay_apply_count = 16
verified_replacement_apply_count = 7
light_prune_overlay_apply_count = 9
legacy_trigger_disabled_in_overlay_count = 16
legacy_trigger_retained_for_rollback_count = 16

challenger_ood_queue_count = 11
challenger_runtime_overlay_active_count = 0
abstract_lineage_split_count = 7
abstract_runtime_overlay_active_count = 0

rollback_snapshot_count = 16
rollback_trigger_count = 0
production_destructive_delete_count = 0
runtime_mutation_allowed_now_count = 16

current_activation_total = 188597925
shadow_selected_activation_total = 166354720
shadow_pruned_activation_total = 22243205

runtime_overlay_activation_total = 185147668
runtime_overlay_removed_activation_total = 3450257
runtime_overlay_removed_activation_ratio = 0.018294
direct_canary_removed_activation_total = 3450257
challenger_candidate_removed_not_applied = 18792948

strict_recall_miss_total = 0
wrong_scope_proxy_total = 0
hard_negative_total = 0
unsupported_answer_total = 0
direct_flow_write_total = 0
```

## Boundary

This is a runtime-facing overlay/apply manifest. It does not destructively
delete legacy operators. Challenger/OOD rows and abstract lineage rows remain
held.

# E136N Primary/Secondary Variant Governance Result

```text
decision = e136n_primary_secondary_variant_governance_confirmed
next     = E136O_CHALLENGER_OOD_RUNTIME_REPLACEMENT_GAUNTLET
```

E136N turns E136M's runtime-facing overlay into a durable primary/secondary
variant registry skeleton. Every operator has exactly one primary variant and
at least one secondary variant. This step creates the governance lanes for
primary active, current primary, abstract-current primary, rollback secondary,
challenger secondary, lineage-hold secondary, and future retired-redundant
states.

## Result

```text
operator_count = 34
variant_registry_row_count = 68
primary_variant_count = 34
secondary_variant_count = 34

primary_active_count = 16
primary_current_count = 11
primary_abstract_current_count = 7
secondary_rollback_count = 16
secondary_challenger_count = 11
secondary_lineage_hold_count = 7
retired_redundant_count = 0

retirement_lane_created_count = 16
retirement_candidate_count = 0
destructive_delete_count = 0
ambiguous_primary_operator_count = 0
missing_primary_operator_count = 0
orphan_secondary_count = 0

runtime_overlay_removed_activation_total = 3,450,257
challenger_candidate_removed_not_applied = 18,792,948
rollback_snapshot_count = 16
rollback_trigger_count = 0

strict_recall_miss_total = 0
wrong_scope_proxy_total = 0
hard_negative_total = 0
unsupported_answer_total = 0
direct_flow_write_total = 0
checker_failure_count = 0
```

## Interpretation

```text
E136M = runtime-facing overlay manifest
E136N = variant governance skeleton over that overlay
```

The current default/runtime side is now expressed as primary variants:

```text
16 primary_active selected overlay variants
11 primary_current legacy/default variants while challenger rows wait
7 primary_abstract_current variants while abstract lineage is resolved
```

The held/fallback side is now explicit as secondary variants:

```text
16 secondary_rollback legacy variants
11 secondary_challenger selected variants awaiting E136O
7 secondary_lineage_hold abstract selected variants
0 retired_redundant variants
```

## Boundary

This is variant governance metadata only. It does not destructively delete
operators, retire variants, or apply the challenger/OOD queue.

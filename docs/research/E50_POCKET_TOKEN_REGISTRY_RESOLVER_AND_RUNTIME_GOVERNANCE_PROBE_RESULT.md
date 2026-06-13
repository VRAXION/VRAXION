# E50 Pocket Token Registry Resolver And Runtime Governance Probe Result

## Decision

```text
decision = e50_pocket_token_registry_governance_positive
target_checker_failure_count = 0
sample_only_checker_failure_count = 0
run_id = 10e4a5d41612d4c0
```

E50 tested whether a Pocket Library can be called through stable
`pocket_uid`/digest/descriptor records rather than human filenames, while the
Registry and Pocket Manager block unsafe runtime calls.

## Result Table

```text
| system | governance_success | route_accuracy | alias_rename_survival | digest_mismatch_block_rate | token_swap_block_rate | banned_quarantine_block_rate | stale_token_reaudit_rate | unsafe_load_rate | active_set_reduction | cost_adjusted_utility |
|---|---|---|---|---|---|---|---|---|---|---|
| filename_alias_router_control | 0.245 | 0.662 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.523 | 0.000 | -0.041 |
| uid_only_no_descriptor_control | 0.031 | 0.084 | 0.075 | 0.000 | 0.000 | 0.000 | 0.000 | 0.688 | 0.000 | -0.337 |
| descriptor_token_router_no_guard | 0.370 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.523 | 0.000 | 0.083 |
| registry_guard_only_static_active_set | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.417 | 1.017 |
| full_library_scan_control | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.976 |
| token_registry_manager_active_set | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.714 | 1.046 |
| oracle_registry_reference | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.917 | 1.066 |
```

## Learned Guard Policy

The mutation/rollback guard-policy loop tried 512 mutations:

```text
accepted = 6
rejected = 506
rollback_count = 506
attempts_to_95 = 17
```

The final enabled guard groups were:

```json
[
  "abi",
  "active_set",
  "alias_independent",
  "digest",
  "lifecycle",
  "stale",
  "token_binding"
]
```

## Interpretation

The primary system confirms the intended lock:

```text
human_alias = documentation only
pocket_uid = immutable machine identity
content_digest = frozen artifact integrity
PocketToken = behavioral/capability descriptor
Registry = uid -> artifact/ABI/lifecycle resolver
Pocket Manager / Agency Guard = runtime load authority
```

The important negative control was `descriptor_token_router_no_guard`: it had
perfect route accuracy on valid route rows, but still allowed unsafe loads on
52.3% of all rows because it did not enforce digest, token-binding, lifecycle,
stale-token, and ABI gates. This shows that PocketToken routing alone is not a
safe library interface.

The primary `token_registry_manager_active_set` matched the clean governance
controls while using a much smaller active set:

```text
avg_active_set_size = 3.430 / 12
active_set_reduction = 0.714
unsafe_load_rate = 0.000
cost_adjusted_utility = 1.046
```

That beat the full-library scan control on cost-adjusted utility while
preserving exact routing and hard-failing alias rename, content digest mismatch,
token/pocket swap, unsafe lifecycle, stale token, and ABI mismatch cases.

## Boundary

This is a controlled symbolic/numeric runtime governance probe. It does not
generate new pockets, test raw language reasoning, deployed assistant behavior,
model-scale behavior, AGI, or consciousness.

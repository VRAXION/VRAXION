# E50 Pocket Token Registry Resolver And Runtime Governance Probe Contract

## Purpose

E50 tests the Pocket Library call boundary after E49.

Core question:

```text
Can the runtime call pockets through stable PocketToken descriptors,
immutable pocket_uid records, content digests, and registry governance without
depending on human filenames or unsafe direct loads?
```

This probe locks the translation layer:

```text
human_alias = documentation only
pocket_uid = immutable machine identity
content_digest = frozen artifact integrity
PocketToken = behavioral/capability descriptor
Registry = uid -> artifact/ABI/lifecycle resolver
Pocket Manager / Agency Guard = runtime load authority
```

## Boundary

This is a controlled symbolic/numeric registry and runtime governance probe. It
does not generate new pockets, train raw language models, prove deployed
assistant behavior, AGI, consciousness, or model-scale behavior.

## Systems

```text
filename_alias_router_control
uid_only_no_descriptor_control
descriptor_token_router_no_guard
registry_guard_only_static_active_set
full_library_scan_control
token_registry_manager_active_set
oracle_registry_reference
```

## Runtime Requirements

The primary system must prove:

```text
alias rename does not break routing
pocket_uid remains immutable
content_digest mismatch hard-fails
token/pocket swap hard-fails
banned/quarantine pockets do not load on primary route
stale token version requests re-audit
ABI mismatch hard-fails
active Pocket Set is smaller than full library scan
PocketToken descriptor routing still finds the right callable pocket
```

## Mutation Contract

The primary system uses mutation/rollback over guard groups:

```text
alias_independent
digest
token_binding
lifecycle
stale
abi
active_set
```

Each attempted guard-policy mutation must either improve the governance score
and be accepted or rollback. Accepted/rejected counts and rollback counts must
be written.

## Metrics

```text
governance_success
route_accuracy
alias_rename_survival
digest_mismatch_block_rate
token_swap_block_rate
banned_quarantine_block_rate
stale_token_reaudit_rate
abi_mismatch_block_rate
unsafe_load_rate
avg_active_set_size
active_set_reduction
avg_cost
cost_adjusted_utility
```

## Decisions

Allowed decisions:

```text
e50_pocket_token_registry_governance_positive
e50_alias_filename_control_sufficient
e50_token_routing_without_guard_unsafe
e50_active_set_overprunes
e50_registry_guard_blocks_but_routing_weak
e50_invalid_artifact_detected
```

Positive requires:

```text
token_registry_manager_active_set governance_success >= 0.95
route_accuracy >= 0.95
unsafe_load_rate = 0.0
alias_rename_survival >= 0.95
digest_mismatch_block_rate = 1.0
token_swap_block_rate = 1.0
banned_quarantine_block_rate = 1.0
stale_token_reaudit_rate = 1.0
abi_mismatch_block_rate = 1.0
active_set_reduction >= 0.25
cost_adjusted_utility > full_library_scan_control
deterministic replay passes
target checker failure_count = 0
sample-only checker passes
```

## Required Artifacts

```text
backend_manifest.json
registry_schema.json
pocket_registry.json
pocket_tokens.json
resolver_events.jsonl
governance_report.json
active_set_report.json
token_swap_report.json
alias_rename_report.json
digest_integrity_report.json
stale_token_report.json
registry_guard_mutation_history.jsonl
system_results.json
aggregate_metrics.json
deterministic_replay.json
decision.json
summary.json
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
results_table.md
report.md
```

## Sample Pack

The sample pack must live under:

```text
docs/research/artifact_samples/e50_pocket_token_registry_resolver_and_runtime_governance_probe/
```

## Hard Requirements

```text
no gradient descent
no optimizer/backprop
row-level resolver events
real mutation/rollback for primary guard policy
accepted/rejected mutation evidence
rollback count equals rejected count
target checker passes with failure_count = 0
sample-only checker passes
```

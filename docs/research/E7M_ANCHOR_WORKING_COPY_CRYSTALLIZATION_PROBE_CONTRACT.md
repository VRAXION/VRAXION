# E7M Anchor Working Copy Crystallization Probe Contract

## Purpose

`E7M_ANCHOR_WORKING_COPY_CRYSTALLIZATION_PROBE` tests one pocket-library lifecycle rule:

```text
keep a frozen minimal anchor
mutate/prune/repair only a working copy
promote only through guarded validation
discard the working copy when the guard fails
```

The question is not whether the pocket-flow proxy can answer rows. E7M asks whether this lifecycle discipline improves net utility, stability, drift recovery, and safe pruning compared with mutating the only pocket directly.

## Systems

Exactly these systems are compared:

```text
no_anchor_direct_mutation
frozen_anchor_only
frozen_anchor_plus_mutable_copy
frozen_anchor_plus_mutable_copy_plus_pruning
frozen_anchor_plus_mutable_copy_plus_prune_and_promote
multi_copy_competition
random_copy_control
oracle_anchor_reference
```

## Lifecycle

```text
spawn
-> validate
-> crystallize/prune
-> save frozen_anchor_v1
-> fork mutable_working_copy
-> mutate/prune/repair working copy
-> promote to frozen_anchor_v2 only if guarded improvement passes
-> otherwise discard working copy and keep anchor
```

Frozen anchors are not overwritten directly. Mutation and pruning target direct pockets, pre-anchor candidates, or mutable working copies only.

## Net Utility

```text
net_utility =
  raw_usefulness
  - spawn_cost
  - repair_cost
  - prune_cost
  - maintenance_cost
  - copy_cost
  - route_step_cost
  - bad_promotion_penalty
  - junk_penalty
  - delayed_regret_penalty
```

Promotion requires net improvement, delayed-validation survival, reuse above threshold, route or cost improvement, and random-control separation.

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
lifecycle_contract_report.json
anchor_working_copy_report.json
crystallization_pruning_report.json
promotion_guard_report.json
system_results.json
mutation_history.json
leakage_report.json
deterministic_replay.json
aggregate_metrics.json
decision.json
summary.json
report.md
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
```

Every long run writes heartbeat/progress, mutation snapshots, and partial aggregate state before final artifacts.

## Checker Gates

The checker fails on missing systems, missing row-level eval, missing mutation accept/reject/rollback counts, rollback mismatch, missing parameter diff/hash, deterministic replay mismatch, optimizer/backprop usage inside mutation functions, frozen-anchor overwrite flags, missing lineage/version history, failed random control, hardcoded artifact-style decisions, or broad claims beyond this controlled proxy.

## Decision Labels

```text
e7m_anchor_working_copy_positive
e7m_post_spawn_crystallization_positive
e7m_safe_mutable_copy_promotion_positive
e7m_multi_copy_competition_positive
e7m_freeze_only_preferred_mutation_too_risky
e7m_direct_mutation_sufficient_anchor_unneeded
e7m_anchor_copy_overhead_too_high
e7m_pruning_brittleness_detected
e7m_promotion_guard_failure
e7m_artifact_or_task_too_easy
```

## Boundary

E7M is a controlled symbolic/numeric pocket-library lifecycle probe. It is not a raw-language or deployed-model result.

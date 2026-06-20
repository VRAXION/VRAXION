# E112 Gold To Core Prune Heavy Probation Wave Contract

## Purpose

E112 takes the merged scoped Gold pool from E109, E110, and E111 and runs a
prune-heavy CoreMemoryCandidate probation wave.

This is a CoreMemoryCandidate qualification run only. It is not PermaCore,
TrueGolden, or final training.

## Inputs

```text
target/pilot_wave/e109_operator_rank_ladder_and_golden_watch_probation_mode/
target/pilot_wave/e110_promote_or_drop_operator_grind_wave1/
target/pilot_wave/e111_bronze_mutation_prune_promote_or_drop_wave/
```

Candidate source:

```text
merged rank = Gold
```

## Required Variant Search

For every Gold Operator, evaluate:

```text
current_gold
deep_prune_mutation
minimal_core_prune
core_simplification_challenger
no_harm_shadow_import
```

At least 50% of selected variants must be prune-heavy:

```text
selected_prune_ratio >= 0.50
```

## CoreMemoryCandidate Gates

```text
qualified_activation >= 100000
combined_family_coverage >= 15
campaign_count >= 8
hard_negative = 0
wrong_scope_call_rate = 0
false_commit_rate = 0
unsupported_answer_rate = 0
negative_transfer_rate = 0
reload_shadow_pass = true
challenger_pass = true
prune_pass = true
long_horizon_no_harm_pass = true
negative_scope_pass = true
```

## Required Artifacts

```text
run_manifest.json
wave_manifest.json
input_rank_report.json
wave_results.json
promotion_report.json
operator_stats.json
mutation_variant_report.json
mutation_events.json
mutation_summary.json
duration_report.json
progress.jsonl
partial_aggregate_snapshot.json
aggregate_metrics.json
deterministic_replay.json
decision.json
summary.json
report.md
row_level_samples.jsonl
```

Sample pack:

```text
archived_public_artifact_sample_removed
```

## Decision Labels

```text
e112_gold_to_core_prune_heavy_probation_confirmed
e112_gold_to_core_prune_heavy_probation_incomplete
```

## Boundary

E112 may emit:

```text
CoreMemoryCandidate
```

E112 must not emit:

```text
PermaCore
TrueGolden
```

PermaCore/TrueGolden requires a later, larger grind.

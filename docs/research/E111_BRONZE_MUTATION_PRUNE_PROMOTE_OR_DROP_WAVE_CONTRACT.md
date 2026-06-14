# E111 Bronze Mutation Prune Promote Or Drop Wave Contract

## Purpose

E111 takes the remaining E109 Bronze Operators and applies an active
mutation/prune/challenger wave. The goal is to resolve every Bronze candidate:
promote a validated mutated/pruned variant to scoped Gold, drop/deprecate it, or
RedFlag it.

This is not Diamond, Core, PermaCore, TrueGolden, or final training promotion.

## Inputs

```text
target/pilot_wave/e109_operator_rank_ladder_and_golden_watch_probation_mode/
```

Candidate source:

```text
rank = Bronze
```

## Required Variant Search

For every Bronze Operator, evaluate at least these variants:

```text
base_unmodified
scope_adapter_mutation
io_contract_prune
mutation_plus_prune
sibling_challenger
```

Promotion to Gold is invalid if the selected variant is `base_unmodified`.

## Required Safety Gates

```text
hard_negative = 0
wrong_scope_call_rate = 0
false_commit_rate = 0
unsupported_answer_rate = 0
negative_transfer_rate = 0
reload_shadow_pass = true
challenger_pass = true
prune_pass = true
```

Gold threshold:

```text
qualified_activation >= 3000
combined_family_coverage >= 5
campaign_count >= 3
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
docs/research/artifact_samples/e111_bronze_mutation_prune_promote_or_drop_wave/
```

## Decision Labels

```text
e111_bronze_mutation_prune_wave_gold_conversion_confirmed
e111_bronze_mutation_prune_wave_incomplete
```

## Boundary

E111 resolves Bronze under scoped operator-library ranking only.

It does not claim:

```text
DiamondCandidate
CoreMemoryCandidate
PermaCore
TrueGolden
open-domain capability
final training readiness
```

# E126 E125 Gold To Orange Legendary Probation Gauntlet Contract

## Purpose

E126 takes the twenty scoped Gold text-understanding operators produced by E125
and pushes them through a stricter Orange/Legendary probation gate.

This is a scoped operator lifecycle probe. It is not Core, PermaCore,
TrueGolden, final training, Gemma-level generation, or open-domain reasoning.

## Artifact Root

```text
target/pilot_wave/e126_e125_gold_to_orange_legendary_probation_gauntlet/
```

## Input

```text
target/pilot_wave/e125_broad_text_understanding_candidate_expansion_wave/
```

Expected input:

```text
20 scoped Gold operators
```

## Orange / Legendary Candidate Gate

Each operator must satisfy:

```text
qualified_activation >= 300000
family_coverage >= 12
campaign_count >= 8
hard_negative = 0
false_commit = 0
wrong_scope_call = 0
unsupported_answer = 0
negative_transfer = 0
direct_flow_write = 0
reload_shadow_pass = true
negative_scope_pass = true
challenger_pass = true
prune_pass = true
```

The only allowed promotion label is:

```text
OrangeLegendaryCandidate
```

No operator may be promoted to Core, PermaCore, or TrueGolden in E126.

## Required Artifacts

```text
run_manifest.json
input_gold_report.json
probation_report.json
operator_orange_results.json
operator_cards.json
variant_report.json
mutation_summary.json
row_level_samples.jsonl
progress.jsonl
partial_aggregate_snapshot.json
aggregate_metrics.json
deterministic_replay.json
decision.json
summary.json
report.md
checker_summary.json
operator_registry/*.json
```

## Decision Labels

```text
e126_orange_legendary_probation_confirmed
e126_insufficient_orange_probation_evidence
e126_redflag_detected
```

## Pass Rules

The checker must pass only if:

```text
artifact contract matches
exactly 20 E125 Gold inputs are evaluated
all 20 reach OrangeLegendaryCandidate
all no-harm metrics remain zero
all reload/negative-scope/challenger/prune gates pass
deterministic replay hash matches
progress and row-level samples exist
no Core / PermaCore / TrueGolden / final training claim is made
```

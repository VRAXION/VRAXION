# E120 FineWeb Skill Farm To Gold Wave Contract

## Purpose

E119 found FineWeb skill gaps. E120 turns the strong FarmCandidates into saved
scoped Operator candidates and tests whether they can reach Gold.

This is:

```text
FineWeb skill farm
scoped Operator creation
Gold promotion gate
```

This is not:

```text
Core promotion
PermaCore
TrueGolden
Gemma-style free-form generation
final training
```

## Inputs

```text
target/pilot_wave/e119_fineweb_skill_mining_and_text_io_delta_probe/
```

E120 uses only E119 `FarmCandidate` rows above the support threshold.

## Candidate Gates

Gold requires:

```text
qualified_activation >= 3000
family_coverage >= 5
campaign_count >= 3
hard_negative = 0
wrong_scope_call = 0
false_commit = 0
unsupported_answer = 0
negative_transfer = 0
negative_scope_pass = true
challenger_pass = true
prune_pass = true
reload_shadow_pass = true
```

## Required Artifacts

```text
run_manifest.json
input_candidate_report.json
operator_library_manifest.json
operator_cards.json
operator_gold_results.json
variant_report.json
promotion_report.json
negative_scope_report.json
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
e120_fineweb_skill_farm_gold_positive
e120_skill_farm_partial
e120_skill_farm_hard_negative_detected
e120_no_farm_candidates
```


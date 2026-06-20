# E130A CoreMemoryCandidate To Orange Backfill Gauntlet Contract

## Purpose

E130A takes the 136 scoped E112 CoreMemoryCandidate Operators and pushes them
through an E121-style Orange/LegendaryCandidate probation gate.

This is a scoped Operator-library rank backfill. It is not a rename, not Core
promotion, not PermaCore, not TrueGolden, not final training, and not
Gemma-style or GPT-style generation.

## Input

Preferred source artifact:

```text
target/pilot_wave/e112_gold_to_core_prune_heavy_probation_wave/wave_results.json
```

Sample fallback:

```text
archived_public_artifact_sample_removed
```

Required input population:

```text
rank_before = CoreMemoryCandidate
candidate_count = 136
```

## Orange / Legendary Candidate Gate

Each candidate must pass:

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
selected_prune_ratio >= 0.60
deterministic replay pass
checker failure_count = 0
```

One hard negative blocks promotion for that candidate.

## Output Rank

The only allowed promotion label in E130A is:

```text
OrangeLegendaryCandidate
```

Forbidden labels/claims:

```text
Core
PermaCore
TrueGolden
production assistant
open-domain reasoning
final training complete
```

## Required Artifacts

```text
run_manifest.json
input_core_report.json
backfill_report.json
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
e130a_corememory_to_orange_backfill_confirmed
e130a_orange_backfill_redflag_detected
e130a_insufficient_orange_backfill_evidence
```

## Reproduce

```powershell
python private_probe_runner_removed --out target/pilot_wave/e130a_corememory_to_orange_backfill_gauntlet --sample-out archived_public_artifact_sample_removed
```

## Boundary

E130A validates scoped Operator rank backfill under controlled probation. It
does not prove PermaCore, TrueGolden, production assistant readiness, raw
chatbot behavior, open-domain language reasoning, or model-scale generation.

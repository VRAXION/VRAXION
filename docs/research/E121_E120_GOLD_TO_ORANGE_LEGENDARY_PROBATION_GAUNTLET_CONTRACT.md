# E121 E120 Gold To Orange Legendary Probation Gauntlet Contract

## Purpose

E121 takes the eight scoped Gold FineWeb text-grounding Operators created by
E120 and pushes them through a stricter Orange/Legendary probation gate.

This is a scoped Operator-library rank grind. It is not Core promotion, not
PermaCore, not TrueGolden, not final training, and not Gemma-style text
generation.

## Input

Source artifact:

```text
target/pilot_wave/e120_fineweb_skill_farm_to_gold_wave/
```

Required candidates:

```text
definition_term_anchor_lens
named_entity_anchor_lens
causal_relation_lens
date_entity_timeline_lens
comparison_quantifier_guard
procedure_step_parser_lens
safety_domain_caution_guard
quote_speaker_attribution_lens
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

The only allowed promotion label in E121 is:

```text
OrangeLegendaryCandidate
```

Forbidden labels/claims:

```text
Core
PermaCore
TrueGolden
Gemma-level generation
final training complete
```

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
e121_orange_legendary_probation_confirmed
e121_redflag_detected
e121_insufficient_orange_probation_evidence
```

## Boundary

E121 validates scoped FineWeb text-grounding Operators under controlled
probation. It does not prove open-domain language reasoning, raw chatbot
behavior, model-scale generation, or production readiness.

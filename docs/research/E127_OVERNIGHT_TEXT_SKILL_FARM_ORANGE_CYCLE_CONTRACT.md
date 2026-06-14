# E127 Overnight Text Skill Farm Orange Cycle Contract

## Purpose

E127 is an unattended cyclic runner for the current text-skill farming loop:

```text
candidate discovery
-> scoped Gold farm
-> Orange/Legendary probation
-> repeat
```

The runner excludes already-active Orange operators from later cycles and writes
continuous root-level and per-cycle progress artifacts.

This is scoped operator farming only. It is not Core, PermaCore, TrueGolden,
final training, Gemma-level generation, or open-domain reasoning.

## Artifact Root

```text
target/pilot_wave/e127_overnight_text_skill_farm_orange_cycle/
```

Each cycle writes:

```text
cycles/cycle_###/
```

## Continuous Progress Contract

The runner must write:

```text
progress.jsonl
partial_aggregate_snapshot.json
cycles/cycle_###/progress.jsonl
cycles/cycle_###/partial_aggregate_snapshot.json
```

During scan, progress is written every `--chunk-rows` dataset rows. During
promotion, progress is written per selected candidate.

## Stop Contract

The runner stops at a cycle boundary if this file exists:

```text
target/pilot_wave/e127_overnight_text_skill_farm_orange_cycle/STOP
```

## Per-Cycle Required Artifacts

```text
run_manifest.json
candidate_pool_report.json
operator_cards.json
operator_gold_results.json
operator_orange_results.json
variant_report.json
mutation_summary.json
aggregate_metrics.json
deterministic_replay.json
decision.json
summary.json
row_level_samples.jsonl
candidate_examples.jsonl
report.md
```

## Safety Rules

Every Orange/Legendary candidate must have:

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

No E127 cycle may promote to Core, PermaCore, or TrueGolden.

# E123 Orange Baseline FineWeb New Skill Discovery Probe Contract

## Goal

Run a new skill-discovery scan after E122, using the scoped orange-only active
Operator baseline and the planner-only Negative Knowledge Cards.

Core question:

```text
After the active library is clean orange-only, do real FineWeb rows still expose
repeated under-covered skill candidates?
```

This is discovery only. E123 must not promote any skill.

## Inputs

```text
dataset:
  data/high_quality_seed_v1/fineweb_edu/local_fineweb_edu_sample_100000.jsonl

orange baseline:
  target/pilot_wave/e122_orange_only_baseline_and_negative_card_recall_probe/
```

## Candidate Status

```text
CoveredByOrangeBaseline:
  support exists, but orange library coverage is sufficient

NewFarmCandidate:
  support is high and orange coverage remains low

Watch:
  signal exists, but evidence is not yet enough for farming
```

## Negative Card Rule

Negative Knowledge Cards remain planner-only:

```text
normal_router_callable_cards = 0
negative_card_false_block_count = 0
```

They may block unsafe mutation variants, but may not become answer-generating
operators.

## Required Artifacts

```text
run_manifest.json
dataset_report.json
orange_library_report.json
candidate_discovery_report.json
negative_card_interaction_report.json
text_io_probe_report.json
row_level_samples.jsonl
candidate_examples.jsonl
progress.jsonl
partial_aggregate_snapshot.json
aggregate_metrics.json
deterministic_replay.json
decision.json
summary.json
report.md
checker_summary.json
```

## Decision Labels

```text
e123_new_skill_candidates_found_after_orange_baseline
e123_orange_baseline_covers_candidate_space
e123_only_watch_level_candidates_found
```

## Pass Requirements

```text
rows_seen >= 10000
active_operator_count = 144
orange_only_confirmed = true
negative_card_count > 0
negative_card_false_block_count = 0
normal_router_callable_cards = 0
progress artifacts present
deterministic replay passes
checker failure_count = 0
```

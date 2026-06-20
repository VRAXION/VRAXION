# E114 FineWeb Next Limit Stability Projection Contract

## Purpose

E114 extends E113 from the 100k local FineWeb-Edu JSONL seed pack to the local
FineWeb-Edu parquet source. It keeps the E113 selected recycle policy and
streams a larger row budget while writing chunk-level trend artifacts.

Core questions:

```text
1. Does the E113 recycle policy stay clean as more FineWeb rows arrive?
2. Do hard negatives or neutral waste reappear over time?
3. Would full local FineWeb naturally provide enough qualified activation for
   the next PermaCore-probation target?
4. Which operators remain too rare and need targeted pressure data?
```

## Inputs

```text
source_alias = private_fineweb_edu_local
e112_root = target/pilot_wave/e112_gold_to_core_prune_heavy_probation_wave/
e113_root = target/pilot_wave/e113_fineweb_light_stress_hard_mutation_recycle/
```

## Default Run

```text
limit = 1,000,000 kept FineWeb rows
chunk_rows = 100,000
filters:
  language = en
  score >= 3.0
  64 <= token_count <= 2048
heartbeat = 20s
```

## Required Gates

```text
selected_hard_negative_total = 0
selected_neutral_waste_total = 0
chunk trend written
target sufficiency projection written
deterministic replay passes
checker failure_count = 0
no gradient/backprop/optimizer
```

## Required Artifacts

```text
run_manifest.json
source_inventory.json
operator_projection_report.json
stability_trend_report.json
target_sufficiency_report.json
chunk_trend.jsonl
progress.jsonl
partial_aggregate_snapshot.json
aggregate_metrics.json
deterministic_replay.json
decision.json
summary.json
report.md
row_level_samples.jsonl
checker_summary.json
```

## Boundary

This is a next-limit projection and stability stress test. It is not
PermaCore/TrueGolden promotion and not final training.

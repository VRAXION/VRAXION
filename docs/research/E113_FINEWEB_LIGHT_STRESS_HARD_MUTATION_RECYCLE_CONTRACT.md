# E113 FineWeb Light Stress Hard Mutation Recycle Contract

## Purpose

E113 runs a light dataset-backed stress pass over the E112 CoreMemoryCandidate
operator pool using the local FineWeb-Edu 100k seed pack.

This is not final training, not PermaCore promotion, and not TrueGolden
promotion. It tests whether scoped operators remain safe on real web-text rows
and whether hard scope-prune/recycle copies improve unsafe or wasteful baseline
behavior.

## Inputs

```text
data/high_quality_seed_v1/fineweb_edu/local_fineweb_edu_sample_100000.jsonl
target/pilot_wave/e112_gold_to_core_prune_heavy_probation_wave/
```

## Compared Variants

```text
current_core_candidate_baseline
hard_scope_prune_copy
recycle_repair_copy
negative_scope_sentinel_copy
```

## Required Gates

```text
selected_hard_negative_total = 0
deterministic replay passes
checker failure_count = 0
progress artifacts exist
mutation/recycle pressure recorded
no gradient/backprop/optimizer
```

## Required Artifacts

```text
run_manifest.json
dataset_report.json
operator_stress_results.json
mutation_variant_report.json
mutation_events.jsonl
mutation_summary.json
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

FineWeb-Edu rows are used as real-text stress inputs. The probe does not claim
open-domain reasoning, assistant readiness, final training success, PermaCore,
or TrueGolden status.

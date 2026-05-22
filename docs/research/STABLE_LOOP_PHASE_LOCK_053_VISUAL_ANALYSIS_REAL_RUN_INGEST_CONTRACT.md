# STABLE_LOOP_PHASE_LOCK_053_VISUAL_ANALYSIS_REAL_RUN_INGEST Contract

## Summary

053 connects the 052 schema-first visual lab to a real bounded research run
artifact shape. It ingests the already-positive 049 adversarial frozen-eval
metrics and control outcomes into `visual_snapshot_v1` so the viewer can inspect
checkpoint evolution, route-health projection, shortcut-control pruning, and
metric alignment.

This is visual infrastructure only. It is not a new training result, model
capability claim, production API, public beta API, full VRAXION claim, language
grounding claim, or consciousness claim.

## Scope

```text
049 metrics/control artifacts
  -> research-only ingest adapter
  -> visual_snapshot_v1 graph/timeline projection
  -> committed 053 real-run ingest sample
  -> tools/visual_lab topology/playback/diff/metrics validation
```

The 049 runner did not emit internal topology snapshots. 053 therefore creates
a visual projection from real 049 arm metrics and known failure controls. It
does not assert that the projected graph is a raw internal model graph.

## Required Inputs

```text
target/pilot_wave/stable_loop_phase_lock_049_adversarial_frozen_eval_scale/smoke/metrics.jsonl
target/pilot_wave/stable_loop_phase_lock_049_adversarial_frozen_eval_scale/smoke/summary.json
target/pilot_wave/stable_loop_phase_lock_049_adversarial_frozen_eval_scale/smoke/leakage_audit.jsonl
```

The ingest path must require these 049 arms in `metrics.jsonl`:

```text
NO_ROUTE_GRAMMAR_ADVERSARIAL_FROZEN_BASELINE
FROZEN_EVAL_048_REFERENCE
ADVERSARIAL_FROZEN_ROUTE_GRAMMAR_TRAIN_AND_INFER
ADVERSARIAL_FROZEN_ROUTE_GRAMMAR_ROLLBACK_GATED
```

## Required Outputs

```text
visual/run_manifest.json
visual/schema_version.json
visual/checkpoint_index.jsonl
visual/metrics.jsonl
visual/mutation_events.jsonl
visual/route_traces.jsonl
visual/pocket_summaries.jsonl
visual/graph/checkpoint_000.json
visual/graph/checkpoint_050.json
visual/graph/checkpoint_100.json
visual/ticks/checkpoint_100_tick_000.json
visual/ticks/checkpoint_100_tick_001.json
```

053 also commits a tiny reusable sample at:

```text
docs/research/visual_samples/053_real_run_ingest/
```

## Viewer Requirements

```text
load 053 committed sample
display 053 sample in the visual lab by default
keep 052 smoke sample as a fixture
diff checkpoint_000 -> checkpoint_100
added_edges > 0
pruned_edges > 0
retained_edges > 0
show metrics aligned to 049 heldout/OOD/family/hard/long-OOD values
```

## Guardrails

```text
schema_version = visual_snapshot_v1
visual_export remains research-only / doc-hidden
tools/visual_lab remains isolated from normal Rust builds
do not commit target/
do not rerun 049 as part of frontend validation
do not make new model/training claims
do not expose visual_export as production API
```

## Positive Verdicts

```text
REAL_RUN_VISUAL_INGEST_POSITIVE
REAL_RUN_SAMPLE_BUNDLE_WRITTEN
049_METRIC_ALIGNMENT_POSITIVE
REAL_RUN_TOPOLOGY_VIEW_POSITIVE
REAL_RUN_PLAYBACK_VIEW_POSITIVE
REAL_RUN_DIFF_VIEW_POSITIVE
VIEWER_REAL_RUN_COMPATIBLE
NON_BRITTLE_REAL_RUN_INGEST_POSITIVE
PRODUCTION_API_NOT_READY
```

## Boundary

053 supports visual ingestion of the bounded 049 adversarial frozen-eval result.
It does not support production readiness, public beta API, full VRAXION,
language grounding, biological/FlyWire equivalence, physical quantum behavior,
or consciousness.

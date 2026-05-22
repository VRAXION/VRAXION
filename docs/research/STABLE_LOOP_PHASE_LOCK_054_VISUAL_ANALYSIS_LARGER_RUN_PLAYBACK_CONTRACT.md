# STABLE_LOOP_PHASE_LOCK_054_VISUAL_ANALYSIS_LARGER_RUN_PLAYBACK Contract

## Summary

054 validates the visual lab against a larger deterministic playback bundle. It
is a V1+ visual stress/readability/playback probe, not a model experiment and
not a production dashboard claim.

The repo already contains `053_VISUAL_ANALYSIS_REAL_RUN_INGEST`, so 054 keeps
its number and intentionally tests larger deterministic playback after real-run
ingest.

## Scope

```text
deterministic visual_snapshot_v1 larger bundle
  -> bounded committed 054 sample
  -> bundle selector
  -> checkpoint playback
  -> tick playback
  -> event timeline
  -> first/previous diff modes
  -> render metadata reporting
```

## Required Sample

```text
docs/research/visual_samples/054_larger_playback_smoke/
```

The committed sample must remain bounded and must include:

```text
schema_version = visual_snapshot_v1
12 checkpoints: 000, 010, ..., 110
at least 2 tick snapshots for selected checkpoints
larger-but-bounded graph
mutation/prune/repair/crystallize events
non-empty first -> final diff
```

## Viewer Requirements

```text
explicit bundle selector
available bundles include 052_smoke_minimal, 053_real_run_ingest, 054_larger_playback_smoke
checkpoint slider follows selected bundle
tick slider appears when selected checkpoint has ticks
event list changes with selected checkpoint/tick
diff modes include first -> selected and previous -> selected
render metadata records graph counts and render duration
```

## Guardrails

```text
do not commit target/
do not commit node_modules or .svelte-kit
do not add PixiJS or custom WebGL in 054
do not bump visual_snapshot_v1 unless required
do not claim new model capability
do not claim production visualization readiness
```

## Verdicts

Positive:

```text
LARGER_PLAYBACK_VISUAL_EXPORT_POSITIVE
LARGER_PLAYBACK_SAMPLE_WRITTEN
BUNDLE_SELECTOR_POSITIVE
CHECKPOINT_PLAYBACK_POSITIVE
TICK_PLAYBACK_POSITIVE
EVENT_TIMELINE_POSITIVE
DIFF_LARGER_RUN_POSITIVE
VIEWER_LARGE_GRAPH_SMOKE_POSITIVE
RENDER_METADATA_RECORDED
LARGER_SAMPLE_SIZE_BOUNDED
PRODUCTION_API_NOT_READY
```

Failure:

```text
LARGER_PLAYBACK_VISUAL_EXPORT_FAILS
BUNDLE_SELECTOR_FAILS
CHECKPOINT_PLAYBACK_FAILS
TICK_PLAYBACK_FAILS
EVENT_TIMELINE_FAILS
DIFF_LARGER_RUN_FAILS
RENDER_METADATA_MISSING
COMMITTED_SAMPLE_TOO_LARGE
VIEWER_LARGE_GRAPH_SMOKE_FAILS
```

## Boundary

054 is visual playback infrastructure only. It does not support production
dashboard readiness, production API readiness, public beta promotion, new model
capability, full VRAXION, language grounding, biological/FlyWire equivalence,
physical quantum behavior, or consciousness.

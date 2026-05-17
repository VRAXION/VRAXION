# STABLE_LOOP_PHASE_LOCK_052_VISUAL_ANALYSIS_PACKAGE Contract

## Summary

052 introduces a schema-first visual analysis package for INSTNCT /
route-grammar research runs. V1 adds a Rust exporter, a tiny committed sample
bundle, and an isolated SvelteKit visual lab.

This is visual infrastructure only. It is not a new training result, model
capability claim, production API, public beta API, full VRAXION claim, language
grounding claim, or consciousness claim.

## Architecture

```text
instnct-core visual_export
  -> visual_snapshot_v1 JSON / JSONL artifacts
  -> committed smoke sample
  -> tools/visual_lab SvelteKit viewer
```

The schema is the contract. The viewer must not depend on one probe's internal
Rust structs.

## Required Artifacts

```text
visual/run_manifest.json
visual/schema_version.json
visual/checkpoint_index.jsonl
visual/metrics.jsonl
visual/mutation_events.jsonl
visual/route_traces.jsonl
visual/pocket_summaries.jsonl
visual/graph/checkpoint_*.json
visual/ticks/checkpoint_*_tick_*.json
```

V1 uses plain JSON/JSONL. Compression is deliberately deferred.

## Required Sample Contents

```text
1 highway chain
2 side pockets
1 candidate edge
1 pruned edge
1 active route trace
1 mutation event
1 prune event
checkpoint_000
checkpoint_010
at least one tick file
```

## Required Viewer Capabilities

```text
overview page
topology page
playback page
diff page
metrics page
zoom / pan through Sigma.js
node / edge / pocket selection
role filters
checkpoint slider
tick slider when tick artifacts exist
route trace highlighting
basic before/after diff
```

## Guardrails

```text
schema_version = visual_snapshot_v1
stable node ids
stable edge ids
optional fields tolerated
wrong schema_version rejected
visual_export remains research-only / doc-hidden
tools/visual_lab remains isolated from normal Rust builds
no PixiJS in V1
no production API claim
```

## Positive Verdicts

```text
VISUAL_SCHEMA_EXPORT_POSITIVE
SMOKE_VISUAL_BUNDLE_WRITTEN
TOPOLOGY_VIEW_POSITIVE
PLAYBACK_VIEW_POSITIVE
DIFF_VIEW_POSITIVE
POCKET_VISUALIZATION_POSITIVE
VIEWER_SCHEMA_COMPATIBLE
OPTIONAL_FIELD_TOLERANCE_POSITIVE
NON_BRITTLE_EXPORT_LAYER_POSITIVE
PRODUCTION_API_NOT_READY
```

## Boundary

052 is visual infrastructure only. It does not support production readiness,
public beta API, full VRAXION, language grounding, or consciousness.

# STABLE_LOOP_PHASE_LOCK_055_VISUAL_SECTION_CLOSURE_REAL_RUN_REPLAY Contract

## Summary

055 closes Visual V1 by replaying a real research-result projection in the
visual lab. It uses 049/050 adversarial frozen-eval evidence through the
existing visual ingest path and writes a committed `visual_snapshot_v1` bundle.

This is a real-metric visual projection, not raw internal topology capture.
It is not a new training result, not production dashboard readiness, not
production API readiness, not full VRAXION, not language grounding, and not
consciousness.

## Required Bundle

```text
docs/research/visual_samples/055_real_run_replay_closure/
```

Required contents:

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

The bundle must use:

```text
schema_version = visual_snapshot_v1
run_id = stable_loop_phase_lock_055_real_run_replay_closure
```

## Metric Alignment

Metric alignment is strict but float-safe:

```text
float epsilon <= 1e-9
exact equality for booleans, counts, IDs, arm names, and schema_version
```

Required checkpoints:

```text
checkpoint 000:
  source_arm = NO_ROUTE_GRAMMAR_ADVERSARIAL_FROZEN_BASELINE
  heldout_score = 0.060546875
  ood_score = 0.048828125
  family_min_accuracy = 0
  unique_output_count = 1 / 75
  top_output_rate = 1.000
  collapse_detected = true

checkpoint 050:
  source_arm = FROZEN_EVAL_048_REFERENCE
  heldout_score = 0.166015625
  ood_score = 0.15625
  family_min_accuracy = 0
  unique_output_count = 4 / 75
  top_output_rate = 0.8935546875
  collapse_detected = true

checkpoint 100:
  source_arm = ADVERSARIAL_FROZEN_ROUTE_GRAMMAR_TRAIN_AND_INFER
  heldout_score = 1.000
  ood_score = 1.000
  family_min_accuracy = 1.000
  hard_distractor_accuracy = 1.000
  long_ood_accuracy = 1.000
  unique_output_count = 75 / 75
  top_output_rate = 0.0732421875
  majority_output_rate = 0.0546875
  output_entropy = 5.40437231483324
  collapse_detected = false
```

The rounded result-doc shorthand `top_output_rate ~= 0.073` refers to the exact
source value `0.0732421875`.

## Viewer Requirements

The bundle selector must include all committed visual bundles:

```text
052_smoke_minimal
053_real_run_ingest
054_larger_playback_smoke
055_real_run_replay_closure
```

055 must be the default bundle. The viewer must not hard-code behavior to 055;
Topology, Playback, Diff, and Metrics must use the same shared bundle loader as
the earlier samples.

## Closure Checker

Required static check:

```powershell
python scripts/probes/run_stable_loop_phase_lock_055_visual_closure_check.py --check-only
```

The checker reads committed files only. It must fail if the 055 bundle is
missing, schema is wrong, 055 is not default, event coverage is incomplete,
metric alignment fails, shared loader paths are missing, or claim boundary text
is missing.

## Boundary

055 supports Visual V1 closure, schema-first visual lab replay, tiny sample,
larger playback, real-result replay, metric alignment against bounded 049/050
evidence, demo docs, and static closure checking.

055 does not support production dashboard, production API, public beta
promotion, new model capability, raw internal model graph capture, full
VRAXION, language grounding, biological/FlyWire equivalence, physical quantum
behavior, or consciousness.

# E116 Alpha-Weave Synthetic Pressure Generation Contract

## Purpose

E114 showed that natural FineWeb is clean but too sparse for 77 rare/scoped
Operators. E115 locked the `AlphaWeavePressureCell-v1` schema.

E116 generates synthetic alpha-Weave pressure cells for those 77 operators and
runs targeted activation accounting.

Core question:

```text
Can synthetic, schema-valid, no-leak alpha-Weave pressure cells provide enough
targeted activation for the E114 FineWeb-rare operators to reach the next
300k probation activation limit?
```

## Synthetic Origin Rule

Every generated cell must disclose its origin in `training_metadata`:

```text
data_origin = synthetic_codex_generated
generator = codex
generator_version = e116_alpha_weave_generator_v1
human_review_status = unreviewed
synthetic_disclosure = true
```

This origin marker must not appear in `public_input`.

## Inputs

```text
schema = docs/research/ALPHA_WEAVE_PRESSURE_CELL_SCHEMA_V1.json
rare operators = target/pilot_wave/e114_fineweb_next_limit_stability_projection/operator_projection_report.json
```

## Required Gates

```text
schema_failure_count = 0
public_leak_failure_count = 0
synthetic_origin_metadata_rate = 1
synthetic_origin_public_leak_rate = 0
target_reach_count = rare_operator_count
targeted_needed_remaining_count = 0
deterministic replay passes
checker failure_count = 0
```

## Required Artifacts

```text
run_manifest.json
generation_manifest.json
synthetic_origin_report.json
rare_operator_input_report.json
generated_cells.jsonl
operator_target_coverage.json
activation_projection_report.json
leakage_check_report.json
public_sample_cells.json
aggregate_metrics.json
deterministic_replay.json
decision.json
summary.json
partial_aggregate_snapshot.json
progress.jsonl
human_machine_sample_report.md
report.md
checker_summary.json
```

## Boundary

This is synthetic pressure-data generation and activation accounting only.
It is not final training, not PermaCore/TrueGolden promotion, and not proof that
the runtime has learned the generated examples.

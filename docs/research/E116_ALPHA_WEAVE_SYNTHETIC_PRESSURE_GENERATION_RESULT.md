# E116 Alpha-Weave Synthetic Pressure Generation Result

```text
decision = e116_synthetic_pressure_reaches_next_activation_limit
checker_failure_count = 0
schema_version = AlphaWeavePressureCell-v1
```

## Result

E116 generated synthetic alpha-Weave pressure cells for the 77 E114 FineWeb-rare
operators.

```text
rare_operator_count = 77
generated_cell_packs = 9,856
variant_count = 118,272
scheduled_case_count = 15,257,088
target_reach_count = 77
targeted_needed_remaining_count = 0
schema_failure_count = 0
public_leak_failure_count = 0
synthetic_origin_metadata_rate = 1.000000
synthetic_origin_public_leak_rate = 0.000000
```

## Template Coverage

```text
evidence_conflict = 38 operators
task_progress = 13 operators
answer_integrity = 11 operators
alias_symbol = 9 operators
frame_sync = 6 operators
```

## Interpretation

The generated curriculum is sufficient, under synthetic targeted-pressure
accounting, to push all 77 FineWeb-rare operators past the 300k probation
activation limit.

This is not a PermaCore/TrueGolden promotion. The result means the next run has
the needed targeted synthetic pressure data available. Promotion still requires
the later no-harm, replay, negative-scope, reload, challenger, and human-review
gates.

## Synthetic Disclosure

Every generated cell includes:

```text
training_metadata.data_origin = synthetic_codex_generated
training_metadata.generator = codex
training_metadata.generator_version = e116_alpha_weave_generator_v1
training_metadata.human_review_status = unreviewed
training_metadata.synthetic_disclosure = true
```

The synthetic marker is intentionally absent from `public_input`.

Boundary: synthetic pressure-data generation and activation accounting only.
No final-training, PermaCore, TrueGolden, or open-domain reasoning claim.

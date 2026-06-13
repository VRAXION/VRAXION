# E56A Text Field Byte Frame Size And Overlap Sweep Result

Status: completed and checker validated.

## Decision

```text
decision = e56a_overlap_required_for_boundary_robustness
checker_failure_count = 0
sample_only_checker_passed = true
run_id = d8a315e59a2b8b77
gradient_descent_used = false
optimizer_used = false
backprop_used = false
```

## Primary Comparison

| system | stress success | boundary split | adversarial contrast | real-like weak | bytes/decision |
|---|---:|---:|---:|---:|---:|
| legacy_direct_text_ingress_baseline | 0.424861 | 0.000000 | 0.124306 | 0.000000 | 0 |
| text_field_single_128 | 0.600000 | 0.000000 | 1.000000 | 1.000000 | 128 |
| text_field_single_256 | 0.850556 | 1.000000 | 1.000000 | 1.000000 | 256 |
| text_field_single_512 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 512 |
| text_field_4x128_overlap0 | 0.758889 | 0.000000 | 1.000000 | 1.000000 | 512 |
| text_field_4x128_overlap32 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 512 |
| text_field_4x128_overlap64 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 512 |
| keyword_shortcut_control | 0.065625 | 0.000000 | 0.032639 | 0.295486 | 0 |
| oracle_text_field_reference | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 2048 |

## Recommendation

```text
best_system = text_field_single_512
recommended_default = text_field_4x128_overlap32
stress_gain_vs_legacy = 0.575139
overlap_gain_vs_no_overlap = 0.241111
```

## Interpretation

The Text Field / Byte Field improves the E55 text ingress frontier by giving
Text Lens pockets a local raw UTF-8 byte matrix instead of forcing a single
direct parser path. Overlap matters when evidence spans a frame boundary.

The score winner is the `512 x 8` single frame. The recommended default is
`4 x 128 x 8` with `32` byte overlap because it matches the ceiling on this
sweep while preserving local 128-byte lens work and boundary robustness.

## Boundary

E56A is a controlled Text Field / Byte Field ingress sweep. It tests whether
raw UTF-8 byte frames improve text evidence extraction before the monolith
integration. It does not claim raw language reasoning, AGI, consciousness,
deployment quality, or model-scale behavior.

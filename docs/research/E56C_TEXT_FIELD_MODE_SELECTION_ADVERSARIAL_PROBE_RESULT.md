# E56C Text Field Mode Selection Adversarial Probe Result

Status: completed and checker validated.

## Decision

```text
decision = e56c_three_mode_agency_selector_adversarial_confirmed
checker_failure_count = 0
sample_only_checker_passed = true
run_id = 99f178f7ba8036c9
gradient_descent_used = false
optimizer_used = false
backprop_used = false
```

## Overall Systems

| system | success | mode accuracy | false commit | overpay | mean cost | net utility |
|---|---:|---:|---:|---:|---:|---:|
| always_fast_default | 0.375000 | 0.375000 | 0.625000 | 0.000000 | 1.000 | -0.531250 |
| always_long_capped | 0.500000 | 0.125000 | 0.500000 | 0.375000 | 2.750 | -0.456250 |
| always_clean_long | 0.750000 | 0.250000 | 0.250000 | 0.500000 | 4.500 | 0.015000 |
| naive_length_router | 0.632065 | 0.482190 | 0.367935 | 0.149875 | 2.958 | -0.063088 |
| clean_long_without_cost_guard | 1.000000 | 0.500000 | 0.000000 | 0.500000 | 3.538 | 0.675625 |
| three_mode_agency_router | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 2.006 | 0.939844 |
| oracle_mode_selector | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 2.006 | 0.939844 |
| random_mode_control | 0.467459 | 0.249660 | 0.344373 | 0.217799 | 2.218 | -0.276177 |

## Adversarial Subset

| system | adversarial success | adversarial mode accuracy | adversarial false commit | overall net utility |
|---|---:|---:|---:|---:|
| always_fast_default | 0.250000 | 0.250000 | 0.750000 | -0.531250 |
| always_long_capped | 0.250000 | 0.000000 | 0.750000 | -0.456250 |
| always_clean_long | 0.500000 | 0.250000 | 0.500000 | 0.015000 |
| naive_length_router | 0.264130 | 0.014130 | 0.735870 | -0.063088 |
| clean_long_without_cost_guard | 1.000000 | 0.750000 | 0.000000 | 0.675625 |
| three_mode_agency_router | 1.000000 | 1.000000 | 0.000000 | 0.939844 |
| oracle_mode_selector | 1.000000 | 1.000000 | 0.000000 | 0.939844 |
| random_mode_control | 0.373505 | 0.249615 | 0.501313 | -0.276177 |

## Recommendation

```text
recommended_policy = three_mode_agency_router
fast_default = 4x128 overlap32
long_capped = 5x256 overlap64
clean_long = 4x512 overlap128
lock_statement = Do not lock one universal Text Field max. Lock three mechanically validated modes and require Agency/Router selection with evidence, coverage, integrity, and cost guards.
```

## Interpretation

The adversarial rows show why a single universal Text Field max is the wrong
lock. Always-clean mode can answer many rows, but it overpays on short and
long-lure rows and still commits incorrectly when the correct behavior is
ASK/MULTI_CYCLE. A length-only router is also insufficient because long input
does not imply long-context reasoning is needed.

The clean result is the Agency-selected mode policy: use fast/default when the
evidence footprint is local, long-capped when the needed footprint fits under
the 3x budget, clean-long only when integrity/coverage requires it, and
ASK/MULTI_CYCLE when visible evidence or single-frame capacity is insufficient.

## Boundary

E56C is a deterministic adversarial Text Field mode-selection probe. It tests
whether fast/default, long-capped, and clean-long Text Field modes should be
selected by Agency/Router policy rather than locked as one universal max. It
does not claim raw language reasoning, AGI, consciousness, deployment quality,
or model-scale behavior.

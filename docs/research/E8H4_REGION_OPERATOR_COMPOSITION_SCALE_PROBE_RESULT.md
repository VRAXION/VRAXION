# E8H4 Region Operator Composition Scale Probe Result

Status: completed.

## Decision

```text
decision = e8h4_region_operator_composition_scale_positive
best_learned_system = mutation_discovered_plus_prune
deterministic_replay_passed = true
checker_failure_count = 0
```

Official run root:

```text
target/pilot_wave/e8h4_region_operator_composition_scale_probe/
```

Evidence configuration:

```text
seeds = 108401..108412
cpu_workers = 12
route_lengths = 1,3,6,12,24
grid_sizes = 6,8 with OOD larger-grid rows
population = 16
generations = 16
```

## System Summary

| system | usefulness | trace validity | answer accuracy | drift slope |
|---|---:|---:|---:|---:|
| identity_noop_baseline | 0.586 | 0.773 | 0.252 | 0.015 |
| direct_overwrite_matrix_baseline | 0.767 | 0.883 | 0.551 | 0.006 |
| mutation_discovered_single_operator | 0.701 | 0.837 | 0.466 | 0.020 |
| mutation_discovered_composed_6_step | 0.814 | 0.899 | 0.664 | 0.006 |
| mutation_discovered_composed_12_step | 0.831 | 0.907 | 0.697 | 0.005 |
| mutation_discovered_composed_24_step | 0.841 | 0.911 | 0.712 | 0.004 |
| mutation_discovered_plus_trace_check | 0.827 | 0.898 | 0.696 | 0.005 |
| mutation_discovered_plus_prune | 0.854 | 0.923 | 0.729 | 0.005 |
| reusable_operator_library_router | 0.827 | 0.898 | 0.696 | 0.005 |
| random_region_rule_control | 0.542 | 0.667 | 0.307 | 0.009 |
| dense_transform_danger_control | 0.858 | 0.782 | 1.000 | 0.010 |
| answer_shortcut_control | 0.858 | 0.782 | 1.000 | 0.010 |
| handcoded_oracle_region_operator_reference | 1.000 | 1.000 | 1.000 | 0.000 |

The dense/answer shortcut controls have slightly higher usefulness than the
best learned system because they exploit final answer cues, but their trace
validity is far lower. They are invalid as primary successes.

## Route Length Scaling

Best learned system: `mutation_discovered_plus_prune`.

| route length | usefulness | trace validity | drift slope | first divergence step |
|---:|---:|---:|---:|---:|
| 1 | 0.923 | 0.954 | 0.000 | 0.777 |
| 3 | 0.877 | 0.930 | 0.013 | 1.907 |
| 6 | 0.771 | 0.901 | 0.009 | 2.988 |
| 12 | 0.834 | 0.906 | 0.004 | 6.195 |
| 24 | 0.858 | 0.920 | -0.002 | 10.998 |

Depth gates:

```text
depth6_ok = true
depth12_ok = true
depth24_ok = true
drift_explosive = false
```

## Split Robustness

Best learned system: `mutation_discovered_plus_prune`.

| split | usefulness | trace validity | answer accuracy |
|---|---:|---:|---:|
| validation | 0.873 | 0.929 | 0.766 |
| heldout | 0.857 | 0.927 | 0.722 |
| OOD | 0.856 | 0.928 | 0.715 |
| counterfactual | 0.864 | 0.929 | 0.741 |
| adversarial | 0.822 | 0.902 | 0.699 |

The OOD/counterfactual/adversarial splits did not collapse.

## Operator Reuse

All discovered operator skills were reused across all 12 seeds.

| skill | calls |
|---|---:|
| cleanup | 4512 |
| shift_right | 2916 |
| bind_marker | 2820 |
| fill_gap | 2580 |
| threshold_center | 2532 |
| shift_down | 1668 |
| clear_border | 1956 |
| invert_center | 396 |

## Interpretation

E8H4 gives a clean controlled-proxy positive for the new abstraction:

```text
pocket = detector/condition + direct region transform over the shared Flow grid
```

This performed better than identity, random rules, and the direct overwrite
baseline while preserving trace validity across longer composed routes. Pruning
improved the discovered operator library, suggesting that smaller read/write
footprints helped rather than hurt.

The important caveat is that answer-only shortcut controls can still produce
high answer usefulness while tracing poorly. The positive result is therefore
about trace-valid learned region operators, not final-answer shortcutting.

## Recommended Next Step

Run a stricter operator-library falsification:

```text
E8H5_REGION_OPERATOR_LIBRARY_TRANSFER_AND_NOISY_ROUTE_PROBE
```

Core question:

```text
Do discovered region operators transfer to new operator mixtures and noisy or
partially wrong routes without re-running mutation discovery from scratch?
```

Boundary: E8H4 is a controlled binary Flow-grid region-operator proxy only. It
does not prove raw-language reasoning, AGI, consciousness, deployed-model
behavior, or model-scale behavior.

# E7A13A Capture Radius Atlas Result

## Decision

```text
decision = e7a13a_capture_radius_measured
classification = smooth_falloff
center_eval_accuracy = 0.950000
shell_count = 48
repair_run_count = 96
deterministic_replay_passed = true
checker_failure_count = 0
```

Artifact root:

```text
target/pilot_wave/e7a13a_capture_radius_atlas/
```

## Falloff Fit

Best falloff model:

```text
power_law_falloff
params = {a: 0.8, b: 0.75}
sse = 0.198756432728
```

The fitted average recovery curve is a smooth falloff, not an unlimited repair basin.

## Distance Summary

| distance | recovery rate | solve rate | mean repair gain | mean final eval |
|---:|---:|---:|---:|---:|
| 0.0000 | 1.000000 | 1.000000 | -0.002995 | 0.947005 |
| 0.0050 | 0.916667 | 1.000000 | 0.004427 | 0.945052 |
| 0.0100 | 0.416667 | 0.916667 | 0.005729 | 0.931250 |
| 0.0200 | 0.500000 | 0.666667 | 0.010286 | 0.917839 |
| 0.0500 | 0.416667 | 0.666667 | 0.039583 | 0.819792 |
| 0.1000 | 0.333333 | 0.500000 | 0.220182 | 0.835286 |
| 0.2000 | 0.416667 | 0.583333 | 0.148828 | 0.867578 |
| 0.4000 | 0.166667 | 0.500000 | 0.221224 | 0.796745 |

## Mode And Budget Summary

| corruption mode | budget | mean final | mean gain | recovery | solve |
|---|---:|---:|---:|---:|---:|
| scale_perturbation_shell | 1x | 0.948828 | -0.000977 | 1.000000 | 1.000000 |
| scale_perturbation_shell | 4x | 0.947461 | -0.002344 | 1.000000 | 1.000000 |
| least_sensitive_bit_flip_shell | 1x | 0.933594 | 0.005273 | 0.750000 | 0.875000 |
| least_sensitive_bit_flip_shell | 4x | 0.943750 | 0.015430 | 0.875000 | 1.000000 |
| bits_plus_scale_corruption_shell | 1x | 0.866016 | 0.014453 | 0.500000 | 0.750000 |
| bits_plus_scale_corruption_shell | 4x | 0.908984 | 0.057422 | 0.250000 | 0.875000 |
| random_bit_flip_shell | 1x | 0.862891 | 0.037500 | 0.250000 | 0.625000 |
| random_bit_flip_shell | 4x | 0.891992 | 0.066602 | 0.500000 | 0.625000 |
| most_sensitive_bit_flip_shell | 1x | 0.790039 | 0.058594 | 0.250000 | 0.500000 |
| most_sensitive_bit_flip_shell | 4x | 0.864648 | 0.133203 | 0.375000 | 0.625000 |
| block_corruption_shell | 1x | 0.773242 | 0.249805 | 0.250000 | 0.375000 |
| block_corruption_shell | 4x | 0.859375 | 0.335938 | 0.250000 | 0.500000 |

## Interpretation

The good binary seed has a measurable local repair basin. The basin is strongest for scale perturbations and least-sensitive bit flips, weaker for random bit flips, and fragile under most-sensitive or block-localized bit corruption.

The 4x budget improves several damaged shells, but it does not make recovery universal. This means E7A12's seeded repair result was real, but it is local and distance-sensitive.

## Next Recommendation

Proceed to a focused seed-bridge experiment: progressively reduce reliance on QAT while staying inside the measured repair basin. The repair operator should prioritize sensitivity-aware bit repair and block-aware recovery, because those axes explain most of the basin boundary.

# Phase D32B: Angle-Knob Activation

D32B tests the stronger version of the voltage-knob idea: each active input votes for a direction on a `uint8` angle wheel, then output lanes receive voltage according to angular closeness.

```text
input votes -> resultant angle -> output lane voltages
```

Comparator families:

```text
threshold_scalar
c19_scalar
relu_scalar
swish_scalar
hard_router
angle_knob_hard
angle_knob_soft
angle_knob_top2
fixed_random_angle_control
shuffled_angle_control
```

The pass gate is intentionally small and adversarial. Angle-knob only matters if it solves the routing/mux tasks cleanly, inactive lanes remain empty, controls fail, and its edge-equivalent size is competitive with hard-router.

Telemetry outputs:

```text
angle_telemetry.csv
angle_flow_ascii.txt
activation_results.csv
D32B_ANGLE_KNOB_ACTIVATION_REPORT.md
```

This phase is still a microproof. Passing D32B would justify D33: testing angle-knob neurons inside the B64 C-router.

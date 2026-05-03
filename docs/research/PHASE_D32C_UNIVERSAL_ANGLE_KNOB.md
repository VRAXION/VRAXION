# Phase D32C: Universal Angle-Knob Stress

D32C asks whether the angle-knob is more than a routing trick. It compares sparse/discrete angle-knob candidates against scalar threshold, C19, ReLU, Swish, and hard-router baselines across scalar, logic, routing, and tiny state tasks.

The angle candidate uses `uint8` vote angles, small strengths, optional bias, aperture, and hard/linear curves. The search is deterministic and CPU-only.

Output files:

```text
activation_family_results.csv
task_group_summary.csv
candidate_telemetry.csv
angle_flow_ascii.txt
size_fairness_table.csv
control_results.csv
top_candidates.json
D32C_UNIVERSAL_ANGLE_KNOB_REPORT.md
```

Key verdicts:

```text
D32C_ANGLE_KNOB_NEW_BEST
D32C_ANGLE_KNOB_ROUTING_ONLY
D32C_BEUKERS_COMBO_WINS
D32C_C19_OR_SCALAR_STILL_BEST
D32C_CONTROL_LEAK
```

This is still a scratch microbenchmark, not integration into the production recurrent core. A real pass should lead to D33: B64 C-router with angle-knob neurons.

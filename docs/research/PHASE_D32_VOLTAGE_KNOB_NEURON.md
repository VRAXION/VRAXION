# Phase D32: Voltage-Knob Neuron

D32 tests a neuron-level idea before integrating it into A/B/C/D: a neuron should not only decide how much activation to emit, but also how to split that voltage across outgoing lanes.

```text
incoming charge
      |
      v
[ voltage-knob neuron ]
  |       |       |
 ADD     SUB     MUL
```

The proof compares scalar threshold, scalar C19, hard-router, voltage-knob, and negative controls on deterministic microtasks. A pass only counts if the knob solves the tasks cleanly, inactive lanes stay empty, controls fail, and the size accounting is competitive with hard-router.

Expected output files live under `output/phase_d32_voltage_knob_neuron_20260503/`:

```text
primitive_results.csv
primitive_size_table.csv
primitive_controls.csv
knob_flow_ascii.txt
knob_flow_matrix.csv
D32_VOLTAGE_KNOB_NEURON_REPORT.md
```

Verdict meanings:

```text
D32_VOLTAGE_KNOB_PRIMITIVE_PASS
  the knob is worth testing inside the C-router next

D32_HARD_ROUTER_WINS
  keep explicit hard routing; do not add the knob primitive yet

D32_SCALAR_ACTIVATION_ENOUGH
  threshold/C19 was enough for these tasks

D32_KNOB_CONTROL_LEAK
  the proof task is invalid because controls also solve it
```

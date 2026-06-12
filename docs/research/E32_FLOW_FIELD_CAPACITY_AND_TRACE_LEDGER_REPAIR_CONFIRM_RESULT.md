# E32 Flow Field Capacity And Trace Ledger Repair Confirm Result

Status: complete.

## Decision

```text
decision = e32_capacity_only_repair_confirmed
target_checker_passed = true
sample_only_checker_passed = true
checker_failure_count = 0
primary_run_id = 6178bf37bbd9b614
cpu_confirm_decision = e32_capacity_only_repair_confirmed
cpu_confirm_checker_failure_count = 0
```

## Primary Result

```text
baseline_d96_p8:
  heldout_resolution = 0.747727
  heldout_action = 0.802273
  heldout_trace_exact = 0.672273
  heldout_trace_bit = 0.912216
  wrong_confident = 0.050391

capacity_flow_d192_p8:
  heldout_resolution = 0.810455
  heldout_action = 0.815909
  heldout_trace_exact = 0.938636
  heldout_trace_bit = 0.972614
  wrong_confident = 0.039628

trace_ledger_weighted_d96_p8:
  heldout_resolution = 0.777727
  heldout_trace_exact = 0.780909

trace_ledger_weighted_d192_p8:
  heldout_resolution = 0.809545
  heldout_trace_exact = 0.945000

span_bucket_aux_d96_p8:
  heldout_resolution = 0.802727
  heldout_trace_exact = 0.735000

ingress_event_aux_d96_p8:
  heldout_resolution = 0.814091
  heldout_trace_exact = 0.853636

combined_capacity_aux_d192_p8:
  heldout_resolution = 0.801818
  heldout_trace_exact = 0.917727
```

## Repair Deltas

Against `baseline_d96_p8`:

```text
capacity_flow_d192_p8:
  resolution_delta = +0.062727
  trace_exact_delta = +0.266364
  wrong_confident_delta = -0.010763

trace_ledger_weighted_d192_p8:
  resolution_delta = +0.061818
  trace_exact_delta = +0.272727

ingress_event_aux_d96_p8:
  resolution_delta = +0.066364
  trace_exact_delta = +0.181364

trace_ledger_weighted_d96_p8:
  resolution_delta = +0.030000
  trace_exact_delta = +0.108636

span_bucket_aux_d96_p8:
  resolution_delta = +0.055000
  trace_exact_delta = +0.062727
```

## Interpretation

The cleanest repair for the E31 controlled Trace Ledger break is still larger Flow Field state bandwidth.

Nuance:

```text
ingress_event_aux_d96_p8 had the highest overall heldout resolution,
but it did not close the exact Trace Ledger gap.
```

The strongest trace-stability system was:

```text
trace_ledger_weighted_d192_p8:
  heldout_trace_exact = 0.945000
```

That is only a small trace improvement over plain `capacity_flow_d192_p8`, and it slightly increased wrong-confident rate. The combined capacity+aux branch underperformed plain capacity, so auxiliary objectives can interfere when stacked naively.

## Per-Rung Notes

Controlled R0-R8:

```text
capacity_flow_d192_p8:
  R0 trace = 1.000
  R1 trace = 1.000
  R2 trace = 1.000
  R3 trace = 0.977
  R4 trace = 0.977
  R7 trace = 0.986

trace_ledger_weighted_d192_p8:
  R0 trace = 1.000
  R1 trace = 1.000
  R2 trace = 1.000
  R3 trace = 1.000
  R4 trace = 1.000
  R7 trace = 0.995
```

R9 weak mined real text remained weak:

```text
baseline_d96_p8:
  R9 resolution = 0.232
  R9 trace = 0.005

capacity_flow_d192_p8:
  R9 resolution = 0.300
  R9 trace = 0.445

trace_ledger_weighted_d192_p8:
  R9 resolution = 0.268
  R9 trace = 0.455
```

So E32 repairs the controlled text/trace bottleneck, but it does not solve weak real-text supervision.

## Conclusion

```text
Capacity/state-bandwidth is real.
Trace weighting helps once capacity exists.
Ingress auxiliary helps d96 resolution, but not enough for exact Trace Ledger stability.
Naively combining all auxiliaries is not better.
Weak real text remains a separate data/label/Ingress problem.
```

## Next Step

Use `d192 Flow Field + mild Trace Ledger weighting` as the controlled-text baseline for the next bridge. Do not keep stacking auxiliaries blindly.

Recommended next:

```text
E33_REAL_TEXT_BRIDGE_WITH_E32_CAPACITY_BASELINE
```

Question:

```text
Does the E32 capacity baseline improve FineWeb weak-label behavior,
or is the remaining failure mostly weak supervision / real-text Ingress Codec data quality?
```

Boundary: E32 is a controlled Flow/Pocket repair probe. It does not test raw language reasoning, AGI, consciousness, deployment quality, or model-scale behavior.

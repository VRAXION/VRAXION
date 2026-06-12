# E31 Breakpoint Ladder And Bottleneck Localization Result

Status: complete.

## Decision

```text
decision = e31_no_single_bottleneck_multiple_breaks
target_checker_passed = true
sample_only_checker_passed = true
checker_failure_count = 0
primary_run_id = a4e2bcd40a23e2e5
cpu_confirm_decision = e31_capacity_bottleneck_localized
cpu_confirm_checker_failure_count = 0
```

## Primary Result

```text
baseline_text_ingress_d96_p8:
  heldout_resolution = 0.747273
  heldout_action = 0.804091
  heldout_trace_exact = 0.647273
  heldout_trace_bit = 0.910682
  wrong_confident = 0.040040

capacity_flow_d192_p8:
  heldout_resolution = 0.806818
  heldout_action = 0.815455
  heldout_trace_exact = 0.925455
  heldout_trace_bit = 0.967614

capacity_pockets_d96_p16:
  heldout_resolution = 0.805909
  heldout_action = 0.814091
  heldout_trace_exact = 0.748636
  heldout_trace_bit = 0.941307

oracle_ingress_d96_p8:
  heldout_resolution = 0.795455
  heldout_action = 0.803636
  heldout_trace_exact = 0.791364
  heldout_trace_bit = 0.937727

oracle_evidence_span_d96_p8:
  heldout_resolution = 0.789545
  heldout_action = 0.805455
  heldout_trace_exact = 0.722727
  heldout_trace_bit = 0.917557
```

## Localization

```text
capacity_flow_trace_exact_delta = +0.278182
oracle_ingress_trace_exact_delta = +0.144091
capacity_pocket_trace_exact_delta = +0.101364
oracle_span_trace_exact_delta = +0.075455

capacity_flow_resolution_delta = +0.059545
capacity_pocket_resolution_delta = +0.058636
oracle_ingress_resolution_delta = +0.048182
oracle_span_resolution_delta = +0.042273
```

The strongest clean repair signal was larger Flow Field state bandwidth, not simply more Pocket Operators. The diagnostic oracle ingress also helped, so the primary result is not a single bottleneck. It is a combined Flow Field capacity plus Ingress Codec/Trace Ledger problem.

## Breakpoint Ladder

Notable heldout rungs:

```text
R0_explicit_controlled_evidence:
  baseline_resolution = 0.500000
  baseline_action = 1.000000
  baseline_trace_exact = 0.500000
  capacity_flow_resolution = 1.000000
  oracle_ingress_resolution = 1.000000

R4_decoy_density:
  baseline_trace_exact = 0.386363
  oracle_span_trace_exact = 0.740909

R7_long_context_evidence_span:
  baseline_trace_exact = 0.368182
  oracle_span_trace_exact = 0.945455

R9_mined_real_text_weak_labels:
  baseline_resolution = 0.190909
  baseline_trace_exact = 0.095455
  capacity_flow_resolution = 0.290909
```

Interpretation:

```text
1. The small baseline can choose actions fairly well, but exact Trace Ledger state is brittle.
2. Larger Flow Field capacity repairs much of the controlled trace failure.
3. Span oracle helps especially on decoy-density and long-context rungs, so evidence binding remains a real sub-bottleneck.
4. Weak mined real text remains far below controlled text; that is not solved by capacity alone.
```

## CPU Confirm

A concurrent CPU confirm used a smaller second seed run:

```text
decision = e31_capacity_bottleneck_localized
baseline_heldout_resolution = 0.540833
capacity_flow_heldout_resolution = 0.676667
baseline_trace_exact = 0.370833
capacity_flow_trace_exact = 0.585000
target_checker_passed = true
sample_only_checker_passed = true
```

This confirm supports the capacity/state-bandwidth part of the primary result, but the primary run remains the source of truth because it used the larger evidence configuration.

## Next Step

Do not jump straight to raw real-text training. The next repair should test the strongest localized fix:

```text
E32_FLOW_FIELD_CAPACITY_AND_TRACE_LEDGER_REPAIR_CONFIRM
```

The repair should compare:

```text
baseline d96/p8
d192 Flow Field
d192 Flow Field + trace-ledger auxiliary loss
d192 Flow Field + evidence-span contrastive auxiliary
d192 Flow Field + compact Ingress Codec pretraining
```

The goal is to determine whether the E31 improvement is merely larger hidden capacity, or whether a better Trace Ledger/Ingress Codec objective can reach the same stability with less state bandwidth.

Expected run:

```text
python scripts/probes/run_e31_breakpoint_ladder_and_bottleneck_localization.py \
  --out target/pilot_wave/e31_breakpoint_ladder_and_bottleneck_localization \
  --artifact-sample-dir docs/research/artifact_samples/e31_breakpoint_ladder_and_bottleneck_localization \
  --device auto

python scripts/probes/run_e31_breakpoint_ladder_and_bottleneck_localization_check.py \
  --out target/pilot_wave/e31_breakpoint_ladder_and_bottleneck_localization \
  --artifact-sample-dir docs/research/artifact_samples/e31_breakpoint_ladder_and_bottleneck_localization \
  --write-summary

python scripts/probes/run_e31_breakpoint_ladder_and_bottleneck_localization_check.py \
  --sample-only docs/research/artifact_samples/e31_breakpoint_ladder_and_bottleneck_localization \
  --write-summary
```

Boundary: E31 localizes controlled Flow/Pocket bottlenecks only. It does not claim raw language reasoning, AGI, consciousness, deployment quality, or model-scale behavior.

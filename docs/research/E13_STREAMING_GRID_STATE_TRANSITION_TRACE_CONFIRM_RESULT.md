# E13 Streaming Grid State-Transition Trace Confirm Result

Status: completed.

## Decision

```text
decision = e13_streaming_grid_state_transition_trace_confirmed
next = E14_REGION_AWARE_PARALLEL_POCKET_SCHEDULER_CONFIRM
primary_system = FLOW_GRID_PRUNED_SCHEDULED_POCKET_PRIMARY
positive_gate_passed = true
deterministic_replay_passed = true
checker_failure_count = 0
```

Run root:

```text
target/pilot_wave/e13_streaming_grid_state_transition_trace_confirm/
```

## What Was Tested

E13 tests streaming binary grid-frame transitions:

```text
observed frame stream
-> internal Flow grid state
-> region pocket/operator transition
-> trace-stamped gated writeback
-> repaired grid state and operator trace
```

The primary keeps an internal grid state and rejects invalid trace-stamp frames.
The direct and observed baselines copy observed frames and do not recover an
operator trace.

## Key Metrics

| system | final exact | final sim | trace exact | step op | trace | noisy repair | missing repair | decoy reject | wrong | destructive | cost/tick |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OBSERVED_FRAME_DIFF_BASELINE | 0.778 | 0.994 | 0.000 | 0.000 | 0.985 | 0.000 | 0.000 | 0.000 | 1.000 | 0.038 | 1.669 |
| DIRECT_OVERWRITE_GRID_BASELINE | 0.778 | 0.994 | 0.000 | 0.000 | 0.985 | 0.000 | 0.000 | 0.000 | 1.000 | 0.050 | 5.500 |
| NO_INTERNAL_STATE_BASELINE | 0.778 | 0.994 | 0.000 | 0.000 | 0.985 | 0.000 | 0.000 | 0.000 | 1.000 | 0.038 | 1.669 |
| ORACLE_TRACE_REFERENCE | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| FLOW_GRID_GATED_WRITEBACK | 0.889 | 0.997 | 0.889 | 0.931 | 0.998 | 1.000 | 0.000 | 1.000 | 0.034 | 0.000 | 4.600 |
| FLOW_GRID_TRACE_REPAIR | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 5.200 |
| FLOW_GRID_SCHEDULED_POCKET_PRIMARY | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 3.100 |
| FLOW_GRID_PRUNED_SCHEDULED_POCKET_PRIMARY | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 2.200 |
| TINY_GRID_MLP_CONTROL | 0.014 | 0.943 | 0.014 | 0.059 | 0.952 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 8.901 |

Primary details:

```text
final_grid_exact_accuracy = 1.000
final_grid_similarity = 1.000
operator_trace_exact_accuracy = 1.000
per_step_operator_accuracy = 1.000
frame_transition_accuracy = 1.000
trace_validity = 1.000
delta_validity = 1.000
internal_state_consistency = 1.000
internal_state_vs_observed_noisy_gap = 0.006231
noisy_repair_rate = 1.000
missing_frame_repair_rate = 1.000
decoy_rejection_rate = 1.000
heldout_composition_accuracy = 1.000
long_horizon_survival_rate = 1.000
ood_grid_generalization = 1.000
branch_switch_accuracy = 1.000
licensed_destructive_accept_rate = 1.000
unlicensed_destructive_reject_rate = 1.000
wrong_writeback_rate = 0.000
destructive_overwrite_rate = 0.000
branch_contamination_rate = 0.000
temporal_drift_rate = 0.000
drift_slope_by_horizon = 0.000
cost_per_tick = 2.200
```

Positive-gate deltas:

```text
final_exact_delta_vs_observed = +0.222222
trace_validity_delta_vs_direct = +0.015442
wrong_writeback_reduction_vs_direct = 1.000000
cost_reduction_vs_trace_repair = 0.576923
```

## Split Robustness

The primary passed all generated splits:

```text
validation = 1.000 final exact, 1.000 trace
heldout_composition = 1.000 final exact, 1.000 trace
noisy = 1.000 final exact, 1.000 trace
adversarial_noise = 1.000 final exact, 1.000 trace
missing_frame = 1.000 final exact, 1.000 trace
ood_grid_size = 1.000 final exact, 1.000 trace
long_horizon = 1.000 final exact, 1.000 trace
branch_switch = 1.000 final exact, 1.000 trace
```

Horizon trace validity stayed flat:

```text
1 = 1.000
3 = 1.000
6 = 1.000
12 = 1.000
24 = 1.000
```

## Repair And Safety

```text
internal_state_vs_observed_noisy_gap = +0.006231
noisy_repair_rate = 1.000
missing_frame_repair_rate = 1.000
decoy_rejection_rate = 1.000
licensed_destructive_accept_rate = 1.000
unlicensed_destructive_reject_rate = 1.000
wrong_writeback_rate = 0.000
destructive_overwrite_rate = 0.000
branch_contamination_rate = 0.000
```

The direct overwrite baseline had:

```text
wrong_writeback_rate = 1.000
destructive_overwrite_rate = 0.049750
operator_trace_exact_accuracy = 0.000
```

## Semantic-Leak Audit

```text
runtime_receives_forbidden_semantic_slots = false
debug_names_confined_to_harness_reports = true
no_semantic_slot_leak_detected = true
no_neural_dependency_detected = true
```

Runtime config:

```text
input = binary_grid_frames
state = binary_flow_grid
writeback = gated_region_transform
```

## Interpretation

E13 confirms the synthetic streaming grid transition proxy: a pruned scheduled
pocket runtime can maintain a repaired internal grid state and exact operator
trace across noisy frames, adversarial invalid frames, missing intermediate
frames, heldout compositions, long horizons, OOD grid size, and branch switches.

This result is still a deterministic proxy. It does not prove deployed behavior,
broad model-scale behavior, or hardware latency.

## Verification

```text
python3 scripts/probes/run_e13_streaming_grid_state_transition_trace_confirm.py --out target/pilot_wave/e13_streaming_grid_state_transition_trace_confirm
python3 scripts/probes/run_e13_streaming_grid_state_transition_trace_confirm_check.py --out target/pilot_wave/e13_streaming_grid_state_transition_trace_confirm --write-summary
```

The checker passed with `failure_count = 0`.

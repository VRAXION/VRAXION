# Inferred Pilot Pulse Probe

## Goal

Test whether the Pilot Pulse can be inferred from delayed evidence without explicit pulse-phase input.

The primary arm receives event and cue evidence, predicts wait/commit/reframe/update/hold, carries a hard committed state, and selects the final action from that committed state only.

## Main Results

| Arm | Semantic | Mode | Pulse | State | Action | Wait | Commit Confirm | Reframe | Update | False Commit | Leakage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `inferred_pulse_recursive_model` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.000000` | `0.001805` |
| `explicit_phase_reference` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.000000` | `0.001371` |
| `no_pulse_baseline` | `1.000000` | `1.000000` | `0.660000` | `0.050000` | `0.600000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.000000` | `0.000694` |
| `always_commit_baseline` | `1.000000` | `1.000000` | `0.150000` | `0.350000` | `0.350000` | `0.333333` | `1.000000` | `0.000000` | `0.000000` | `1.000000` | `0.997527` |
| `never_reframe_baseline` | `1.000000` | `1.000000` | `0.750000` | `0.750000` | `0.750000` | `1.000000` | `1.000000` | `0.500000` | `0.000000` | `0.333333` | `0.334157` |
| `static_full_context_baseline` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.000000` | `0.001588` |
| `oracle_state_upper_bound` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.000000` | `0.001609` |
| `shuffled_evidence_order_control` | `1.000000` | `0.500000` | `0.450000` | `0.450000` | `0.650000` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.333333` | `0.333547` |
| `wrong_grounding_control` | `0.910000` | `0.150000` | `0.650000` | `0.660000` | `0.660000` | `0.666667` | `0.000000` | `1.000000` | `0.000000` | `0.000000` | `0.003652` |

## Gaps And Controls

- gap_vs_explicit_phase_reference: `0.000000`
- gap_vs_no_pulse: `0.400000`
- gap_vs_always_commit: `0.650000`
- gap_vs_never_reframe: `0.250000`
- gap_vs_static: `0.000000`

| Control | Mean | Std |
|---|---:|---:|
| `free_run_vs_teacher_forced_gap` | `0.000000` | `0.000000` |
| `shuffled_evidence_drop` | `0.350000` | `0.000000` |
| `teacher_forced_action_accuracy` | `1.000000` | `0.000000` |
| `teacher_forced_state_accuracy` | `1.000000` | `0.000000` |
| `wrong_mode_drop` | `0.971673` | `0.004234` |

## Verdict

```json
{
  "supports_inferred_pilot_pulse": true,
  "supports_wait_inference": true,
  "supports_commit_inference": true,
  "supports_reframe_inference": true,
  "supports_update_inference": true,
  "supports_free_run_committed_state_control": true,
  "explicit_phase_no_longer_required": true,
  "static_baseline_shortcut_detected": true,
  "full_pilot_v1_strengthened": true
}
```

## Interpretation

A positive result means the pulse phase no longer needs to be externally supplied: the model infers wait, commit, reframe, and update from delayed grounding evidence, then acts from the hard committed state.

The static full-context baseline is a label shortcut baseline. It is not counted as a Pilot mechanism unless it also matches committed-state and pulse-control behavior.

## Claim Boundary

Toy evidence only. No consciousness, biology, quantum behavior, natural-language-understanding, full VRAXION, production, or deployment claim.

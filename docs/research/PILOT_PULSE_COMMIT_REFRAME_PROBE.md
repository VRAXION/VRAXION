# Pilot Pulse Commit-Reframe Probe

## Goal

Stress-test whether a toy Pilot can decide when to wait, commit, reframe, update, and act after delayed grounding evidence.

The final action step is generic and receives only a hard one-hot committed state, not event/mode/patient fields or a soft hidden state.

## Main Results

| Arm | Semantic | Mode | State | Action | Pulse | Wait | Reframe | Update | False Commit | Leakage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pilot_pulse_recursive_model` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.000000` | `0.001395` |
| `no_pulse_baseline` | `1.000000` | `1.000000` | `0.050000` | `0.600000` | `0.670000` | `1.000000` | `1.000000` | `1.000000` | `0.000000` | `0.000908` |
| `always_commit_baseline` | `1.000000` | `1.000000` | `0.350000` | `0.350000` | `0.150000` | `0.333333` | `0.000000` | `0.000000` | `1.000000` | `0.997527` |
| `never_reframe_baseline` | `1.000000` | `1.000000` | `0.750000` | `0.750000` | `0.750000` | `1.000000` | `0.500000` | `0.000000` | `0.333333` | `0.334157` |
| `static_full_context_baseline` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.000000` | `0.001162` |
| `oracle_state_upper_bound` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.000000` | `0.001282` |
| `shuffled_pulse_control` | `1.000000` | `0.990000` | `0.990000` | `0.990000` | `0.720000` | `1.000000` | `1.000000` | `0.800000` | `0.000000` | `0.026265` |
| `wrong_grounding_control` | `0.910000` | `0.150000` | `0.870000` | `0.870000` | `0.870000` | `0.733333` | `1.000000` | `0.000000` | `0.000000` | `0.002300` |

## Gaps And Controls

- recursive_gap_vs_no_pulse: `0.400000`
- recursive_gap_vs_always_commit: `0.650000`
- recursive_gap_vs_never_reframe: `0.250000`
- recursive_gap_vs_static: `0.000000`

| Control | Mean | Std |
|---|---:|---:|
| `reframe_authority_drop` | `0.997641` | `0.000268` |
| `shuffled_pulse_action_drop` | `0.010000` | `0.020000` |
| `shuffled_pulse_drop` | `0.280000` | `0.040000` |
| `wrong_mode_drop` | `0.779995` | `0.065794` |
| `wrong_phase_drop` | `0.310000` | `0.020000` |

## Verdict

```json
{
  "supports_pilot_pulse_commit_timing": true,
  "supports_wait_under_ambiguity": true,
  "supports_reframe_after_delayed_correction": true,
  "supports_reality_confirmation_commit": true,
  "supports_committed_state_update": true,
  "supports_pulse_causal_control": true,
  "static_baseline_shortcut_detected": true,
  "full_pilot_v0_strengthened": true
}
```

## Interpretation

Positive readout requires correct labels and pulse-specific behavior: waiting under ambiguity, reframing delayed nonreal corrections, committing delayed reality confirmations, updating recovered state, and failing the always-commit / never-reframe controls.

If the static full-context baseline solves labels, this is reported as a shortcut baseline rather than a recursive Pilot mechanism.

## Claim Boundary

Toy evidence only. No consciousness, biology, quantum behavior, natural-language-understanding, full VRAXION, production, or deployment claim.

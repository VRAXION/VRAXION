# Pilot Supervision Knockout Matrix

## Goal

Measure how much explicit pulse/self-state supervision is required for the inferred Pilot Pulse mechanism.

Reduced-supervision arms train with differentiable soft committed-state carry, but all primary verdict metrics use hard free-run evaluation.

## Main Results

| Arm | Action | Pulse Align | State Align | Wait | Commit | Reframe | Update | False Commit | Leakage | Soft-Hard Gap | Shuffled Drop | Wrong Mode Drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `full_supervision` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.000000` | `0.001356` | `0.000000` | `0.380000` | `0.976910` |
| `no_pulse_supervision` | `1.000000` | `0.260000` | `0.880000` | `0.433333` | `0.200000` | `0.550000` | `0.400000` | `0.000000` | `0.000851` | `0.000000` | `0.340000` | `0.984221` |
| `no_state_supervision` | `0.790000` | `0.960000` | `0.170000` | `0.966667` | `0.900000` | `1.000000` | `1.000000` | `0.400000` | `0.001268` | `0.050000` | `0.310000` | `0.947161` |
| `no_pulse_no_state` | `0.760000` | `0.220000` | `0.210000` | `0.133333` | `0.000000` | `0.200000` | `0.400000` | `0.200000` | `0.000572` | `0.120000` | `0.220000` | `0.995396` |
| `action_only` | `0.770000` | `0.120000` | `0.050000` | `0.033333` | `0.100000` | `0.000000` | `0.600000` | `0.200000` | `0.572851` | `0.120000` | `0.290000` | `-0.104975` |
| `action_plus_counterfactual_controls` | `0.760000` | `0.190000` | `0.170000` | `0.166667` | `0.100000` | `0.200000` | `0.400000` | `0.000000` | `0.530347` | `0.010000` | `0.330000` | `-0.019501` |
| `static_full_context_baseline` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.000000` | `0.001163` | `null` | `null` | `null` |
| `oracle_full_supervision` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.000000` | `0.001523` | `null` | `null` | `null` |

## Robustness

Best reduced-supervision arm by action/alignment: `no_pulse_supervision`.

| Arm | Heldout Cue | Distractor Cue | Longer Delay | Delayed Correction | Reframe After Commit | Recovery |
|---|---:|---:|---:|---:|---:|---:|
| `full_supervision` | `1.000000` | `0.860000` | `0.970000` | `1.000000` | `1.000000` | `1.000000` |
| `no_pulse_supervision` | `1.000000` | `0.000000` | `0.000000` | `1.000000` | `1.000000` | `1.000000` |
| `action_plus_counterfactual_controls` | `0.760000` | `0.000000` | `0.000000` | `1.000000` | `1.000000` | `0.000000` |

## Verdict

```json
{
  "action_only_collapses_to_shortcut": false,
  "counterfactual_pressure_recovers_mechanism": false,
  "full_supervision_still_upper_bound": true,
  "latent_pilot_mechanism_supported": false,
  "pulse_emerges_without_pulse_labels": false,
  "state_emerges_without_state_labels": false,
  "supervision_dependency_identified": true
}
```

## Interpretation

If reduced-supervision arms preserve action but lose pulse/state alignment, they are reported as shortcut solvers rather than latent Pilot mechanisms.

If counterfactual pressure recovers alignment without direct pulse/state labels, this supports consequence-based training pressure as a replacement for component supervision.

Observed result: full supervision remains the only arm that preserves action, pulse alignment, state alignment, low leakage, and robustness together.

`no_pulse_supervision` preserves final action and state alignment, but pulse alignment collapses, so the pulse policy does not emerge from state/action supervision alone.

`no_state_supervision` preserves most pulse labels but loses committed-state alignment and action reliability, so pulse labels alone do not recover the state mechanism.

`action_only` and `action_plus_counterfactual_controls` do not recover latent pulse/state structure under hard free-run evaluation.

The supervision dependency is therefore identified: current Pilot Pulse v1 works as a supervised component stack, but this toy setup does not yet show self-discovery of the pulse/state mechanism from final consequences alone.

## Claim Boundary

Toy evidence only. No consciousness, biology, quantum behavior, natural-language-understanding, full VRAXION, production, or deployment claim.

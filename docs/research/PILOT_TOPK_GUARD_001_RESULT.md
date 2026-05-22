# PILOT_TOPK_GUARD_001 Result

## Goal

Compare deterministic Pilot commit policies on clean, weak, ambiguous, unknown, conflict, negated, and delayed-correction evidence.

This is a decision-policy probe only. It does not train a model and does not integrate with Pilot Pulse.

## Setup

Evidence vector: `e = [ADD_evidence, MUL_evidence, UNKNOWN_evidence]`.

All policies use fair UNKNOWN handling: if the selected top hypothesis is `UNKNOWN`, the action is `REJECT_UNKNOWN`.

## Policies Compared

- `softmax_argmax`
- `entropy_guard`
- `evidence_strength_margin_guard`
- `topK2_guard`
- `quantum_guard`
- `topK_quantum_guard`

## Fixed Thresholds

```json
{
  "entropy": 0.75,
  "margin": 0.3,
  "purity": 0.7,
  "snr": 3.0,
  "strength": 0.75,
  "top2_active": 0.25
}
```

## Aggregate Metrics

| Policy | Overall | False Commit | Known Exec | Weak Hold | Ambiguous Hold | Conflict Hold | Unknown Reject | Missed Execute | Noise Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `softmax_argmax` | `0.571` | `0.429` | `1.000` | `0.000` | `0.000` | `0.000` | `1.000` | `0.000` | `0.571` |
| `entropy_guard` | `0.500` | `0.000` | `0.000` | `1.000` | `1.000` | `1.000` | `0.000` | `0.357` | `0.500` |
| `evidence_strength_margin_guard` | `1.000` | `0.000` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` | `0.996` |
| `topK2_guard` | `1.000` | `0.000` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` | `0.996` |
| `quantum_guard` | `1.000` | `0.000` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` | `0.970` |
| `topK_quantum_guard` | `1.000` | `0.000` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` | `0.963` |

## Per-Case Metrics

| Policy | Case | Expected | Action | Correct | False Commit | Margin | Evidence H | Softmax H | Purity | SNR |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `softmax_argmax` | `known_ADD` | `EXEC_ADD` | `EXEC_ADD` | `1` | `0` | `1.000` | `0.000` | `0.975` | `1.000` | `21.582` |
| `softmax_argmax` | `known_MUL` | `EXEC_MUL` | `EXEC_MUL` | `1` | `0` | `1.000` | `0.000` | `0.975` | `1.000` | `20.725` |
| `softmax_argmax` | `unknown_DIV` | `REJECT_UNKNOWN` | `REJECT_UNKNOWN` | `1` | `0` | `1.000` | `0.000` | `0.975` | `1.000` | `20.330` |
| `softmax_argmax` | `weak_ADD` | `HOLD_ASK_RESEARCH` | `EXEC_ADD` | `0` | `1` | `0.400` | `0.325` | `1.077` | `0.820` | `6.126` |
| `softmax_argmax` | `weak_MUL` | `HOLD_ASK_RESEARCH` | `EXEC_MUL` | `0` | `1` | `0.400` | `0.325` | `1.077` | `0.820` | `5.940` |
| `softmax_argmax` | `ambiguous_ADD_MUL` | `HOLD_ASK_RESEARCH` | `EXEC_ADD` | `0` | `1` | `0.000` | `0.693` | `1.074` | `0.500` | `1.332` |
| `softmax_argmax` | `near_ADD_strong` | `EXEC_ADD` | `EXEC_ADD` | `1` | `0` | `0.750` | `0.336` | `1.019` | `0.812` | `10.530` |
| `softmax_argmax` | `near_MUL_strong` | `EXEC_MUL` | `EXEC_MUL` | `1` | `0` | `0.750` | `0.336` | `1.019` | `0.812` | `10.642` |
| `softmax_argmax` | `near_UNKNOWN` | `REJECT_UNKNOWN` | `REJECT_UNKNOWN` | `1` | `0` | `0.800` | `0.409` | `1.020` | `0.806` | `11.862` |
| `softmax_argmax` | `no_evidence` | `HOLD_ASK_RESEARCH` | `EXEC_ADD` | `0` | `1` | `0.000` | `1.099` | `1.099` | `0.333` | `1.057` |
| `softmax_argmax` | `conflict_all` | `HOLD_ASK_RESEARCH` | `EXEC_ADD` | `0` | `1` | `0.000` | `1.099` | `1.099` | `0.333` | `1.283` |
| `softmax_argmax` | `negated_ADD` | `HOLD_ASK_RESEARCH|REFRAME` | `HOLD_ASK_RESEARCH` | `1` | `0` | `0.900` | `0.000` | `0.999` | `1.000` | `14.056` |
| `softmax_argmax` | `delayed_correction_step1` | `HOLD_ASK_RESEARCH` | `EXEC_ADD` | `0` | `1` | `0.000` | `0.693` | `1.074` | `0.500` | `1.372` |
| `softmax_argmax` | `delayed_correction_step2` | `EXEC_MUL` | `EXEC_MUL` | `1` | `0` | `1.000` | `0.000` | `0.975` | `1.000` | `21.054` |
| `entropy_guard` | `known_ADD` | `EXEC_ADD` | `HOLD_ASK_RESEARCH` | `0` | `0` | `1.000` | `0.000` | `0.975` | `1.000` | `21.582` |
| `entropy_guard` | `known_MUL` | `EXEC_MUL` | `HOLD_ASK_RESEARCH` | `0` | `0` | `1.000` | `0.000` | `0.975` | `1.000` | `20.725` |
| `entropy_guard` | `unknown_DIV` | `REJECT_UNKNOWN` | `HOLD_ASK_RESEARCH` | `0` | `0` | `1.000` | `0.000` | `0.975` | `1.000` | `20.330` |
| `entropy_guard` | `weak_ADD` | `HOLD_ASK_RESEARCH` | `HOLD_ASK_RESEARCH` | `1` | `0` | `0.400` | `0.325` | `1.077` | `0.820` | `6.126` |
| `entropy_guard` | `weak_MUL` | `HOLD_ASK_RESEARCH` | `HOLD_ASK_RESEARCH` | `1` | `0` | `0.400` | `0.325` | `1.077` | `0.820` | `5.940` |
| `entropy_guard` | `ambiguous_ADD_MUL` | `HOLD_ASK_RESEARCH` | `HOLD_ASK_RESEARCH` | `1` | `0` | `0.000` | `0.693` | `1.074` | `0.500` | `1.332` |
| `entropy_guard` | `near_ADD_strong` | `EXEC_ADD` | `HOLD_ASK_RESEARCH` | `0` | `0` | `0.750` | `0.336` | `1.019` | `0.812` | `10.530` |
| `entropy_guard` | `near_MUL_strong` | `EXEC_MUL` | `HOLD_ASK_RESEARCH` | `0` | `0` | `0.750` | `0.336` | `1.019` | `0.812` | `10.642` |
| `entropy_guard` | `near_UNKNOWN` | `REJECT_UNKNOWN` | `HOLD_ASK_RESEARCH` | `0` | `0` | `0.800` | `0.409` | `1.020` | `0.806` | `11.862` |
| `entropy_guard` | `no_evidence` | `HOLD_ASK_RESEARCH` | `HOLD_ASK_RESEARCH` | `1` | `0` | `0.000` | `1.099` | `1.099` | `0.333` | `1.057` |
| `entropy_guard` | `conflict_all` | `HOLD_ASK_RESEARCH` | `HOLD_ASK_RESEARCH` | `1` | `0` | `0.000` | `1.099` | `1.099` | `0.333` | `1.283` |
| `entropy_guard` | `negated_ADD` | `HOLD_ASK_RESEARCH|REFRAME` | `HOLD_ASK_RESEARCH` | `1` | `0` | `0.900` | `0.000` | `0.999` | `1.000` | `14.056` |
| `entropy_guard` | `delayed_correction_step1` | `HOLD_ASK_RESEARCH` | `HOLD_ASK_RESEARCH` | `1` | `0` | `0.000` | `0.693` | `1.074` | `0.500` | `1.372` |
| `entropy_guard` | `delayed_correction_step2` | `EXEC_MUL` | `HOLD_ASK_RESEARCH` | `0` | `0` | `1.000` | `0.000` | `0.975` | `1.000` | `21.054` |
| `evidence_strength_margin_guard` | `known_ADD` | `EXEC_ADD` | `EXEC_ADD` | `1` | `0` | `1.000` | `0.000` | `0.975` | `1.000` | `21.582` |
| `evidence_strength_margin_guard` | `known_MUL` | `EXEC_MUL` | `EXEC_MUL` | `1` | `0` | `1.000` | `0.000` | `0.975` | `1.000` | `20.725` |
| `evidence_strength_margin_guard` | `unknown_DIV` | `REJECT_UNKNOWN` | `REJECT_UNKNOWN` | `1` | `0` | `1.000` | `0.000` | `0.975` | `1.000` | `20.330` |
| `evidence_strength_margin_guard` | `weak_ADD` | `HOLD_ASK_RESEARCH` | `HOLD_ASK_RESEARCH` | `1` | `0` | `0.400` | `0.325` | `1.077` | `0.820` | `6.126` |
| `evidence_strength_margin_guard` | `weak_MUL` | `HOLD_ASK_RESEARCH` | `HOLD_ASK_RESEARCH` | `1` | `0` | `0.400` | `0.325` | `1.077` | `0.820` | `5.940` |
| `evidence_strength_margin_guard` | `ambiguous_ADD_MUL` | `HOLD_ASK_RESEARCH` | `HOLD_ASK_RESEARCH` | `1` | `0` | `0.000` | `0.693` | `1.074` | `0.500` | `1.332` |
| `evidence_strength_margin_guard` | `near_ADD_strong` | `EXEC_ADD` | `EXEC_ADD` | `1` | `0` | `0.750` | `0.336` | `1.019` | `0.812` | `10.530` |
| `evidence_strength_margin_guard` | `near_MUL_strong` | `EXEC_MUL` | `EXEC_MUL` | `1` | `0` | `0.750` | `0.336` | `1.019` | `0.812` | `10.642` |
| `evidence_strength_margin_guard` | `near_UNKNOWN` | `REJECT_UNKNOWN` | `REJECT_UNKNOWN` | `1` | `0` | `0.800` | `0.409` | `1.020` | `0.806` | `11.862` |
| `evidence_strength_margin_guard` | `no_evidence` | `HOLD_ASK_RESEARCH` | `HOLD_ASK_RESEARCH` | `1` | `0` | `0.000` | `1.099` | `1.099` | `0.333` | `1.057` |
| `evidence_strength_margin_guard` | `conflict_all` | `HOLD_ASK_RESEARCH` | `HOLD_ASK_RESEARCH` | `1` | `0` | `0.000` | `1.099` | `1.099` | `0.333` | `1.283` |
| `evidence_strength_margin_guard` | `negated_ADD` | `HOLD_ASK_RESEARCH|REFRAME` | `HOLD_ASK_RESEARCH` | `1` | `0` | `0.900` | `0.000` | `0.999` | `1.000` | `14.056` |
| `evidence_strength_margin_guard` | `delayed_correction_step1` | `HOLD_ASK_RESEARCH` | `HOLD_ASK_RESEARCH` | `1` | `0` | `0.000` | `0.693` | `1.074` | `0.500` | `1.372` |
| `evidence_strength_margin_guard` | `delayed_correction_step2` | `EXEC_MUL` | `EXEC_MUL` | `1` | `0` | `1.000` | `0.000` | `0.975` | `1.000` | `21.054` |
| `topK2_guard` | `known_ADD` | `EXEC_ADD` | `EXEC_ADD` | `1` | `0` | `1.000` | `0.000` | `0.975` | `1.000` | `21.582` |
| `topK2_guard` | `known_MUL` | `EXEC_MUL` | `EXEC_MUL` | `1` | `0` | `1.000` | `0.000` | `0.975` | `1.000` | `20.725` |
| `topK2_guard` | `unknown_DIV` | `REJECT_UNKNOWN` | `REJECT_UNKNOWN` | `1` | `0` | `1.000` | `0.000` | `0.975` | `1.000` | `20.330` |
| `topK2_guard` | `weak_ADD` | `HOLD_ASK_RESEARCH` | `HOLD_ASK_RESEARCH` | `1` | `0` | `0.400` | `0.325` | `1.077` | `0.820` | `6.126` |
| `topK2_guard` | `weak_MUL` | `HOLD_ASK_RESEARCH` | `HOLD_ASK_RESEARCH` | `1` | `0` | `0.400` | `0.325` | `1.077` | `0.820` | `5.940` |
| `topK2_guard` | `ambiguous_ADD_MUL` | `HOLD_ASK_RESEARCH` | `HOLD_ASK_RESEARCH` | `1` | `0` | `0.000` | `0.693` | `1.074` | `0.500` | `1.332` |
| `topK2_guard` | `near_ADD_strong` | `EXEC_ADD` | `EXEC_ADD` | `1` | `0` | `0.750` | `0.336` | `1.019` | `0.812` | `10.530` |
| `topK2_guard` | `near_MUL_strong` | `EXEC_MUL` | `EXEC_MUL` | `1` | `0` | `0.750` | `0.336` | `1.019` | `0.812` | `10.642` |
| `topK2_guard` | `near_UNKNOWN` | `REJECT_UNKNOWN` | `REJECT_UNKNOWN` | `1` | `0` | `0.800` | `0.409` | `1.020` | `0.806` | `11.862` |
| `topK2_guard` | `no_evidence` | `HOLD_ASK_RESEARCH` | `HOLD_ASK_RESEARCH` | `1` | `0` | `0.000` | `1.099` | `1.099` | `0.333` | `1.057` |
| `topK2_guard` | `conflict_all` | `HOLD_ASK_RESEARCH` | `HOLD_ASK_RESEARCH` | `1` | `0` | `0.000` | `1.099` | `1.099` | `0.333` | `1.283` |
| `topK2_guard` | `negated_ADD` | `HOLD_ASK_RESEARCH|REFRAME` | `HOLD_ASK_RESEARCH` | `1` | `0` | `0.900` | `0.000` | `0.999` | `1.000` | `14.056` |
| `topK2_guard` | `delayed_correction_step1` | `HOLD_ASK_RESEARCH` | `HOLD_ASK_RESEARCH` | `1` | `0` | `0.000` | `0.693` | `1.074` | `0.500` | `1.372` |
| `topK2_guard` | `delayed_correction_step2` | `EXEC_MUL` | `EXEC_MUL` | `1` | `0` | `1.000` | `0.000` | `0.975` | `1.000` | `21.054` |
| `quantum_guard` | `known_ADD` | `EXEC_ADD` | `EXEC_ADD` | `1` | `0` | `1.000` | `0.000` | `0.975` | `1.000` | `21.582` |
| `quantum_guard` | `known_MUL` | `EXEC_MUL` | `EXEC_MUL` | `1` | `0` | `1.000` | `0.000` | `0.975` | `1.000` | `20.725` |
| `quantum_guard` | `unknown_DIV` | `REJECT_UNKNOWN` | `REJECT_UNKNOWN` | `1` | `0` | `1.000` | `0.000` | `0.975` | `1.000` | `20.330` |
| `quantum_guard` | `weak_ADD` | `HOLD_ASK_RESEARCH` | `HOLD_ASK_RESEARCH` | `1` | `0` | `0.400` | `0.325` | `1.077` | `0.820` | `6.126` |
| `quantum_guard` | `weak_MUL` | `HOLD_ASK_RESEARCH` | `HOLD_ASK_RESEARCH` | `1` | `0` | `0.400` | `0.325` | `1.077` | `0.820` | `5.940` |
| `quantum_guard` | `ambiguous_ADD_MUL` | `HOLD_ASK_RESEARCH` | `HOLD_ASK_RESEARCH` | `1` | `0` | `0.000` | `0.693` | `1.074` | `0.500` | `1.332` |
| `quantum_guard` | `near_ADD_strong` | `EXEC_ADD` | `EXEC_ADD` | `1` | `0` | `0.750` | `0.336` | `1.019` | `0.812` | `10.530` |
| `quantum_guard` | `near_MUL_strong` | `EXEC_MUL` | `EXEC_MUL` | `1` | `0` | `0.750` | `0.336` | `1.019` | `0.812` | `10.642` |
| `quantum_guard` | `near_UNKNOWN` | `REJECT_UNKNOWN` | `REJECT_UNKNOWN` | `1` | `0` | `0.800` | `0.409` | `1.020` | `0.806` | `11.862` |
| `quantum_guard` | `no_evidence` | `HOLD_ASK_RESEARCH` | `HOLD_ASK_RESEARCH` | `1` | `0` | `0.000` | `1.099` | `1.099` | `0.333` | `1.057` |
| `quantum_guard` | `conflict_all` | `HOLD_ASK_RESEARCH` | `HOLD_ASK_RESEARCH` | `1` | `0` | `0.000` | `1.099` | `1.099` | `0.333` | `1.283` |
| `quantum_guard` | `negated_ADD` | `HOLD_ASK_RESEARCH|REFRAME` | `HOLD_ASK_RESEARCH` | `1` | `0` | `0.900` | `0.000` | `0.999` | `1.000` | `14.056` |
| `quantum_guard` | `delayed_correction_step1` | `HOLD_ASK_RESEARCH` | `HOLD_ASK_RESEARCH` | `1` | `0` | `0.000` | `0.693` | `1.074` | `0.500` | `1.372` |
| `quantum_guard` | `delayed_correction_step2` | `EXEC_MUL` | `EXEC_MUL` | `1` | `0` | `1.000` | `0.000` | `0.975` | `1.000` | `21.054` |
| `topK_quantum_guard` | `known_ADD` | `EXEC_ADD` | `EXEC_ADD` | `1` | `0` | `1.000` | `0.000` | `0.975` | `1.000` | `21.582` |
| `topK_quantum_guard` | `known_MUL` | `EXEC_MUL` | `EXEC_MUL` | `1` | `0` | `1.000` | `0.000` | `0.975` | `1.000` | `20.725` |
| `topK_quantum_guard` | `unknown_DIV` | `REJECT_UNKNOWN` | `REJECT_UNKNOWN` | `1` | `0` | `1.000` | `0.000` | `0.975` | `1.000` | `20.330` |
| `topK_quantum_guard` | `weak_ADD` | `HOLD_ASK_RESEARCH` | `HOLD_ASK_RESEARCH` | `1` | `0` | `0.400` | `0.325` | `1.077` | `0.820` | `6.126` |
| `topK_quantum_guard` | `weak_MUL` | `HOLD_ASK_RESEARCH` | `HOLD_ASK_RESEARCH` | `1` | `0` | `0.400` | `0.325` | `1.077` | `0.820` | `5.940` |
| `topK_quantum_guard` | `ambiguous_ADD_MUL` | `HOLD_ASK_RESEARCH` | `HOLD_ASK_RESEARCH` | `1` | `0` | `0.000` | `0.693` | `1.074` | `0.500` | `1.332` |
| `topK_quantum_guard` | `near_ADD_strong` | `EXEC_ADD` | `EXEC_ADD` | `1` | `0` | `0.750` | `0.336` | `1.019` | `0.812` | `10.530` |
| `topK_quantum_guard` | `near_MUL_strong` | `EXEC_MUL` | `EXEC_MUL` | `1` | `0` | `0.750` | `0.336` | `1.019` | `0.812` | `10.642` |
| `topK_quantum_guard` | `near_UNKNOWN` | `REJECT_UNKNOWN` | `REJECT_UNKNOWN` | `1` | `0` | `0.800` | `0.409` | `1.020` | `0.806` | `11.862` |
| `topK_quantum_guard` | `no_evidence` | `HOLD_ASK_RESEARCH` | `HOLD_ASK_RESEARCH` | `1` | `0` | `0.000` | `1.099` | `1.099` | `0.333` | `1.057` |
| `topK_quantum_guard` | `conflict_all` | `HOLD_ASK_RESEARCH` | `HOLD_ASK_RESEARCH` | `1` | `0` | `0.000` | `1.099` | `1.099` | `0.333` | `1.283` |
| `topK_quantum_guard` | `negated_ADD` | `HOLD_ASK_RESEARCH|REFRAME` | `HOLD_ASK_RESEARCH` | `1` | `0` | `0.900` | `0.000` | `0.999` | `1.000` | `14.056` |
| `topK_quantum_guard` | `delayed_correction_step1` | `HOLD_ASK_RESEARCH` | `HOLD_ASK_RESEARCH` | `1` | `0` | `0.000` | `0.693` | `1.074` | `0.500` | `1.372` |
| `topK_quantum_guard` | `delayed_correction_step2` | `EXEC_MUL` | `EXEC_MUL` | `1` | `0` | `1.000` | `0.000` | `0.975` | `1.000` | `21.054` |

## Noise Stability

Noise uses 500 clipped Gaussian perturbations per case at sigma 0.05, plus a sigma 0.08 stress variant. The CSV contains action distributions and per-case noise statistics.

## Failure Cases

- `softmax_argmax` on `weak_ADD`: expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD`.
- `softmax_argmax` on `weak_MUL`: expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL`.
- `softmax_argmax` on `ambiguous_ADD_MUL`: expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD`.
- `softmax_argmax` on `no_evidence`: expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD`. softmax([0,0,0]) is uniform; deterministic argmax committed ADD.
- `softmax_argmax` on `conflict_all`: expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD`.
- `softmax_argmax` on `delayed_correction_step1`: expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD`.
- `entropy_guard` on `known_ADD`: expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH`.
- `entropy_guard` on `known_MUL`: expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH`.
- `entropy_guard` on `unknown_DIV`: expected `REJECT_UNKNOWN`, got `HOLD_ASK_RESEARCH`.
- `entropy_guard` on `near_ADD_strong`: expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH`.
- `entropy_guard` on `near_MUL_strong`: expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH`.
- `entropy_guard` on `near_UNKNOWN`: expected `REJECT_UNKNOWN`, got `HOLD_ASK_RESEARCH`.
- `entropy_guard` on `delayed_correction_step2`: expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH`.

## Verdict

```json
[
  "TOPK_GUARD_POSITIVE",
  "TOPK_GUARD_NO_BETTER_THAN_EVIDENCE",
  "ENTROPY_ONLY_INSUFFICIENT",
  "BRITTLE_SWITCH_CONFIRMED"
]
```

## Next Action

If top-K/evidence guards are positive, the next probe is `PILOT_SENSOR_001`: raw command text -> evidence vector -> guarded pilot -> locked skill.

## Claim Boundary

This does not prove full PilotPulse learning, raw text understanding, full VRAXION/INSTNCT behavior, production architecture, or consciousness.

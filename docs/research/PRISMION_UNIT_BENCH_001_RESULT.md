# PRISMION_UNIT_BENCH_001 Result

## Goal

Test whether Prismion-style phase/interference units are a useful primitive for cancellation and command-scope decisions.

This is a unit-level expressivity and learnability probe, not a full model benchmark.

## Arm Summary

| Arm | Mean Acc | Perfect Rate | False Exec | Mean Params |
|---|---:|---:|---:|---:|
| `one_relu_neuron` | `0.548` | `0.000` | `0.075` | `14.3` |
| `two_relu_mlp` | `0.641` | `0.207` | `0.086` | `25.3` |
| `fixed_prismion` | `1.000` | `1.000` | `0.000` | `0.0` |
| `learned_prismion_gain_only` | `0.833` | `0.667` | `0.042` | `36.3` |
| `learned_prismion_gain_phase` | `0.827` | `0.667` | `0.070` | `61.7` |
| `hybrid_prismion_relu` | `0.792` | `0.660` | `0.066` | `179.7` |

## Task / Arm Detail

| Task / Arm | Mean Acc | Min Acc | Perfect Rate | False Exec | Median Epoch | Params |
|---|---:|---:|---:|---:|---:|---:|
| `pilot_scope_core/fixed_prismion` | `1.000` | `1.000` | `1.000` | `0.000` | `null` | `0` |
| `pilot_scope_core/hybrid_prismion_relu` | `0.998` | `0.867` | `0.980` | `0.001` | `29` | `233` |
| `pilot_scope_core/learned_prismion_gain_only` | `1.000` | `1.000` | `1.000` | `0.000` | `27` | `49` |
| `pilot_scope_core/learned_prismion_gain_phase` | `1.000` | `1.000` | `1.000` | `0.000` | `15` | `85` |
| `pilot_scope_core/one_relu_neuron` | `0.684` | `0.533` | `0.000` | `0.031` | `null` | `18` |
| `pilot_scope_core/two_relu_mlp` | `0.842` | `0.533` | `0.300` | `0.035` | `147` | `32` |
| `pilot_scope_factor_heldout/fixed_prismion` | `1.000` | `1.000` | `1.000` | `0.000` | `null` | `0` |
| `pilot_scope_factor_heldout/hybrid_prismion_relu` | `0.378` | `0.125` | `0.000` | `0.198` | `21` | `233` |
| `pilot_scope_factor_heldout/learned_prismion_gain_only` | `0.500` | `0.500` | `0.000` | `0.125` | `8` | `49` |
| `pilot_scope_factor_heldout/learned_prismion_gain_phase` | `0.480` | `0.375` | `0.000` | `0.211` | `7` | `85` |
| `pilot_scope_factor_heldout/one_relu_neuron` | `0.346` | `0.000` | `0.000` | `0.194` | `199` | `18` |
| `pilot_scope_factor_heldout/two_relu_mlp` | `0.330` | `0.125` | `0.000` | `0.223` | `90` | `32` |
| `xor_cancellation/fixed_prismion` | `1.000` | `1.000` | `1.000` | `0.000` | `null` | `0` |
| `xor_cancellation/hybrid_prismion_relu` | `1.000` | `1.000` | `1.000` | `0.000` | `5` | `73` |
| `xor_cancellation/learned_prismion_gain_only` | `1.000` | `1.000` | `1.000` | `0.000` | `12` | `11` |
| `xor_cancellation/learned_prismion_gain_phase` | `1.000` | `1.000` | `1.000` | `0.000` | `10` | `15` |
| `xor_cancellation/one_relu_neuron` | `0.615` | `0.250` | `0.000` | `0.000` | `null` | `7` |
| `xor_cancellation/two_relu_mlp` | `0.752` | `0.250` | `0.320` | `0.000` | `42` | `12` |

## Verdict

```json
[
  "FIXED_PRISMION_POSITIVE",
  "LEARNED_PRISMION_BEATS_TWO_RELU",
  "PRISMION_ONLY_FIXED_POSITIVE",
  "FACTOR_HELDOUT_LEARNABILITY_WEAK"
]
```

## Interpretation

The fixed Prismion arm is an upper-bound representation check: it asks whether the primitive can express the target cancellation/interference rules directly.

The learned Prismion arms test whether the primitive is learnable from labels across many seeds. If fixed Prismion passes but learned Prismion is weak, the representation is promising but the learning rule/cell still needs work.

The ReLU arms are deliberately tiny. A two-ReLU model solving parts of the bench means ordinary neurons can represent some cases, not that phase/interference is useless.

## Claim Boundary

Toy command/cancellation domain only. No consciousness claim, no quantum physics claim, no general NLU claim, and no full VRAXION/INSTNCT claim.

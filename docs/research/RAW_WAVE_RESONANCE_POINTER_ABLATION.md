# Raw Wave Resonance Pointer Ablation

Source runs:

- `target/context-cancellation-probe/raw-wave-ablation/none/20260504T214618Z/multi_aspect_refraction_report.json`
- `target/context-cancellation-probe/raw-wave-ablation/token_wave/20260504T214836Z/multi_aspect_refraction_report.json`
- `target/context-cancellation-probe/raw-wave-ablation/neuron_resonance/20260504T215052Z/multi_aspect_refraction_report.json`
- `target/context-cancellation-probe/raw-wave-ablation/pointer_resonance/20260504T215311Z/multi_aspect_refraction_report.json`
- `target/context-cancellation-probe/raw-wave-ablation/pointer_resonance_signed/20260504T215535Z/multi_aspect_refraction_report.json`

## 1. Goal

This ablation tests a direct version of the wave/resonance interpretation:

> Tokens enter as simple wave-like vectors, hidden neurons behave like resonance chambers, and a frame-derived pointer phase tunes which token interactions become decision-relevant.

The experiment keeps the existing multi-aspect token refraction task. It does not add semantic bands such as "dog danger band" or "dog friend band". The question is whether an explicit wave/pointer construction improves frame-conditioned authority switching over the existing recurrent baseline.

## 2. Setup

Main task:

```bash
.venv/bin/python scripts/run_context_cancellation_probe.py \
  --experiment multi_aspect_token_refraction \
  --input-mode entangled \
  --hidden 128 \
  --update-rate 0.3 \
  --steps 5 \
  --epochs 220 \
  --seeds 5
```

Modes:

- `none`: existing multi-aspect recurrent baseline.
- `token_wave`: fixed sin/cos token wave vectors.
- `neuron_resonance`: fixed token phases plus learned-style neuron preferred phases used to build resonant inputs.
- `pointer_resonance`: frame-derived pointer phase shifts token phases before neuron resonance.
- `pointer_resonance_signed`: pointer resonance plus signed/inverting recurrent edges.

Implementation boundary:

This is a small precomputed-input ablation, not a new end-to-end trainable wave architecture. The recurrent model is still the same toy recurrent classifier, and the resonance modes alter token/input construction plus post-training interventions.

## 3. Main Results

| Mode | Accuracy | Recurrence Gain | Zero Recurrent | Randomized Recurrent | No Frame | Shuffled Frame | Refraction Final | Authority Switch | Mean Actor Switch | Dog Switch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `none` | `0.933333` | `0.393333` | `0.540000` | `0.530333` | `0.671333` | `0.607500` | `0.098167` | `0.123000` | `0.309111` | `0.288000` |
| `token_wave` | `0.939167` | `0.407667` | `0.531500` | `0.516667` | `0.672333` | `0.611000` | `0.105667` | `0.133167` | `0.339556` | `0.317333` |
| `neuron_resonance` | `0.914500` | `0.367167` | `0.547333` | `0.509833` | `0.670000` | `0.605667` | `0.102500` | `0.117333` | `0.318889` | `0.248000` |
| `pointer_resonance` | `0.938667` | `0.379333` | `0.559333` | `0.527667` | `0.670000` | `0.750000` | `0.095833` | `0.128833` | `0.310444` | `0.270667` |
| `pointer_resonance_signed` | `0.914500` | `0.331000` | `0.583500` | `0.525000` | `0.668000` | `0.804333` | `0.089333` | `0.132000` | `0.266667` | `0.218667` |

Seed stability:

- `none`: accuracy std `0.014501`, refraction std `0.022885`, authority std `0.018934`
- `token_wave`: accuracy std `0.018626`, refraction std `0.012354`, authority std `0.009449`
- `neuron_resonance`: accuracy std `0.015684`, refraction std `0.015120`, authority std `0.017380`
- `pointer_resonance`: accuracy std `0.004170`, refraction std `0.013601`, authority std `0.019512`
- `pointer_resonance_signed`: accuracy std `0.010640`, refraction std `0.010664`, authority std `0.016326`

## 4. Pointer Diagnostics

| Mode | Wrong Pointer Accuracy | Wrong Pointer Drop | Frozen Pointer Accuracy | Frozen Pointer Drop | Shuffled Pointer Accuracy | Shuffled Pointer Drop | Randomized Neuron Accuracy | Randomized Neuron Drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pointer_resonance` | `0.493333` | `0.445333` | `0.490000` | `0.448667` | `0.503667` | `0.435000` | `0.489333` | `0.449333` |
| `pointer_resonance_signed` | `0.500167` | `0.414333` | `0.495333` | `0.419167` | `0.494833` | `0.419667` | `0.518667` | `0.395833` |

Interpretation:

- Pointer interventions strongly hurt the pointer modes, so the pointer pathway is being used.
- However, pointer modes do not improve the intended authority-switch/refraction metrics over `token_wave`, and they do not clearly beat the `none` baseline.
- The high ordinary `shuffled_frame_token` score in pointer modes is an implementation caveat: pointer-shifted observation construction means that ordinary frame-token shuffling is no longer the cleanest pointer control. The pointer-specific shuffled mapping is the better diagnostic here.
- The current `pointer_distance_between_frames` metric was not informative in this implementation and should be redesigned before being used as evidence.

## 5. Verdict

```json
{
  "raw_wave_resonance_pointer_ablation": "negative_for_explicit_pointer_resonance",
  "token_wave_embedding": "weak_positive",
  "pointer_resonance_required": false,
  "reason": "fixed sin/cos token waves slightly improve accuracy, refraction, authority switching, mean actor switch, and dog switch. Explicit neuron/pointer resonance modes are sensitive to pointer interventions, but they do not produce cleaner frame-conditioned authority switching than token_wave or the existing recurrent baseline."
}
```

## 6. Interpretation

The strongest result is the small `token_wave` improvement:

- accuracy: `0.933333 -> 0.939167`
- refraction final: `0.098167 -> 0.105667`
- authority switch: `0.123000 -> 0.133167`
- mean actor switch: `0.309111 -> 0.339556`
- dog switch: `0.288000 -> 0.317333`

This suggests that simple sin/cos token geometry can be a mild inductive bias for the existing multi-aspect refraction task.

The explicit pointer/neuron resonance variants are not the current winner. They show that the pointer can control the task, because wrong/frozen/shuffled pointer interventions collapse accuracy near chance. But this sensitivity did not translate into better authority switching or cleaner refraction.

## 7. Safe Claim

In this controlled toy setting, simple wave-like token inputs provide a weak positive bias for frame-conditioned authority switching. Explicit pointer/neuron resonance is not necessary and did not improve the current task beyond the simpler token-wave baseline.

The current best mechanism read remains:

```text
frame pointer + recurrent attractor / authority switching
```

not:

```text
explicit wave pointer resonance is the main mechanism
```

## 8. Claim Boundary

Do not claim:

- consciousness,
- biological equivalence,
- full VRAXION behavior,
- production validation,
- novelty as a general principle,
- that explicit frequency/pointer resonance is required.

Useful next direction:

Future wave/interference work should focus more on recurrent dynamics, edge/node modulation, and trainable interaction structure than on token embedding geometry alone.

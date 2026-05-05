# Frequency Embedding Ablation

Source runs:

- `target/context-cancellation-probe/frequency-ablation/learned/20260504T212418Z/multi_aspect_refraction_report.json`
- `target/context-cancellation-probe/frequency-ablation/fixed_sincos/20260504T212711Z/multi_aspect_refraction_report.json`
- `target/context-cancellation-probe/frequency-ablation/trainable_phase/20260504T212944Z/multi_aspect_refraction_report.json`
- `target/context-cancellation-probe/frequency-ablation/multi_band_phase/20260504T213215Z/multi_aspect_refraction_report.json`

## Question

This small ablation asks whether frequency/phase-style token vectors make the existing multi-aspect token refraction task cleaner.

The hypothesis was:

> If the frame behaves like a resonance selector, sin/cos or phase-style token vectors might improve frame-conditioned authority switching beyond ordinary token vectors.

## Setup

Task:

- experiment: `multi_aspect_token_refraction`
- input mode: `entangled`
- hidden: `128`
- update rate: `0.3`
- recurrent steps: `5`
- epochs: `220`
- seeds: `5`
- train size: `2400`
- test size: `1200`

Embedding modes:

- `learned`: existing random-vector token baseline in this precomputed-input probe.
- `fixed_sincos`: deterministic real-valued sin/cos token vectors.
- `trainable_phase`: phase-parameterized random sin/cos table. In this small ablation it is not an end-to-end trainable embedding layer.
- `multi_band_phase`: token vectors split into several phase bands.

Implementation note:

The recurrent model and task are unchanged. This ablation changes only how token vectors are generated before the toy dataset is built. Because this script precomputes input vectors, the phase modes are fixed token-vector construction modes rather than a larger trainable embedding architecture.

## Results

| embedding_mode | accuracy | zero_recurrent | randomized_recurrent | no_frame_token | shuffled_frame_token | refraction_index_final | authority_switch_score | mean_actor_switch | dog_switch | accuracy_std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `learned` | 0.933333 | 0.540000 | 0.530333 | 0.671333 | 0.607500 | 0.098167 | 0.123000 | 0.309111 | 0.288000 | 0.014501 |
| `fixed_sincos` | 0.939167 | 0.531500 | 0.516667 | 0.672333 | 0.611000 | 0.105667 | 0.133167 | 0.339556 | 0.317333 | 0.018626 |
| `trainable_phase` | 0.915333 | 0.511667 | 0.503167 | 0.672333 | 0.608167 | 0.110167 | 0.126500 | 0.287778 | 0.240000 | 0.022339 |
| `multi_band_phase` | 0.924333 | 0.528000 | 0.508333 | 0.669000 | 0.603000 | 0.096333 | 0.115333 | 0.286000 | 0.233333 | 0.008472 |

Dog influence by frame:

| embedding_mode | danger_frame | friendship_frame | sound_frame | environment_frame | causal_mean_minus_environment |
|---|---:|---:|---:|---:|---:|
| `learned` | 0.292000 | 0.164000 | 0.528000 | 0.040000 | 0.288000 |
| `fixed_sincos` | 0.308000 | 0.184000 | 0.532000 | 0.024000 | 0.317333 |
| `trainable_phase` | 0.228000 | 0.128000 | 0.532000 | 0.056000 | 0.240000 |
| `multi_band_phase` | 0.288000 | 0.208000 | 0.528000 | 0.108000 | 0.233333 |

Seed-variance read:

- `fixed_sincos` has lower variance on `refraction_index_final`, `authority_switch_score`, and `dog_switch` than the learned/random-vector baseline.
- `multi_band_phase` has the lowest raw accuracy variance but weaker authority metrics.
- `trainable_phase` is the least stable on raw accuracy and dog-switch separation.

## Interpretation

`fixed_sincos` gives a small but consistent improvement over the baseline:

- accuracy improves from `0.933333` to `0.939167`,
- refraction index improves from `0.098167` to `0.105667`,
- authority switch improves from `0.123000` to `0.133167`,
- mean actor authority switch improves from `0.309111` to `0.339556`,
- dog causal-vs-environment influence separation improves from `0.288000` to `0.317333`.

This is not a huge effect, but it is not only raw accuracy. The improvement appears in the authority-switch metrics that matter for the prism/refraction claim.

The more elaborate phase variants do not improve the finding:

- `trainable_phase` has the highest final refraction index, but lower accuracy and weaker dog authority separation.
- `multi_band_phase` is stable but weaker than baseline on authority switching and dog separation.

## Verdict

```json
{
  "frequency_embedding_ablation": "weak_positive_for_fixed_sincos",
  "explicit_phase_embeddings_necessary": "false",
  "reason": "Fixed sin/cos token vectors slightly improve authority-switch and dog-separation metrics, but larger phase-style variants do not beat the baseline. The effect is useful as a small ablation, not a new main mechanism."
}
```

Safe claim:

> In this controlled toy setting, fixed real-valued sin/cos token vectors can make multi-aspect token refraction slightly cleaner, but explicit phase-style embeddings are not necessary for the effect.

## Claim Boundary

Toy evidence only. Do not claim:

- consciousness,
- biology,
- novelty,
- full VRAXION behavior,
- that resonance-style embeddings are required.

The current winner remains:

> frame pointer + recurrent attractor / authority switching.

The frequency metaphor has weak operational support through `fixed_sincos`, but the learned/random-vector baseline already captures most of the effect.

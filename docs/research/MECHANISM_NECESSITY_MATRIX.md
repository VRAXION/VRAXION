# Mechanism Necessity Matrix

Source runs and notes:

- `target/context-cancellation-probe/v6-robustness-grid/h64_u0.3/20260504T195157Z/latent_refraction_report.json`
- `target/context-cancellation-probe/multi-aspect-h128-u03-v2/20260504T201450Z/multi_aspect_refraction_report.json`
- `target/context-cancellation-probe/temporal-order-contrast-refraction/20260505T214443Z/temporal_order_contrast_refraction_report.json`
- `docs/research/FREQUENCY_EMBEDDING_ABLATION.md`
- `docs/research/RAW_WAVE_RESONANCE_POINTER_ABLATION.md`
- `docs/research/HUB_RICH_TOPOLOGY_PRIOR_VALIDATION.md`
- `docs/research/HUB_DEGREE_PRESERVING_CONTROL.md`
- `docs/research/QUERY_CUED_POINTER_BOTTLENECK_FINDING.md`

## Why This Test Was Run

The probe line has produced several positive toy findings:

- latent refraction: same entangled input plus different frame gives different feature-group authority,
- multi-aspect token refraction: the same token can change decision authority by frame,
- temporal order contrast: the same token set can require different role/frame labels depending on order.

The next question is not whether more toy effects can be produced. The question is what survives when controls remove parts of the system.

This matrix classifies components as:

| Class | Meaning |
|---|---|
| `necessary_core` | Removing it kills or strongly weakens the target effect across the relevant task family. |
| `task_specific` | Useful in one task family but not a universal survivor. |
| `helpful_bias` | Improves metrics or stability but is not required. |
| `shortcut_risk` | Can make accuracy good while weakening mechanism interpretation. |
| `nonessential` | Removal does not meaningfully hurt, or stronger controls show it is not needed. |

## Components Tested

| Component | Classification | Evidence |
|---|---|---|
| Control/frame signal for frame-conditioned refraction | `necessary_core` | In `latent_refraction`, full accuracy is `0.967`, no-frame is `0.705`, shuffled-frame is `0.653`. In `multi_aspect_token_refraction`, full accuracy is `0.933`, no-frame is `0.671`, shuffled-frame is `0.608`. |
| Trained recurrent update | `necessary_core` | In `latent_refraction`, zero recurrent is `0.553` and randomized recurrent is `0.506` versus full `0.967`. In `multi_aspect`, zero is `0.540` and randomized is `0.530` versus full `0.933`. In temporal order contrast, zero is `0.320` and randomized is `0.480` versus full `1.000`. |
| Recurrent depth | `necessary_core` for refraction | Freezing early hurts: latent freeze after `1/2/3` gives `0.581/0.672/0.825` versus full `0.967`; multi-aspect gives `0.570/0.623/0.745` versus full `0.933`. |
| Order/role binding | `necessary_core` for sequence tasks | Temporal order contrast: streaming `1.000`, bag-of-tokens `0.500`, static-no-position `0.500`, shuffled order `0.492`. Same-token-set pair accuracy is `1.000` for streaming and `0.500` for bag. |
| Position-aware static encoding | `shortcut_risk` / alternative solution | Static-with-position reaches `1.000` on temporal order contrast. This proves order information is sufficient, but not uniquely streaming-specific. |
| Discrete predicted frame pointer as bottleneck | `nonessential` / `shortcut_risk` | Query-cued bottleneck: predicted pointer accuracy `0.641`; direct query bottleneck size `2` reaches `0.781` with stronger authority/refraction. Pointer-specific necessity is not supported. |
| Direct query/control path | `shortcut_risk` | Query conditioning can solve labels while bypassing the intended internal pointer geometry. Useful as a baseline, not as survivor evidence. |
| Hub-rich topology | `task_specific` / `helpful_bias` | Helps latent refraction: hub-rich update `0.3` gives accuracy `0.971`, refraction `0.457`, authority `0.445`, beating random sparse. It does not universally win on multi-aspect token refraction. |
| Hub degree concentration | `task_specific` / `helpful_bias` | Degree-preserving random hub masks recover much of the latent-refraction benefit and avoid some sampled-hub failures, but do not establish a universal topology prior. |
| Exact FlyWire / mushroom-body sampled wiring | `nonessential` | FlyWire sampled did not beat random sparse on the core authority/refraction metrics. No biology/FlyWire claim survives. |
| Fixed sin/cos / token-wave embeddings | `helpful_bias` | Fixed sin/cos improves multi-aspect accuracy `0.933 -> 0.939`, refraction `0.098 -> 0.106`, authority `0.123 -> 0.133`. Small useful bias only. |
| Explicit phase / pointer resonance machinery | `nonessential` | Pointer/neuron resonance variants are intervention-sensitive but do not improve authority/refraction over simpler token-wave or baseline recurrent models. |
| Clean nuisance erasure | `nonessential` as main claim | Earlier probes show core dominance/recovery and authority switching, not strong nuisance deletion. Decodability and influence must remain separate. |
| Decision-authority influence measurement | `necessary_core` for interpretation | Accuracy and decodability alone repeatedly overclaim. The survivor effects are visible only when output influence/refraction metrics are measured. |

## Results By Experiment

### 1. `latent_refraction`

Main run: hidden `64`, update rate `0.3`, seeds `5`.

| Path | Accuracy | Drop vs Full |
|---|---:|---:|
| full recurrent frame-conditioned | `0.967` | `0.000` |
| no frame token | `0.705` | `0.263` |
| shuffled frame token | `0.653` | `0.314` |
| zero recurrent | `0.553` | `0.414` |
| randomized recurrent | `0.506` | `0.461` |
| threshold only | `0.512` | `0.455` |
| random label | `0.494` | `0.473` |

Authority/refraction:

| Metric | Value |
|---|---:|
| refraction index final | `0.462` |
| authority switch score | `0.450` |
| recurrence gain | `0.414` |
| accuracy std across seeds | `0.0043` |

Depth knockout:

| Path | Accuracy |
|---|---:|
| freeze after step 1 | `0.581` |
| freeze after step 2 | `0.672` |
| freeze after step 3 | `0.825` |
| full | `0.967` |

Read:

```text
frame/control signal + trained recurrent dynamics + recurrent depth are necessary for the latent-refraction effect.
```

### 2. `multi_aspect_token_refraction`

Main run: hidden `128`, update rate `0.3`, seeds `5`.

| Path | Accuracy | Drop vs Full |
|---|---:|---:|
| full recurrent frame-conditioned | `0.933` | `0.000` |
| no frame token | `0.671` | `0.262` |
| shuffled frame token | `0.608` | `0.326` |
| zero recurrent | `0.540` | `0.393` |
| randomized recurrent | `0.530` | `0.403` |
| threshold only | `0.516` | `0.418` |
| random label | `0.502` | `0.432` |

Authority/refraction:

| Metric | Value |
|---|---:|
| refraction index final | `0.098` |
| authority switch score | `0.123` |
| mean actor switch | `0.309` |
| recurrence gain | `0.393` |
| accuracy std across seeds | `0.0145` |

Depth knockout:

| Path | Accuracy |
|---|---:|
| freeze after step 1 | `0.570` |
| freeze after step 2 | `0.623` |
| freeze after step 3 | `0.745` |
| full | `0.933` |

Dog authority by frame:

| Frame | Dog Output-Change Rate |
|---|---:|
| danger | `0.292` |
| friendship | `0.164` |
| sound | `0.528` |
| environment | `0.040` |

Read:

```text
same-token multi-aspect authority switching needs the frame/control signal and trained recurrent dynamics.
The effect is weaker and harder than group-level latent refraction, but the control knockouts are clean.
```

### 3. `temporal_order_contrast_refraction`

Main run: hidden `64`, update rate `0.3`, seeds `5`.

| Path | Accuracy | Same-Token-Set Pair Accuracy |
|---|---:|---:|
| streaming recurrent | `1.000` | `1.000` |
| bag of tokens | `0.500` | `0.500` |
| static no position | `0.500` | `0.500` |
| static with position | `1.000` | `1.000` |
| zero recurrent carry | `0.320` | `0.320` |
| shuffled order | `0.492` | `0.492` |
| randomized recurrent | `0.480` | `0.480` |
| random label | `0.100` | `n/a` |

Key gaps:

| Metric | Value |
|---|---:|
| bag failure gap | `0.500` |
| static-no-position gap | `0.500` |
| order sensitivity drop | `0.508` |
| zero recurrent drop | `0.680` |
| randomized recurrent drop | `0.520` |
| role authority score | `0.9996` |
| suffix resolution score | `0.307` |
| seed variance | `0.000` accuracy std |

Read:

```text
order/role binding is necessary beyond bag-of-token representations.
Streaming recurrence can learn it.
However, explicit position-aware static encoding also solves it, so streaming is not uniquely necessary when position binding is supplied directly.
```

## Survivor List

### Necessary Core

These survive the knockout matrix:

1. **Input representation**

   The system needs distinguishable feature/token vectors. This is basic substrate, not a special mechanism.

2. **Control/frame signal for frame-conditioned tasks**

   Removing or shuffling the frame/control signal strongly hurts latent and multi-aspect refraction.

3. **Trained recurrent state update**

   Zero/randomized recurrence collapses or heavily weakens all three survivor tasks.

4. **Recurrent depth**

   Early freezing shows that later recurrent passes are load-bearing in latent and multi-aspect refraction.

5. **Decision-authority readout**

   The core claims depend on output influence/refraction metrics, not just feature decodability or accuracy.

6. **Order/role binding for temporal tasks**

   Same-token-set contrast pairs force order to carry meaning. Bag/no-position encodings fail.

### Task-Specific

| Component | Read |
|---|---|
| hub-rich / degree-concentrated topology | Helps latent refraction, mixed or weak on multi-aspect. Load-bearing when used, but not universal. |
| reset/reframe trigger | Relevant to reframe diagnostics, not part of the three-task survivor core here. |
| explicit discrete frame pointer | Useful diagnostic tool, but not necessary when direct query paths solve the toy. |

### Helpful Bias

| Component | Read |
|---|---|
| fixed sin/cos / token-wave vectors | Small improvement in multi-aspect authority/refraction; not required. |
| hub degree concentration | Can improve or stabilize some recurrent masks; not a universal mechanism. |

### Shortcut Risk

| Component | Why |
|---|---|
| full direct query conditioning | Can solve labels while bypassing the intended internal pointer geometry. |
| static with position | Solves order contrast perfectly; valid strong baseline, but weakens any claim that streaming is uniquely necessary. |
| raw accuracy alone | Often high even when mechanism-specific metrics are weak. |
| decodability alone | Features can remain decodable while losing decision influence. |

### Nonessential / Falls Out

| Component | Reason |
|---|---|
| clean context/nuisance erasure | Current positives are core recovery, authority switching, and role binding, not strong erasure. |
| explicit phase embeddings | Learned/random and fixed sin/cos baselines already capture most of the effect. |
| explicit pointer/neuron resonance | Intervention-sensitive but not better on authority/refraction. |
| exact FlyWire sampled wiring | Did not beat random sparse on core authority/refraction metrics. |
| universal hub-rich topology | Task-specific positive only; not a universal winner. |
| discrete frame pointer necessity | Bottleneck tests show direct query paths can beat the pointer route. |
| free online prism rotation without reset | Frame-switch diagnostics did not support clean free mid-run rotation. |

## Final Minimal Mechanism Hypothesis

Current survivor mechanism:

```text
structured input representation
+ task/control signal when relevance must change by frame
+ trained sparse recurrent state update over multiple steps
+ decision-authority readout
+ order/role binding for streaming sequence tasks
```

More explicit version:

> In controlled toy settings, sparse recurrent dynamics can turn entangled or sequential inputs into decision states where the task-relevant feature group or role has high output authority. The necessary pieces are not clean nuisance erasure, explicit resonance machinery, or exact biological topology. The surviving core is control-conditioned recurrent dynamics plus authority-sensitive readout; for sequence tasks, temporal order/role binding is also necessary.

## Claim Boundary

Do not claim:

- consciousness,
- biology,
- full VRAXION behavior,
- production validation,
- natural-language understanding,
- FlyWire validation,
- clean context erasure.

Safe claim:

> Across the current toy probes, the most stable mechanism is trained recurrent decision-authority formation: frame/control signals determine relevance in refraction tasks, while ordered token arrival determines role binding in temporal tasks. Several attractive embellishments are only helpful biases or nonessential under knockout controls.

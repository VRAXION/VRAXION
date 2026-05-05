# Multi-Aspect Token Refraction Finding

Source run:

- `target/context-cancellation-probe/multi-aspect-h128-u03-v2/20260504T201450Z/multi_aspect_refraction_report.json`

## Hypothesis

**Same token, different frame, different decision authority**

The probe tests whether a reused actor token can retain multiple possible aspects while a task frame decides which aspect gets decision authority. The intended behavior is not a contradictory mapping such as:

```text
dog -> danger
dog -> friend
```

Instead, the task is frame-conditioned:

```text
dog + bite + danger_frame -> danger
dog + owner/play + friendship_frame -> friend
dog + bark + sound_frame -> sound_source
dog + street/noise + environment_frame -> actor is distractor
```

## Setup

Every base observation contains an entangled bundle of:

- actor,
- action,
- relation,
- sound,
- place/noise,
- object.

The same observation is evaluated under four frames:

- `danger_frame`: label depends on `actor_action`.
- `friendship_frame`: label depends on `actor_relation`.
- `sound_frame`: label depends on `actor_sound`.
- `environment_frame`: label depends on `place_noise`; actor is a distractor.

The main intervention keeps the rest of the observation fixed and swaps the actor token. This measures whether the same actor token has high influence when actor identity is causal and low influence when actor identity is irrelevant.

## Key Results

Run settings:

- experiment: `multi_aspect_refraction`
- input mode: `entangled`
- hidden: `128`
- update rate: `0.3`
- recurrent steps: `5`
- epochs: `220`
- seeds: `5`

Accuracy:

- overall accuracy: `0.933333`
- `danger_frame`: `0.926667`
- `friendship_frame`: `0.946000`
- `sound_frame`: `0.888000`
- `environment_frame`: `0.972667`

Controls:

- zero-recurrent accuracy: `0.540000`
- threshold-only accuracy: `0.515833`
- freeze-after-step-1 accuracy: `0.569833`
- freeze-after-step-2 accuracy: `0.622833`
- freeze-after-step-3 accuracy: `0.744500`
- shuffled-frame-token accuracy: `0.607500`
- no-frame-token accuracy: `0.671333`
- randomized-recurrent accuracy: `0.530333`
- random-label accuracy: `0.501500`

The recurrence gain over the zero-recurrent baseline is `+0.393333`.

## Dog Influence By Frame

Actor-token intervention for `dog`: keep the rest of the observation fixed and replace the actor token.

| Frame | output_change_rate | label_change_rate | target_accuracy | mean_abs_label_probability_delta |
|---|---:|---:|---:|---:|
| `danger_frame` | 0.292000 | 0.400000 | 0.796000 | 0.298601 |
| `friendship_frame` | 0.164000 | 0.248000 | 0.876000 | 0.171222 |
| `sound_frame` | 0.528000 | 0.592000 | 0.828000 | 0.517937 |
| `environment_frame` | 0.040000 | 0.000000 | 0.980000 | 0.055328 |

Read:

- `dog` has substantial decision influence in actor-causal frames.
- `dog` has very low decision influence in `environment_frame`, where actor identity is distractor.
- The strongest `dog` influence appears in `sound_frame`, where actor identity determines the expected sound.

## Actor Authority Switch Scores

Authority switch score compares actor influence in actor-causal frames against actor influence in `environment_frame`.

```json
{
  "dog": 0.288000,
  "cat": 0.254667,
  "bird": 0.337333,
  "child": 0.300000,
  "robot": 0.313333,
  "snake": 0.361333
}
```

Mean actor authority switch score: `0.309111`

Actor-token decodability by step:

```json
[
  1.000000,
  1.000000,
  1.000000,
  1.000000,
  0.999167,
  0.995167
]
```

The actor token remains decodable while its decision influence changes by frame. This separates "information is still present" from "information has decision authority."

## Refraction Index

Mean refraction index by step:

```json
[
  -0.038167,
  -0.057167,
  -0.029000,
  0.021667,
  0.081333,
  0.098167
]
```

This is positive but modest. The earlier feature-group latent refraction probe had a much stronger final refraction index of about `0.443333`. This multi-aspect token task is harder and noisier because the same token participates in multiple frame-conditioned relations.

## Interpretation

```json
{
  "supports_multi_aspect_token_refraction": "true",
  "supports_same_token_different_frame_authority": "true"
}
```

This is a positive toy result for **Multi-Aspect Token Refraction** / **Frame-Conditioned Token Authority Switching**.

The useful interpretation is:

> In a controlled toy setting, the same actor token can retain multiple frame-conditioned aspects, gaining decision authority in frames where actor identity is causal and losing it when actor identity is distractor.

## Limitations

- The refraction index is modest: final value `0.098167`.
- `sound_frame` is the weakest accuracy frame at `0.888000`.
- The result is weaker and noisier than the simpler feature-group latent refraction result.
- The current probe does not yet prove that the hidden state can dynamically reorient under a mid-run frame switch.
- The result could still partly reflect learned frame-label rules rather than a fully general latent reorientation mechanism.

## Claim Boundary

Toy evidence only. Do not claim:

- consciousness,
- biological equivalence,
- full VRAXION behavior,
- production architecture validation,
- that the full VRAXION architecture already implements this mechanism.

Safe claim:

> In a controlled toy setting, the same actor token can retain multiple frame-conditioned aspects, gaining decision authority in frames where actor identity is causal and losing it when actor identity is distractor.

## Next Test

The next useful probe is a mid-run frame switch:

```text
same dog observation
start under danger_frame
switch during recurrence to environment_frame
measure whether dog influence drops over subsequent recurrent steps
```

If the model can reorient after a mid-run frame change, that would strengthen the "rotating prism" interpretation beyond static frame-conditioned classification.

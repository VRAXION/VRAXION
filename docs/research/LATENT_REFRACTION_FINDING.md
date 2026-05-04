# Latent Refraction Finding

Source run:

- `target/context-cancellation-probe/20260504T194024Z/latent_refraction_report.json`

Robustness grid:

- `target/context-cancellation-probe/v6-robustness-grid/`

## Old Finding Summary

The previous toy finding supported **Recurrent Core Recovery under Entangled Interference**:

- recurrence can recover a task-causal core from entangled core+nuisance input,
- nuisance can remain decodable,
- decision authority can still shift toward the recovered core.

That result did not fully test the prism idea, because each feature group kept the same role across the task.

## New Hypothesis

**Recurrent Latent Refraction / Task-Frame Conditional Core Dominance**

The same observed feature bundle should be reinterpreted depending on a task-frame token. A feature group that is nuisance in one frame should become causal core in another frame. The recurrent loop should reorient the hidden state so the active task-core gains decision authority while inactive groups remain decodable but output-inert.

## Frame Task Setup

Every base observation is evaluated under three frames:

- `danger_frame`: label depends on `actor_action`.
- `environment_frame`: label depends on `place_noise`.
- `visibility_frame`: label depends on `light`.

Object features are included as an always-inactive distractor in this v6 version. This keeps the first prism test small and falsifiable.

## Accuracy Results

- input mode: `entangled`
- overall accuracy: `0.955111`
- zero-recurrent accuracy: `0.577111`
- recurrence gain: `0.378000`
- no-frame-token accuracy: `0.708000`
- shuffled-frame-token accuracy: `0.644444`
- randomized-recurrent accuracy: `0.513111`
- random-label accuracy: `0.493556`
- same-observation label diversity: `0.755333`

Accuracy by frame:

```json
{
  "danger_frame": 0.920667,
  "environment_frame": 0.953333,
  "visibility_frame": 0.991333
}
```

## Influence Table

Final-step output-change rate when swapping each feature group while holding the rest fixed:

| Frame | Active group | actor_action | place_noise | light | object |
|---|---|---:|---:|---:|---:|
| danger_frame | actor_action | 0.5067 | 0.0873 | 0.0640 | 0.0527 |
| environment_frame | place_noise | 0.0787 | 0.5047 | 0.0493 | 0.0387 |
| visibility_frame | light | 0.0127 | 0.0140 | 0.5020 | 0.0080 |

## Refraction Index

Definition:

```text
refraction_index_by_step = active_core_output_change_rate - max(inactive_group_output_change_rate)
```

Mean refraction index by step:

```json
[
  -0.000667,
  0.003778,
  0.091778,
  0.240444,
  0.385556,
  0.443333
]
```

Authority switch score:

```json
{
  "actor_action": 0.428,
  "place_noise": 0.417333,
  "light": 0.434,
  "object": null
}
```

Mean authority switch score: `0.426444`

## Robustness Grid

Grid settings:

- experiment: `latent_refraction`
- input mode: `entangled`
- seeds: `5`
- hidden: `32`, `64`
- update rate: `0.1`, `0.2`, `0.3`
- steps: `5`
- epochs: `200`
- train size: `1800`
- test size: `900`

Report-only metrics:

| hidden | update_rate | accuracy | zero_recurrent | randomized_recurrent | no_frame_token | shuffled_frame_token | refraction_index_final | authority_switch_score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.1 | 0.746667 | 0.575333 | 0.552889 | 0.671333 | 0.596667 | 0.177111 | 0.140222 |
| 32 | 0.2 | 0.751333 | 0.560889 | 0.540889 | 0.671111 | 0.587556 | 0.182667 | 0.151556 |
| 32 | 0.3 | 0.763556 | 0.545778 | 0.540667 | 0.672222 | 0.597778 | 0.196667 | 0.155556 |
| 64 | 0.1 | 0.930444 | 0.571778 | 0.544000 | 0.708222 | 0.643556 | 0.407556 | 0.381778 |
| 64 | 0.2 | 0.955111 | 0.577111 | 0.513111 | 0.708000 | 0.644444 | 0.443333 | 0.426444 |
| 64 | 0.3 | 0.967333 | 0.553111 | 0.506222 | 0.704667 | 0.652889 | 0.462000 | 0.450000 |

Read:

- `hidden=32` gives a partial but fragile signal: recurrence helps and refraction index rises, but accuracy and authority switching stay below the positive threshold.
- `hidden=64` gives a stable positive signal across all three update rates.
- Higher update rate improves the v6 signal in this grid, with `hidden=64, update_rate=0.3` strongest on accuracy, refraction index, and authority switch score.

## Controls

- `zero_recurrent_update`: tests whether recurrence is carrying the frame-conditioned computation.
- `no_task_frame_token`: tests whether identical observations with different frame labels are unsolvable without the frame.
- `shuffled_task_frame_token`: tests whether the trained model actually uses the frame token.
- `freeze_after_1/2/3`: tests whether recurrent depth matters.
- `randomize_recurrent_matrix`: tests whether the learned recurrent matrix carries the useful dynamic.
- `random_label_control`: sanity check against memorizing arbitrary frame labels.

## Interpretation

```json
{
  "supports_recurrent_latent_refraction": "true",
  "supports_task_frame_conditional_core_dominance": "true",
  "reason": "The same entangled observations are solved under multiple task frames, recurrence beats the zero-recurrent baseline, removing or shuffling the frame token hurts, randomizing recurrence destroys the gain, active-group influence separates from inactive-group influence over steps, and feature groups show higher authority when causal than when nuisance."
}
```

## Claim Boundary

Toy evidence only. Do not claim consciousness, full VRAXION behavior, production architecture validation, clean nuisance erasure, or biological equivalence.

Safe claim if positive:

> In a controlled toy setting, the recurrent loop can reorient an entangled representation according to a task frame, giving decision authority to different feature groups without necessarily erasing the others.

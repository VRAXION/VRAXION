# Frame Switch Refraction Diagnostic

Source run:

- `target/context-cancellation-probe/frame-switch-h128-u03-v2/20260504T204008Z/frame_switch_refraction_diagnostic_report.json`

## Question

This diagnostic tests whether **Multi-Aspect Token Refraction** is real recurrent reorientation or mostly static frame-token routing.

The setup reuses the existing multi-aspect task:

- `danger_frame`
- `friendship_frame`
- `sound_frame`
- `environment_frame`
- same actor tokens, especially `dog`
- entangled input

No new semantic concepts are added.

## Architecture Diagnostic

Frame placement comparison:

| Frame placement mode | accuracy |
|---|---:|
| `frame_in_recurrence_only` | 0.915833 |
| `frame_initial_only` | 0.910833 |
| `frame_at_output_only` | 0.670833 |
| `no_frame` | 0.664583 |

Main comparison:

- frame-in-recurrence-only accuracy: `0.915833`
- frame-at-output-only accuracy: `0.670833`
- recurrent-vs-output-only gain: `0.245000`
- no-frame accuracy: `0.664583`
- zero-recurrent accuracy: `0.522083`
- randomized-recurrent accuracy: `0.497500`
- shuffled-frame-token accuracy: `0.596250`
- random-label accuracy: `0.498333`

## Mid-Run Frame Switch

Switch pairs:

```json
[
  "danger_frame->environment_frame",
  "environment_frame->danger_frame",
  "danger_frame->sound_frame",
  "sound_frame->danger_frame",
  "friendship_frame->environment_frame"
]
```

Aggregate switch metrics:

- mid-run switch success rate: `0.0`
- reorientation score: `-0.104726`
- target-frame convergence after switch: `-0.061039`
- authority switch after frame switch: `-0.202583`

Definition notes:

- `reorientation_score = final_distance_to_old_source_trajectory - final_distance_to_new_target_trajectory`.
- Positive means the switched hidden state ends closer to the direct target-frame trajectory than to the old source-frame trajectory.
- `authority_switch_after_frame_switch` compares new-active-group influence against old-active-group influence after the switch.

## Hidden Trajectory Geometry

Trajectory divergence by step:

```json
[
  0.0,
  0.075352,
  0.171241,
  0.277226,
  0.379434,
  0.460238
]
```

Frame clustering accuracy by step:

```json
[
  0.25,
  0.6425,
  0.76,
  0.789167,
  0.809167,
  0.811667
]
```

Final frame clustering accuracy: `0.811667`

## Soft Frame Interpolation

Interpolation pair:

```json
{
  "source": "danger_frame",
  "target": "environment_frame"
}
```

Interpolation smoothness score: `0.973458`

Mix summary:

```json
{
  "danger_1.00_environment_0.00": {
    "danger_weight": 1.0,
    "environment_weight": 0.0,
    "mean_class_1_probability_by_step": [
      0.797799,
      0.642947,
      0.533583,
      0.495635,
      0.510153,
      0.52576
    ],
    "actor_action_output_change": 0.463333,
    "place_noise_output_change": 0.088333,
    "influence_balance_place_noise_minus_actor_action": -0.375,
    "hidden_position_between_danger_and_environment": 0.0,
    "distance_to_danger_direct_final": 0.0,
    "distance_to_environment_direct_final": 0.172189
  },
  "danger_0.75_environment_0.25": {
    "danger_weight": 0.75,
    "environment_weight": 0.25,
    "mean_class_1_probability_by_step": [
      0.797799,
      0.645757,
      0.54623,
      0.515188,
      0.530468,
      0.548933
    ],
    "actor_action_output_change": 0.466667,
    "place_noise_output_change": 0.155,
    "influence_balance_place_noise_minus_actor_action": -0.311667,
    "hidden_position_between_danger_and_environment": 0.09679,
    "distance_to_danger_direct_final": 0.011573,
    "distance_to_environment_direct_final": 0.109006
  },
  "danger_0.50_environment_0.50": {
    "danger_weight": 0.5,
    "environment_weight": 0.5,
    "mean_class_1_probability_by_step": [
      0.797799,
      0.648663,
      0.559685,
      0.530905,
      0.512943,
      0.512267
    ],
    "actor_action_output_change": 0.328333,
    "place_noise_output_change": 0.358333,
    "influence_balance_place_noise_minus_actor_action": 0.03,
    "hidden_position_between_danger_and_environment": 0.488323,
    "distance_to_danger_direct_final": 0.048906,
    "distance_to_environment_direct_final": 0.051603
  },
  "danger_0.25_environment_0.75": {
    "danger_weight": 0.25,
    "environment_weight": 0.75,
    "mean_class_1_probability_by_step": [
      0.797799,
      0.651652,
      0.573549,
      0.540309,
      0.477591,
      0.451455
    ],
    "actor_action_output_change": 0.136667,
    "place_noise_output_change": 0.461667,
    "influence_balance_place_noise_minus_actor_action": 0.325,
    "hidden_position_between_danger_and_environment": 0.896915,
    "distance_to_danger_direct_final": 0.107247,
    "distance_to_environment_direct_final": 0.012367
  },
  "danger_0.00_environment_1.00": {
    "danger_weight": 0.0,
    "environment_weight": 1.0,
    "mean_class_1_probability_by_step": [
      0.797799,
      0.654709,
      0.587367,
      0.552655,
      0.478004,
      0.457397
    ],
    "actor_action_output_change": 0.071667,
    "place_noise_output_change": 0.48,
    "influence_balance_place_noise_minus_actor_action": 0.408333,
    "hidden_position_between_danger_and_environment": 1.0,
    "distance_to_danger_direct_final": 0.172189,
    "distance_to_environment_direct_final": 0.0
  }
}
```

## Dog Influence By Direct Frame

```json
{
  "danger_frame": {
    "label_change_rate": 0.39,
    "output_change_rate": 0.33,
    "target_accuracy": 0.86,
    "original_label_retention": 0.69,
    "mean_abs_label_probability_delta": 0.32144,
    "mean_kl_divergence": 1.934702,
    "mean_abs_margin_delta": 5.582285,
    "output_change_rate_by_step": [
      0.04,
      0.06,
      0.15,
      0.2,
      0.23,
      0.33
    ],
    "target_accuracy_by_step": [
      0.37,
      0.51,
      0.58,
      0.7,
      0.81,
      0.86
    ],
    "original_label_retention_by_step": [
      0.66,
      0.74,
      0.73,
      0.81,
      0.76,
      0.69
    ],
    "mean_abs_label_probability_delta_by_step": [
      0.0468,
      0.087139,
      0.141729,
      0.186144,
      0.236044,
      0.32144
    ],
    "mean_kl_divergence_by_step": [
      0.0396,
      0.189625,
      0.394893,
      0.792074,
      1.273876,
      1.934702
    ],
    "mean_abs_margin_delta_by_step": [
      0.981018,
      2.270356,
      2.458933,
      3.273256,
      4.433073,
      5.582285
    ]
  },
  "friendship_frame": {
    "label_change_rate": 0.26,
    "output_change_rate": 0.21,
    "target_accuracy": 0.89,
    "original_label_retention": 0.79,
    "mean_abs_label_probability_delta": 0.210787,
    "mean_kl_divergence": 1.068904,
    "mean_abs_margin_delta": 6.764013,
    "output_change_rate_by_step": [
      0.04,
      0.11,
      0.13,
      0.17,
      0.18,
      0.21
    ],
    "target_accuracy_by_step": [
      0.39,
      0.51,
      0.62,
      0.74,
      0.86,
      0.89
    ],
    "original_label_retention_by_step": [
      0.59,
      0.67,
      0.74,
      0.76,
      0.8,
      0.79
    ],
    "mean_abs_label_probability_delta_by_step": [
      0.034511,
      0.106347,
      0.130309,
      0.172875,
      0.202473,
      0.210787
    ],
    "mean_kl_divergence_by_step": [
      0.019641,
      0.164997,
      0.302558,
      0.40047,
      0.685236,
      1.068904
    ],
    "mean_abs_margin_delta_by_step": [
      1.067919,
      2.531502,
      2.71134,
      3.618222,
      5.273594,
      6.764013
    ]
  },
  "sound_frame": {
    "label_change_rate": 0.57,
    "output_change_rate": 0.42,
    "target_accuracy": 0.76,
    "original_label_retention": 0.43,
    "mean_abs_label_probability_delta": 0.429418,
    "mean_kl_divergence": 2.564016,
    "mean_abs_margin_delta": 4.102498,
    "output_change_rate_by_step": [
      0.03,
      0.09,
      0.11,
      0.21,
      0.42,
      0.42
    ],
    "target_accuracy_by_step": [
      0.16,
      0.23,
      0.35,
      0.5,
      0.7,
      0.76
    ],
    "original_label_retention_by_step": [
      0.47,
      0.56,
      0.56,
      0.57,
      0.45,
      0.43
    ],
    "mean_abs_label_probability_delta_by_step": [
      0.039688,
      0.098649,
      0.12494,
      0.199103,
      0.361173,
      0.429418
    ],
    "mean_kl_divergence_by_step": [
      0.030711,
      0.18668,
      0.252354,
      0.546616,
      1.640556,
      2.564016
    ],
    "mean_abs_margin_delta_by_step": [
      0.883107,
      2.123491,
      2.604742,
      2.86503,
      3.494271,
      4.102498
    ]
  },
  "environment_frame": {
    "label_change_rate": 0.0,
    "output_change_rate": 0.04,
    "target_accuracy": 0.96,
    "original_label_retention": 0.96,
    "mean_abs_label_probability_delta": 0.044004,
    "mean_kl_divergence": 0.126478,
    "mean_abs_margin_delta": 5.017796,
    "output_change_rate_by_step": [
      0.02,
      0.1,
      0.08,
      0.16,
      0.14,
      0.04
    ],
    "target_accuracy_by_step": [
      0.47,
      0.48,
      0.52,
      0.64,
      0.85,
      0.96
    ],
    "original_label_retention_by_step": [
      0.47,
      0.48,
      0.52,
      0.64,
      0.85,
      0.96
    ],
    "mean_abs_label_probability_delta_by_step": [
      0.037467,
      0.091304,
      0.099428,
      0.153807,
      0.129834,
      0.044004
    ],
    "mean_kl_divergence_by_step": [
      0.025504,
      0.151564,
      0.241413,
      0.515901,
      0.324868,
      0.126478
    ],
    "mean_abs_margin_delta_by_step": [
      1.092758,
      2.586103,
      2.722367,
      2.707767,
      3.711015,
      5.017796
    ]
  }
}
```

## Verdict

```json
{
  "real_recurrent_reorientation": "unclear",
  "static_frame_routing": "false",
  "early_frame_commitment": "true",
  "reason": "frame_in_recurrence_only_accuracy=0.916, frame_at_output_only_accuracy=0.671, frame_initial_only_accuracy=0.911, randomized_recurrent_accuracy=0.497, shuffled_frame_accuracy=0.59625, mid_run_switch_success_rate=0.0, reorientation_score=-0.10472612828016281, target_frame_convergence_after_switch=-0.06103934198617935, interpolation_smoothness_score=0.9734582135322523, authority_switch_after_frame_switch=-0.20258333333333334."
}
```

## Claim Boundary

Toy diagnostic only. Do not claim consciousness, biology, full VRAXION behavior, or production architecture validation.

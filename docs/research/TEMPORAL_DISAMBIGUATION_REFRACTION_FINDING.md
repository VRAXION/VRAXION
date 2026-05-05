# Temporal Disambiguation Refraction Finding

Source run:

- `target/context-cancellation-probe/temporal-disambiguation-refraction/20260505T212937Z/temporal_disambiguation_refraction_report.json`

## Hypothesis

Streaming recurrence should not treat a short sentence as only a static bag of tokens. Earlier tokens can start partial frame trajectories, and delayed suffix tokens can confirm, redirect, or suppress those trajectories.

## Dataset Examples

```json
[
  {
    "name": "dog_bit_me",
    "tokens": [
      "dog",
      "bit",
      "me"
    ],
    "frame": "user_injury_danger"
  },
  {
    "name": "dog_bit_his_tail",
    "tokens": [
      "dog",
      "bit",
      "his_tail"
    ],
    "frame": "animal_behavior"
  },
  {
    "name": "dog_bit_the_toy",
    "tokens": [
      "dog",
      "bit",
      "the_toy"
    ],
    "frame": "play_object"
  },
  {
    "name": "dog_bit_the_enemy",
    "tokens": [
      "dog",
      "bit",
      "the_enemy"
    ],
    "frame": "combat_threat"
  },
  {
    "name": "dog_bit_a_child",
    "tokens": [
      "dog",
      "bit",
      "a_child"
    ],
    "frame": "user_injury_danger"
  },
  {
    "name": "dog_barked_at_me",
    "tokens": [
      "dog",
      "barked",
      "at_me"
    ],
    "frame": "user_injury_danger"
  },
  {
    "name": "dog_barked_at_nothing",
    "tokens": [
      "dog",
      "barked",
      "at_nothing"
    ],
    "frame": "animal_behavior"
  },
  {
    "name": "dog_barked_at_the_door",
    "tokens": [
      "dog",
      "barked",
      "at_the_door"
    ],
    "frame": "environment_alert"
  },
  {
    "name": "dog_barked_happily",
    "tokens": [
      "dog",
      "barked",
      "happily"
    ],
    "frame": "social_friendship"
  },
  {
    "name": "dog_followed_me_home",
    "tokens": [
      "dog",
      "followed",
      "me_home"
    ],
    "frame": "social_friendship"
  },
  {
    "name": "dog_followed_a_scent",
    "tokens": [
      "dog",
      "followed",
      "a_scent"
    ],
    "frame": "animal_behavior"
  },
  {
    "name": "dog_followed_the_owner",
    "tokens": [
      "dog",
      "followed",
      "the_owner"
    ],
    "frame": "social_friendship"
  }
]
```

## Accuracy And Controls

| Path | Accuracy |
|---|---:|
| streaming recurrent | `1.000000` |
| bag of tokens baseline | `1.000000` |
| full sentence static | `1.000000` |
| zero recurrent carry | `0.934000` |
| shuffled order | `0.693250` |
| randomized recurrent | `0.801250` |
| random label | `0.1595` |

## Prefix Ambiguity

`dog bit` prefix metrics:

```json
{
  "prefix": "dog bit",
  "entropy": 0.967161,
  "normalized_entropy": 0.539783,
  "top_frame": "user_injury_danger",
  "top_probability": 0.621197,
  "candidate_count_prob_ge_0_15": 1.6,
  "uniform_normalized_entropy_reference": 1.0,
  "frame_probabilities": {
    "user_injury_danger": 0.475844,
    "animal_behavior": 0.356793,
    "play_object": 0.023768,
    "combat_threat": 0.017495,
    "social_friendship": 0.112519,
    "environment_alert": 0.013581
  }
}
```

Average `dog bit` frame distribution:

```json
{
  "count": 335.0,
  "entropy": 0.967161,
  "normalized_entropy": 0.539783,
  "frame_probabilities": {
    "user_injury_danger": 0.475844,
    "animal_behavior": 0.356793,
    "play_object": 0.023768,
    "combat_threat": 0.017495,
    "social_friendship": 0.112519,
    "environment_alert": 0.013581
  },
  "top_frame": "user_injury_danger",
  "top_probability": 0.621197
}
```

## Dog-Bit Suffix Case Study

| Completion | Target Frame | Prefix Correct Prob | Final Correct Prob | Correct Rise | Danger Prefix | Danger Final | Danger Drop |
|---|---|---:|---:|---:|---:|---:|---:|
| `a_child` | `user_injury_danger` | `0.475845` | `0.996427` | `0.520582` | `0.475845` | `0.996427` | `-0.520582` |
| `the_enemy` | `combat_threat` | `0.017495` | `0.99543` | `0.977935` | `0.475845` | `0.003064` | `0.472781` |
| `his_tail` | `animal_behavior` | `0.356793` | `0.993363` | `0.636571` | `0.475845` | `0.003341` | `0.472504` |
| `the_toy` | `play_object` | `0.023768` | `0.995509` | `0.971741` | `0.475845` | `0.002353` | `0.473493` |
| `me` | `user_injury_danger` | `0.475845` | `0.995851` | `0.520006` | `0.475845` | `0.995851` | `-0.520006` |

## Delayed Evidence Divergence

```json
{
  "pair": "dog bit me vs dog bit his_tail",
  "hidden_cosine_distance_by_step": [
    0.0,
    -0.0,
    0.197038
  ],
  "output_probability_l1_by_step": [
    0.0,
    0.0,
    1.985334
  ],
  "early_shared_distance": -0.0,
  "final_divergence": 0.197038
}
```

## Verdict

```json
{
  "supports_temporal_disambiguation": true,
  "supports_prefix_frame_ambiguity": "false",
  "supports_suffix_frame_resolution": "true",
  "supports_order_sensitive_recurrence": "true",
  "supports_delayed_trajectory_divergence": true,
  "bag_or_static_shortcut_present": true,
  "geometry_read": "suffix resolution is visible, but the order/trajectory evidence is incomplete",
  "reason": "streaming=1.000, bag=1.000, static=1.000, zero=0.934, shuffled=0.693, randomized=0.801, random_label=0.15949999999999998, prefix_entropy=0.5397827982943328, prefix_candidates=1.6, suffix_rise=0.7253669881820678, suffix_drop=0.47292581796646116, early_distance=-4.76837158203125e-08, final_divergence=0.197038471698761."
}
```

## Claim Boundary

Toy evidence only. Do not claim consciousness, biology, full VRAXION behavior, production validation, or natural-language understanding.

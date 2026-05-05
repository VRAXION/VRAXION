# Temporal Order-Contrast Refraction Finding

Source run:

- `target/context-cancellation-probe/temporal-order-contrast-refraction/20260505T214443Z/temporal_order_contrast_refraction_report.json`

## Hypothesis

Order should carry role/frame meaning when the same token set appears in different sequences. A bag-of-token baseline should fail because `dog bit me` and `me bit dog` contain the same tokens but require different labels.

## Contrastive Dataset

```json
[
  {
    "name": "dog_bit_me",
    "tokens": [
      "dog",
      "bit",
      "me"
    ],
    "frame": "user_injury",
    "pair": "dog_me_bit"
  },
  {
    "name": "me_bit_dog",
    "tokens": [
      "me",
      "bit",
      "dog"
    ],
    "frame": "user_attacks_animal",
    "pair": "dog_me_bit"
  },
  {
    "name": "dog_chased_cat",
    "tokens": [
      "dog",
      "chased",
      "cat"
    ],
    "frame": "dog_agent",
    "pair": "dog_cat_chased"
  },
  {
    "name": "cat_chased_dog",
    "tokens": [
      "cat",
      "chased",
      "dog"
    ],
    "frame": "cat_agent",
    "pair": "dog_cat_chased"
  },
  {
    "name": "dog_followed_owner",
    "tokens": [
      "dog",
      "followed",
      "owner"
    ],
    "frame": "companion_follow",
    "pair": "dog_owner_followed"
  },
  {
    "name": "owner_followed_dog",
    "tokens": [
      "owner",
      "followed",
      "dog"
    ],
    "frame": "owner_tracking_dog",
    "pair": "dog_owner_followed"
  },
  {
    "name": "snake_bit_dog",
    "tokens": [
      "snake",
      "bit",
      "dog"
    ],
    "frame": "dog_injured",
    "pair": "dog_snake_bit"
  },
  {
    "name": "dog_bit_snake",
    "tokens": [
      "dog",
      "bit",
      "snake"
    ],
    "frame": "dog_attacks_snake",
    "pair": "dog_snake_bit"
  },
  {
    "name": "child_scared_dog",
    "tokens": [
      "child",
      "scared",
      "dog"
    ],
    "frame": "dog_target",
    "pair": "child_dog_scared"
  },
  {
    "name": "dog_scared_child",
    "tokens": [
      "dog",
      "scared",
      "child"
    ],
    "frame": "child_target",
    "pair": "child_dog_scared"
  }
]
```

## Accuracy And Controls

| Path | Accuracy | Same-Token-Set Pair Accuracy |
|---|---:|---:|
| streaming recurrent | `1.000000` | `1.000000` |
| bag of tokens baseline | `0.500000` | `0.500000` |
| full sentence static with position | `1.000000` | `1.000000` |
| full sentence static no position | `0.500000` | `0.500000` |
| zero recurrent carry | `0.320000` | `0.320000` |
| shuffled order | `0.492000` | `0.492000` |
| randomized recurrent | `0.480000` | `0.480000` |
| random label | `0.09975` | `n/a` |

Gaps:

```json
{
  "bag_failure_gap": 0.5,
  "static_no_position_gap": 0.5,
  "position_baseline_gap": 0.0,
  "order_sensitivity_drop": 0.508,
  "zero_recurrent_drop": 0.68,
  "randomized_recurrent_drop": 0.52
}
```

## Same-Token-Set Pair Results

| Token-Set Pair | Streaming Pair Acc | Bag Pair Acc | Final Role Gap |
|---|---:|---:|---:|
| `child_dog_scared` | `1.0` | `0.5` | `0.999871` |
| `dog_cat_chased` | `1.0` | `0.5` | `0.999875` |
| `dog_me_bit` | `1.0` | `0.5` | `0.999215` |
| `dog_owner_followed` | `1.0` | `0.5` | `0.999871` |
| `dog_snake_bit` | `1.0` | `0.5` | `0.999153` |

## Role Authority By Step

```json
{
  "by_token_set_pair": {
    "child_dog_scared": {
      "labels": [
        "dog_target",
        "child_target"
      ],
      "correct_role_gap_by_step": [
        0.008611,
        0.751272,
        0.999871
      ],
      "role_probability_l1_by_step": [
        0.668145,
        1.853693,
        1.999904
      ]
    },
    "dog_cat_chased": {
      "labels": [
        "dog_agent",
        "cat_agent"
      ],
      "correct_role_gap_by_step": [
        0.046319,
        0.787824,
        0.999875
      ],
      "role_probability_l1_by_step": [
        0.622007,
        1.858304,
        1.999913
      ]
    },
    "dog_me_bit": {
      "labels": [
        "user_injury",
        "user_attacks_animal"
      ],
      "correct_role_gap_by_step": [
        -0.338214,
        0.433397,
        0.999215
      ],
      "role_probability_l1_by_step": [
        1.376912,
        1.574951,
        1.999905
      ]
    },
    "dog_owner_followed": {
      "labels": [
        "companion_follow",
        "owner_tracking_dog"
      ],
      "correct_role_gap_by_step": [
        0.026516,
        0.727825,
        0.999871
      ],
      "role_probability_l1_by_step": [
        0.811397,
        1.820425,
        1.999909
      ]
    },
    "dog_snake_bit": {
      "labels": [
        "dog_injured",
        "dog_attacks_snake"
      ],
      "correct_role_gap_by_step": [
        -0.30522,
        0.446556,
        0.999153
      ],
      "role_probability_l1_by_step": [
        1.285622,
        1.527837,
        1.999904
      ]
    }
  },
  "mean_final_correct_role_gap": 0.999597
}
```

## Prefix Trajectory Similarity

```json
{
  "shared_prefix_dog_bit": {
    "pair": "dog_bit_me vs dog_bit_snake",
    "hidden_cosine_distance_by_step": [
      -0.0,
      0.0,
      0.064981
    ],
    "output_probability_l1_by_step": [
      0.0,
      0.0,
      1.994161
    ],
    "early_shared_distance": -0.0,
    "final_divergence": 0.064981
  },
  "shared_initial_dog": {
    "pair": "dog_chased_cat vs dog_followed_owner",
    "hidden_cosine_distance_by_step": [
      -0.0,
      0.321682,
      0.525311
    ],
    "output_probability_l1_by_step": [
      0.0,
      1.369554,
      1.999719
    ],
    "early_shared_distance": -0.0,
    "final_divergence": 0.525311
  }
}
```

## Suffix Resolution

```json
{
  "mean_correct_frame_probability_rise": 0.306956,
  "by_sequence": {
    "cat_chased_dog": {
      "target_frame": "cat_agent",
      "prefix_correct_probability": 0.922646,
      "final_correct_probability": 0.999957,
      "correct_probability_rise": 0.077311
    },
    "child_scared_dog": {
      "target_frame": "dog_target",
      "prefix_correct_probability": 0.929733,
      "final_correct_probability": 0.999955,
      "correct_probability_rise": 0.070222
    },
    "dog_bit_me": {
      "target_frame": "user_injury",
      "prefix_correct_probability": 0.336335,
      "final_correct_probability": 0.998519,
      "correct_probability_rise": 0.662184
    },
    "dog_bit_snake": {
      "target_frame": "dog_attacks_snake",
      "prefix_correct_probability": 0.368098,
      "final_correct_probability": 0.998392,
      "correct_probability_rise": 0.630294
    },
    "dog_chased_cat": {
      "target_frame": "dog_agent",
      "prefix_correct_probability": 0.703761,
      "final_correct_probability": 0.999836,
      "correct_probability_rise": 0.296075
    },
    "dog_followed_owner": {
      "target_frame": "companion_follow",
      "prefix_correct_probability": 0.605735,
      "final_correct_probability": 0.999828,
      "correct_probability_rise": 0.394093
    },
    "dog_scared_child": {
      "target_frame": "child_target",
      "prefix_correct_probability": 0.625537,
      "final_correct_probability": 0.999834,
      "correct_probability_rise": 0.374297
    },
    "me_bit_dog": {
      "target_frame": "user_attacks_animal",
      "prefix_correct_probability": 0.764478,
      "final_correct_probability": 0.999952,
      "correct_probability_rise": 0.235474
    },
    "owner_followed_dog": {
      "target_frame": "owner_tracking_dog",
      "prefix_correct_probability": 0.908835,
      "final_correct_probability": 0.999958,
      "correct_probability_rise": 0.091122
    },
    "snake_bit_dog": {
      "target_frame": "dog_injured",
      "prefix_correct_probability": 0.761473,
      "final_correct_probability": 0.999957,
      "correct_probability_rise": 0.238483
    }
  }
}
```

## Verdict

```json
{
  "supports_temporal_order_contrast": true,
  "supports_order_sensitive_recurrence": "true",
  "supports_bag_of_tokens_failure": true,
  "supports_role_authority_resolution": true,
  "supports_suffix_resolution": true,
  "position_static_shortcut_present": true,
  "geometry_read": "streaming order/role encoding beats bag and no-position baselines on identical-token-set contrasts",
  "reason": "streaming=1.000, bag=0.500, static_pos=1.000, static_no_pos=0.500, zero=0.320, shuffled=0.492, randomized=0.480, random_label=0.09974999999999999, bag_gap=0.500, static_no_pos_gap=0.500, order_drop=0.508, zero_drop=0.680, randomized_drop=0.520, role_score=0.9995969676971436, suffix_score=0.30695574879646303."
}
```

## Claim Boundary

Toy evidence only. Do not claim consciousness, biology, full VRAXION behavior, production validation, or natural-language understanding.

# Query-Cued Pointer Bottleneck Finding

Source run:

- `target/context-cancellation-probe/query-cued-pointer-bottleneck/20260505T181913Z/query_cued_pointer_bottleneck_report.json`

## Question

The previous query-cued probe showed perfect frame prediction and real query dependence, but pointer-specific necessity stayed unclear because direct query-conditioned baselines were close.

This bottleneck probe asks whether the query cue can be compressed into a small internal frame pointer that controls decision authority more efficiently than a direct query path under the same kind of pressure.

## Setup

The same base observation is duplicated under all four toy query cues:

```json
{
  "danger_query": "danger_frame",
  "friendship_query": "friendship_frame",
  "sound_query": "sound_frame",
  "environment_query": "environment_frame"
}
```

Query cues are separate toy vectors, not natural language and not reused frame embeddings.

Same-observation label diversity: `0.875000`

## Accuracy And Controls

| Path | Accuracy |
|---|---:|
| oracle frame pointer | `0.634750` |
| predicted frame pointer | `0.640750` |
| predicted soft frame pointer | `0.640500` |
| hard discrete predicted pointer | `0.640750` |
| full query direct | `0.672750` |
| frame head only | `0.667750` |
| no query baseline | `0.613250` |
| wrong forced frame | `0.516250` |
| query ablation | `0.556250` |
| query shuffle | `0.551500` |
| zero recurrent | `0.530750` |
| randomized recurrent | `0.504750` |
| random label | `0.5055` |

Frame prediction accuracy: `1.000000`

Predicted-frame confusion matrix:

```json
{
  "danger_frame": {
    "danger_frame": 200.0,
    "friendship_frame": 0.0,
    "sound_frame": 0.0,
    "environment_frame": 0.0
  },
  "friendship_frame": {
    "danger_frame": 0.0,
    "friendship_frame": 200.0,
    "sound_frame": 0.0,
    "environment_frame": 0.0
  },
  "sound_frame": {
    "danger_frame": 0.0,
    "friendship_frame": 0.0,
    "sound_frame": 200.0,
    "environment_frame": 0.0
  },
  "environment_frame": {
    "danger_frame": 0.0,
    "friendship_frame": 0.0,
    "sound_frame": 0.0,
    "environment_frame": 200.0
  }
}
```

## Bottleneck Sweep

| Bottleneck | Accuracy | Authority Switch | Refraction Final | Active Influence | Inactive Influence |
|---:|---:|---:|---:|---:|---:|
| `2` | `0.7805` | `0.0935` | `0.07125` | `0.41975` | `0.3485` |
| `4` | `0.748` | `0.077` | `0.037` | `0.40925` | `0.37225` |
| `8` | `0.75375` | `0.08525` | `0.072` | `0.3785` | `0.3065` |
| `16` | `0.7695` | `0.1415` | `0.07525` | `0.4205` | `0.34525` |

## Authority And Refraction

| Path | Accuracy | Authority Switch | Refraction Final | Active Influence | Inactive Influence |
|---|---:|---:|---:|---:|---:|
| predicted frame pointer | `0.64075` | `0.02125` | `0.00275` | `0.402` | `0.39925` |
| full query direct | `0.67275` | `0.04975` | `0.0255` | `0.39675` | `0.37125` |

## Token-Frame Inventory

Selected token output-change rates by query-implied frame:

```json
{
  "dog": {
    "danger_frame": 0.294118,
    "friendship_frame": 0.282353,
    "sound_frame": 0.282353,
    "environment_frame": 0.305882
  },
  "cat": {
    "danger_frame": 0.258824,
    "friendship_frame": 0.223529,
    "sound_frame": 0.394118,
    "environment_frame": 0.288235
  },
  "snake": {
    "danger_frame": 0.345455,
    "friendship_frame": 0.345455,
    "sound_frame": 0.412121,
    "environment_frame": 0.381818
  },
  "bite": {
    "danger_frame": 0.363142,
    "friendship_frame": 0.259856,
    "sound_frame": 0.363445,
    "environment_frame": 0.301319
  },
  "owner": {
    "danger_frame": 0.360701,
    "friendship_frame": 0.639163,
    "sound_frame": 0.378023,
    "environment_frame": 0.391066
  },
  "bark": {
    "danger_frame": 0.362672,
    "friendship_frame": 0.197716,
    "sound_frame": 0.328584,
    "environment_frame": 0.370797
  },
  "street": {
    "danger_frame": 0.337112,
    "friendship_frame": 0.26409,
    "sound_frame": 0.348436,
    "environment_frame": 0.4139
  },
  "car_noise": {
    "danger_frame": 0.285066,
    "friendship_frame": 0.256122,
    "sound_frame": 0.318667,
    "environment_frame": 0.387599
  },
  "light": {
    "danger_frame": 0.242984,
    "friendship_frame": 0.211948,
    "sound_frame": 0.250682,
    "environment_frame": 0.241111
  }
}
```

## Verdict

```json
{
  "supports_query_frame_prediction": true,
  "supports_pointer_as_compact_control": false,
  "supports_pointer_specific_necessity": "false",
  "direct_query_shortcut_present": true,
  "strict_positive": false,
  "geometry_read": "direct query conditioning remains sufficient under this bottleneck setting",
  "reason": "frame_acc=1.000, oracle=0.635, predicted=0.641, full_direct=0.673, no_query=0.613, wrong_forced=0.516, query_ablation=0.556, query_shuffle=0.552, randomized=0.505, random_label=0.5055000000000001, small_best_bottleneck=0.781, best_bottleneck=0.781, pointer_authority=0.02125, small_best_authority=0.0935, pointer_refraction=0.002749999999999997, small_best_refraction=0.07125000000000001."
}
```

## Claim Boundary

Toy evidence only. Do not claim consciousness, biology, full VRAXION behavior, production validation, or natural-language understanding.

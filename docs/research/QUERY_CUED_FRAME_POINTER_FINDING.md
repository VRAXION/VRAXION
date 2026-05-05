# Query-Cued Frame Pointer Finding

Source run:

- `target/context-cancellation-probe/query-cued-frame-pointer/20260505T175124Z/query_cued_frame_pointer_report.json`

## Question

This probe tests whether a query-like toy goal cue can select an internal frame pointer:

```text
same observation + query cue -> predicted internal frame -> recurrent decision pass
```

The query cues are toy vectors, not natural language. They are separate from frame embeddings and imply a frame through supervision.

## Prior Result

The previous `inferred_frame_pointer` probe showed:

- frame prediction accuracy: `1.000`
- oracle-frame task accuracy: `0.893`
- predicted-frame task accuracy: `0.881`
- no-frame baseline accuracy: `0.868`
- frame-head-only accuracy: `0.853`
- wrong-forced-frame accuracy: `0.826`

Read:

```text
Frame inference from input salience worked,
but pointer-specific necessity was unclear because shortcut baselines remained strong.
```

This query-cued version removes the salience-only framing by duplicating the same observation under multiple query cues.

## Setup

Query cues:

```json
{
  "danger_query": "danger_frame",
  "friendship_query": "friendship_frame",
  "sound_query": "sound_frame",
  "environment_query": "environment_frame"
}
```

Each base observation is duplicated under all four query cues, so the same observed feature bundle can require different labels.

Same-observation label diversity: `0.875000`

## Accuracy And Controls

| Path | Accuracy |
|---|---:|
| oracle frame | `0.661750` |
| predicted frame pointer | `0.680000` |
| predicted soft pointer | `0.680000` |
| query head only | `0.657500` |
| no-pointer query baseline | `0.669000` |
| no-query baseline | `0.603250` |
| wrong forced frame | `0.658000` |
| query ablation | `0.571250` |
| query shuffle | `0.565250` |
| zero recurrent | `0.533250` |
| randomized recurrent | `0.513000` |
| random label | `0.513` |

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

## Pointer-Specific Authority Test

| Path | Refraction Final | Authority Switch |
|---|---:|---:|
| predicted pointer | `0.0325` | `0.05` |
| no-pointer query baseline | `0.0115` | `0.022` |

Final active-group influence by frame:

| Frame | Predicted Pointer | No-Pointer Query |
|---|---:|---:|
| `danger_frame` | `0.405` | `0.401` |
| `friendship_frame` | `0.445` | `0.436` |
| `sound_frame` | `0.381` | `0.364` |
| `environment_frame` | `0.391` | `0.379` |

## Token-Frame Inventory

Selected token output-change rates by query-implied frame:

```json
{
  "dog": {
    "danger_frame": 0.3,
    "friendship_frame": 0.229412,
    "sound_frame": 0.264706,
    "environment_frame": 0.282353
  },
  "cat": {
    "danger_frame": 0.258824,
    "friendship_frame": 0.205882,
    "sound_frame": 0.311765,
    "environment_frame": 0.282353
  },
  "snake": {
    "danger_frame": 0.278788,
    "friendship_frame": 0.387879,
    "sound_frame": 0.309091,
    "environment_frame": 0.327273
  },
  "bite": {
    "danger_frame": 0.531779,
    "friendship_frame": 0.208505,
    "sound_frame": 0.267461,
    "environment_frame": 0.268141
  },
  "owner": {
    "danger_frame": 0.29948,
    "friendship_frame": 0.609686,
    "sound_frame": 0.373149,
    "environment_frame": 0.325942
  },
  "bark": {
    "danger_frame": 0.324039,
    "friendship_frame": 0.198337,
    "sound_frame": 0.361373,
    "environment_frame": 0.305935
  },
  "street": {
    "danger_frame": 0.332217,
    "friendship_frame": 0.257269,
    "sound_frame": 0.325175,
    "environment_frame": 0.416918
  },
  "car_noise": {
    "danger_frame": 0.284991,
    "friendship_frame": 0.255403,
    "sound_frame": 0.406624,
    "environment_frame": 0.403921
  },
  "light": {
    "danger_frame": 0.188255,
    "friendship_frame": 0.214173,
    "sound_frame": 0.302686,
    "environment_frame": 0.248069
  }
}
```

## Interpretation

```json
{
  "supports_query_cued_frame_pointer": false,
  "supports_query_frame_prediction": true,
  "supports_pointer_specific_authority": "false",
  "geometry_read": "pointer-specific necessity is not supported in this toy; query conditioning alone is sufficient",
  "reason": "frame_acc=1.000, oracle=0.662, predicted=0.680, query_head_only=0.657, no_pointer_query=0.669, no_query=0.603, wrong_forced=0.658, query_ablation=0.571, query_shuffle=0.565, randomized=0.513, random_label=0.513, pointer_authority=0.05000000000000001, direct_authority=0.02200000000000001, pointer_refraction=0.0325, direct_refraction=0.011500000000000014."
}
```

## Claim Boundary

Toy evidence only. Do not claim consciousness, biology, full VRAXION behavior, production validation, or natural-language understanding.

Safe read:

```text
Query cue selection works, the query matters, recurrence matters, and the pointer path has slightly cleaner authority/refraction geometry. But the no-pointer query baseline is close on accuracy and geometry, so pointer-specific necessity is not supported in this toy.
```

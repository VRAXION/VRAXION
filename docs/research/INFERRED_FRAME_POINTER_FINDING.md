# Inferred Frame Pointer Finding

Source run:

- `target/context-cancellation-probe/inferred-frame-pointer-fixed/20260505T173142Z/inferred_frame_pointer_report.json`

## Question

This probe moves from explicit frame-token control to inferred frame selection:

```text
input bundle -> predicted frame pointer -> recurrent decision pass
```

The task reuses the existing multi-aspect setup:

- `danger_frame`
- `friendship_frame`
- `sound_frame`
- `environment_frame`

No new semantic concepts are added. The compromise is that the intended frame is made inferable by feature-group salience inside the input bundle rather than by an explicit frame token.

## Main Results

- frame prediction accuracy: `1.000000`
- oracle-frame task accuracy: `0.893250`
- predicted-frame task accuracy: `0.881000`
- predicted-frame soft-pointer task accuracy: `0.881750`
- frame-head-only task accuracy: `0.853000`
- no-frame baseline task accuracy: `0.867500`
- wrong-forced-frame task accuracy: `0.826000`
- zero-recurrent task accuracy: `0.587250`
- randomized-recurrent task accuracy: `0.524500`
- random-label control accuracy: `0.502500`
- authority-switch score, predicted frame: `0.103250`
- refraction index final, predicted frame: `0.095750`

## Accuracy By Frame

Predicted-frame pointer:

```json
{
  "danger_frame": 0.87,
  "friendship_frame": 0.904,
  "sound_frame": 0.848,
  "environment_frame": 0.902
}
```

Frame prediction by frame:

```json
{
  "danger_frame": 1.0,
  "friendship_frame": 1.0,
  "sound_frame": 1.0,
  "environment_frame": 1.0
}
```

## Token-Frame Inventory

Selected token output-change rates by inferred/target frame:

```json
{
  "dog": {
    "danger_frame": 0.341176,
    "friendship_frame": 0.223529,
    "sound_frame": 0.582353,
    "environment_frame": 0.117647
  },
  "cat": {
    "danger_frame": 0.335294,
    "friendship_frame": 0.194118,
    "sound_frame": 0.452941,
    "environment_frame": 0.129412
  },
  "snake": {
    "danger_frame": 0.363636,
    "friendship_frame": 0.496970,
    "sound_frame": 0.509091,
    "environment_frame": 0.115152
  },
  "bite": {
    "danger_frame": 0.687980,
    "friendship_frame": 0.113389,
    "sound_frame": 0.160478,
    "environment_frame": 0.140664
  },
  "owner": {
    "danger_frame": 0.185258,
    "friendship_frame": 0.735556,
    "sound_frame": 0.166018,
    "environment_frame": 0.176729
  },
  "bark": {
    "danger_frame": 0.159805,
    "friendship_frame": 0.107802,
    "sound_frame": 0.507475,
    "environment_frame": 0.149544
  },
  "street": {
    "danger_frame": 0.182840,
    "friendship_frame": 0.142598,
    "sound_frame": 0.209658,
    "environment_frame": 0.502631
  },
  "car_noise": {
    "danger_frame": 0.120508,
    "friendship_frame": 0.096055,
    "sound_frame": 0.169505,
    "environment_frame": 0.504209
  },
  "light": {
    "danger_frame": 0.104871,
    "friendship_frame": 0.079867,
    "sound_frame": 0.072775,
    "environment_frame": 0.086241
  }
}
```

Read:

- `bite` has highest authority in `danger_frame`.
- `owner` has highest authority in `friendship_frame`.
- `bark` has highest authority in `sound_frame`.
- `street` and `car_noise` have highest authority in `environment_frame`.
- `light` stays low across frames, as expected because this multi-aspect frame set has no visibility frame.
- `dog` has high authority in actor-causal frames and low authority in `environment_frame`.

## Interpretation

```json
{
  "supports_inferred_frame_pointer": false,
  "supports_frame_prediction": true,
  "supports_pointer_used_for_authority": false,
  "reason": "frame_acc=1.000, oracle=0.893, predicted=0.881, frame_head_only=0.853, no_frame=0.868, wrong_forced=0.826, refraction=0.09575, authority=0.10325."
}
```

The useful result is partial:

```text
frame inference works cleanly,
predicted-frame decision is close to oracle-frame decision,
recurrence is load-bearing,
token-frame authority inventory has the expected shape,
but pointer-specific necessity is not isolated because no-frame and frame-head-only baselines remain fairly strong.
```

This means the current toy supports inferred frame selection, but not yet a clean claim that the predicted frame pointer is the only or dominant path for authority switching.

## Claim Boundary

Toy evidence only. Do not claim consciousness, biology, full VRAXION behavior, production validation, or that the main VRAXION architecture already performs inferred frame selection.

Safe claim:

> In a controlled toy setting, a model can infer the intended task frame from feature-group salience and solve the multi-aspect task with a predicted internal frame pointer close to an oracle-frame upper bound. However, because no-frame and frame-head-only baselines remain strong, pointer-specific necessity is still unclear.

## Next Clean Test

Make the no-frame shortcut harder without adding new semantic concepts:

```text
same feature groups
less obvious salience cue
stronger bottleneck
compare predicted pointer vs frame-head-only and no-frame
```

Positive next signal:

```text
predicted pointer remains high
frame_head_only drops
no_frame drops
wrong_forced_frame drops strongly
```

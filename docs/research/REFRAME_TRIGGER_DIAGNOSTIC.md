# Reframe Trigger Diagnostic

Source run:

- `target/context-cancellation-probe/reframe-h128-u03-v4/20260504T210227Z/reframe_trigger_diagnostic_report.json`

## Question

This diagnostic tests the updated hypothesis:

> The toy system may commit to a frame early, then run recurrent simulation inside that frame. If the initial frame is wrong, recovery may require an explicit reframe/reset signal.

The setup reuses the existing multi-aspect token task:

- `danger_frame`
- `friendship_frame`
- `sound_frame`
- `environment_frame`
- same actor tokens, especially `dog`
- entangled input

No new semantic concepts are added. The reset pulse is a control signal, not a new world feature.

## Main Metrics

- no-reframe fixed accuracy: `0.921667`
- no-reframe wrong-initial/no-reset accuracy: `0.560833`
- no-reframe wrong-initial/with-reset accuracy: `0.668833`
- trained-reframe fixed accuracy: `0.855417`
- trained-reframe wrong-initial/no-reset accuracy: `0.702667`
- trained-reframe wrong-initial/with-reset accuracy: `0.765333`
- reset accuracy gain: `0.062667`
- reset success gain: `0.300000`
- authority transfer gain: `0.130250`

## Recovery By Switch Step

No reset:

```json
{
  "switch_after_1": 0.834667,
  "switch_after_2": 0.761667,
  "switch_after_3": 0.641667,
  "switch_after_4": 0.572667
}
```

With reset:

```json
{
  "switch_after_1": 0.853667,
  "switch_after_2": 0.827,
  "switch_after_3": 0.756667,
  "switch_after_4": 0.624
}
```

## Authority Transfer

No reset:

- old-frame authority decay: `-0.047333`
- new-frame authority rise: `0.100667`
- authority switch after frame switch: `-0.005833`
- target-frame convergence after switch: `-0.016502`

With reset:

- old-frame authority decay: `0.044667`
- new-frame authority rise: `0.134417`
- authority switch after frame switch: `0.124417`
- target-frame convergence after switch: `-0.171119`
- reset entropy spike: `0.001442`

## Controls

- zero-recurrent accuracy: `0.535417`
- threshold-only accuracy: `0.511250`
- shuffled-frame-token accuracy: `0.615833`
- randomized-recurrent accuracy: `0.526250`
- random-label accuracy: `0.508333`

## Verdict

```json
{
  "supports_early_frame_commitment": "true",
  "supports_online_reframe_with_reset": "true",
  "supports_free_midrun_rotation_without_reset": "false",
  "reason": "baseline_fixed_accuracy=0.922, baseline_wrong_no_reset_accuracy=0.561, trained_fixed_accuracy=0.855, no_reset_reframe_accuracy=0.703, with_reset_reframe_accuracy=0.765, reset_accuracy_gain=0.063, reset_success_gain=0.300, reset_authority_transfer_gain=0.130, randomized_recurrent_accuracy=0.52625, shuffled_frame_accuracy=0.6158333333333333, random_label_accuracy=0.5083333333333333."
}
```

## Claim Boundary

Toy diagnostic only. Do not claim consciousness, biology, full VRAXION behavior, or production architecture validation.

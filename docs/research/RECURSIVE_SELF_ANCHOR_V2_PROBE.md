# Recursive Self-Anchor v2 Probe

## Goal

Hard-counterfactual test for whether the committed self-state is required by the next action decision.

The visible second-step prompt is always `choose_next_action`; event, mode, patient, and next-state labels are not visible at step 2.

## Results

| Arm | Semantic | Mode | Authority | State | Step2 | Hard CF | Leakage |
|---|---:|---:|---:|---:|---:|---:|---:|
| `recursive_state_model` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.000819` |
| `no_recursive_update_baseline` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.555556` | `0.555556` | `0.000824` |
| `static_baseline_without_next_state` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.555556` | `0.555556` | `0.000846` |
| `oracle_next_state_baseline` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.000822` |
| `no_grounding_mode` | `1.000000` | `0.555556` | `0.925926` | `0.888889` | `0.888889` | `0.888889` | `0.200041` |
| `no_self_anchor` | `1.000000` | `1.000000` | `0.888889` | `0.888889` | `0.888889` | `0.888889` | `0.000723` |

## Gaps And Controls

- recursive_gap_vs_no_update: `0.444444`
- recursive_gap_vs_static: `0.444444`

| Control | Mean | Std |
|---|---:|---:|
| `shuffled_committed_state_drop` | `0.666667` | `0.000000` |
| `wrong_mode_drop` | `0.678583` | `0.142171` |
| `wrong_mode_semantic_accuracy` | `0.900000` | `0.064788` |
| `wrong_self_anchor_drop` | `0.568180` | `0.060771` |

## Verdict

```json
{
  "supports_recursive_self_anchor_v2": true,
  "supports_committed_self_state": true,
  "recursive_beats_no_update": true,
  "recursive_beats_static": true,
  "baselines_fail_hard_counterfactual": true,
  "shuffled_committed_state_control_hurts": true,
  "wrong_mode_control_hurts": true,
  "wrong_self_anchor_control_hurts": true
}
```

## Interpretation

This fixes the v1 failure mode. The second-step prompt is identical across hard
counterfactual cases, and the carried committed state is a hard one-hot state,
not a soft side-channel. The recursive model and oracle baseline both solve the
hard counterfactuals, while no-update/static baselines stay near majority-choice
performance.

Safe readout:

- committed self-state is learned cleanly
- hard second-step decisions require the committed state in this toy
- shuffling the committed state breaks the behavior
- wrong grounding and wrong self-anchor controls hurt

## Claim Boundary

Toy evidence only. No consciousness, biology, quantum behavior, production validity, or natural-language-understanding claim.

# Recursive Self-Anchor Authority Probe

## Goal

Test whether grounded authority can update self-state, and whether the updated self-state affects the next decision.

This run uses inferred compositional grounding cues because the inferred grounding phase passed.

## Results

| Arm | Overall | Semantic | Mode | Authority | Next Self | Step2 | Step2 Update Cases | Self Gain | Leakage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `recursive_state_model` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.980416` | `0.005207` |
| `no_recursive_update_baseline` | `0.970000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.850000` | `1.000000` | `0.979473` | `0.004822` |
| `static_baseline_without_next_state` | `0.970000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.850000` | `1.000000` | `0.980610` | `0.005619` |
| `no_self_anchor` | `0.954123` | `1.000000` | `1.000000` | `0.953951` | `0.910741` | `0.905926` | `0.881481` | `0.000000` | `0.002701` |
| `no_grounding_mode` | `0.753395` | `1.000000` | `0.200000` | `0.822531` | `0.944444` | `0.800000` | `1.000000` | `0.427723` | `0.400904` |
| `oracle_next_state_baseline` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.978518` | `0.005876` |

## Controls

| Metric | Mean | Std |
|---|---:|---:|
| `wrong_mode_drop` | `0.985921` | `0.002047` |
| `wrong_mode_semantic_accuracy` | `1.000000` | `0.000000` |
| `wrong_self_state_drop` | `0.000000` | `0.000000` |

## Verdict

```json
{
  "supports_recursive_self_anchor_authority": false,
  "supports_next_self_state_update": true,
  "supports_second_step_recursive_use": false,
  "supports_self_anchor_gain": true,
  "supports_wrong_mode_control": true,
  "wrong_self_state_control_hurts": false
}
```

## Interpretation

This is a partial/negative recursive result. The model can infer grounding, action
authority, and next self-state cleanly. However, the `no_recursive_update_baseline`
and `static_baseline_without_next_state` also reach `1.000000` on update-case
second-step accuracy. That means this task did not isolate recursive use of the
updated self-state strongly enough. The likely failure mode is that the second-step
target is still predictable from coarse grounding/self priors rather than requiring
the committed updated state.

Safe readout:

- next self-state update works in this toy
- self-anchor and wrong-grounding controls work
- recursive second-step authority is not proven
- the next recursive probe needs harder counterfactual second-step pairs where the
  same second-step input requires different action depending only on the committed
  updated self-state

## Claim Boundary

Toy evidence only. No consciousness, biology, quantum, production, or natural-language-understanding claim.

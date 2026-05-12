# Integrated Grounded Modular Self Controller Probe

## Goal

Test whether inferred grounding plus a hard committed self-state can drive a learned controller over frozen modules while preserving primitive skills.

This integrates the grounding, recursive self-anchor, and modular skill-controller toy mechanisms. The committed self-state is hard one-hot, not a soft hidden side channel.

## Event / Self-State Results

| Arm | Semantic | Mode | State | Action | Hard CF | Margin | Leakage |
|---|---:|---:|---:|---:|---:|---:|---:|
| `integrated_recursive_controller` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.997113` | `0.000964` |
| `no_committed_state_baseline` | `1.000000` | `1.000000` | `1.000000` | `0.444444` | `0.444444` | `0.996843` | `0.000975` |
| `static_without_commit_baseline` | `1.000000` | `1.000000` | `1.000000` | `0.444444` | `0.444444` | `0.996894` | `0.001064` |
| `oracle_committed_state_baseline` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.996931` | `0.000970` |
| `shuffled_committed_state_control` | `1.000000` | `1.000000` | `1.000000` | `0.111111` | `0.111111` | `0.997113` | `0.000964` |
| `no_grounding_control` | `1.000000` | `0.555556` | `0.777778` | `0.777778` | `0.777778` | `0.398891` | `0.200037` |

## Controller / Skill Results

| Arm | Primitive Before | Primitive After | Drift | Composition | Program Acc |
|---|---:|---:|---:|---:|---:|
| `integrated_recursive_controller` | `1.000000` | `1.000000` | `0.000000` | `1.000000` | `1.000000` |
| `shared_end_to_end_no_freeze` | `1.000000` | `0.416618` | `0.583382` | `0.862099` | `null` |
| `frozen_learned_controller_reference` | `1.000000` | `1.000000` | `0.000000` | `1.000000` | `1.000000` |

## Gaps And Controls

- recursive_gap_vs_no_commit: `0.555556`
- recursive_gap_vs_static: `0.555556`
- oracle_gap: `0.000000`

| Control | Mean | Std |
|---|---:|---:|
| `shuffled_committed_state_drop` | `0.888889` | `0.000000` |
| `wrong_mode_drop` | `0.993620` | `0.000973` |
| `wrong_mode_semantic_accuracy` | `1.000000` | `0.000000` |

## Verdict

```json
{
  "supports_integrated_grounded_controller": true,
  "committed_state_controls_controller": true,
  "grounding_controls_action_authority": true,
  "frozen_primitives_preserved": true,
  "shared_end_to_end_drifts": true,
  "oracle_upper_bound_matched": true,
  "no_commit_baselines_fail": true
}
```

## Interpretation

Positive readout requires the integrated recursive controller to solve hard counterfactual action choices, beat no-commit/static baselines, preserve frozen primitive modules, and show the shared end-to-end drift failure mode.

Safe claim if positive: in a controlled toy setting, inferred grounding can update a hard committed self-state that later drives a modular controller over frozen skills/actions without primitive drift.

## Claim Boundary

Toy evidence only. No consciousness, biology, quantum behavior, natural-language-understanding, full VRAXION, production, or deployment claim.

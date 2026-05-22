# Grounding Mode Authority Probe

## Goal

Test whether the same semantic event remains stable while action authority changes by grounding mode.

Modes: `reality`, `tv`, `game`, `dream`, `memory`.

## Results

| Metric | Mean | Std |
|---|---:|---:|
| `overall_accuracy` | `1.000000` | `0.000000` |
| `semantic_accuracy` | `1.000000` | `0.000000` |
| `authority_accuracy` | `1.000000` | `0.000000` |
| `reality_action_authority` | `0.988484` | `0.002812` |
| `nonreality_action_leakage` | `0.003718` | `0.000413` |
| `grounding_authority_margin` | `0.984766` | `0.003042` |
| `self_anchor_gain` | `0.985858` | `0.002973` |
| `semantic_consistency_range` | `0.002078` | `0.000886` |
| `wrong_forced_tv_real_action_drop` | `0.984344` | `0.003243` |
| `mode_shuffle_accuracy_against_true_labels` | `0.812063` | `0.007501` |
| `mode_ablation_accuracy` | `0.868571` | `0.002540` |

## Verdict

```json
{
  "supports_semantic_grounding_split": true,
  "supports_grounding_mode_authority": true,
  "supports_self_anchor_authority": true,
  "supports_wrong_mode_control": true,
  "mode_leakage_low": true
}
```

## Interpretation

This is a quick toy diagnostic. A positive result supports separating semantic event understanding from grounded action authority. It does not prove consciousness, biology, production validity, or natural-language understanding.

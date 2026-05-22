# Inferred Grounding Mode Authority Probe

## Goal

Test whether grounding mode can be inferred from compositional cue-feature bundles while semantic event recognition remains stable and action authority changes by grounding/self relevance.

The main split holds out cue-feature combinations, not all cue atoms. Strict unseen cue tokens are diagnostic only.

## Final-Test Results

### compositional_cue_split

| Arm | Overall | Semantic | Mode | Authority | Reality Action | Nonreality Leakage | Margin | Self Gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `explicit_mode_upper_bound` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.965307` | `0.005881` | `0.959426` | `0.956939` |
| `inferred_context_mode` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.968082` | `0.004692` | `0.963390` | `0.957524` |
| `no_context` | `0.677778` | `1.000000` | `0.200000` | `0.833333` | `0.198915` | `0.198915` | `0.000000` | `0.196968` |
| `semantic_only_baseline` | `0.570123` | `1.000000` | `0.195556` | `0.514815` | `0.545751` | `0.543588` | `0.002163` | `-0.000505` |

### seen_cue_heldout_combinations

| Arm | Overall | Semantic | Mode | Authority | Reality Action | Nonreality Leakage | Margin | Self Gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `explicit_mode_upper_bound` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.959452` | `0.004832` | `0.954620` | `0.956997` |
| `inferred_context_mode` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.958849` | `0.003112` | `0.955737` | `0.951564` |
| `no_context` | `0.634100` | `1.000000` | `0.077586` | `0.824713` | `0.153444` | `0.172701` | `-0.019257` | `0.152540` |
| `semantic_only_baseline` | `0.574138` | `1.000000` | `0.205172` | `0.517241` | `0.549992` | `0.551634` | `-0.001642` | `-0.001835` |

## Controls

| Metric | Mean | Std |
|---|---:|---:|
| `mode_ablation_drop` | `0.323210` | `0.001975` |
| `mode_shuffle_drop` | `0.353416` | `0.009418` |
| `wrong_forced_mode_drop` | `0.963430` | `0.003161` |
| `wrong_forced_reality_rise` | `0.963074` | `0.002747` |
| `wrong_forced_reality_semantic_accuracy` | `1.000000` | `0.000000` |
| `wrong_forced_tv_semantic_accuracy` | `1.000000` | `0.000000` |

## Strict Unseen Cue Token Diagnostic

This is not the main result. It is expected to fail or weaken because strict unseen cue atoms have no pretrained/compositional semantics.

| Metric | Mean | Std |
|---|---:|---:|
| `action_authority_accuracy` | `0.832593` | `0.004914` |
| `grounding_authority_margin` | `-0.011400` | `0.036042` |
| `grounding_mode_accuracy` | `0.171852` | `0.046236` |
| `nonreality_action_leakage` | `0.070975` | `0.015368` |
| `overall_accuracy` | `0.668148` | `0.015108` |
| `reality_action_authority` | `0.059575` | `0.029159` |
| `self_anchor_gain` | `0.097106` | `0.059008` |
| `semantic_accuracy` | `1.000000` | `0.000000` |
| `semantic_consistency_range` | `0.000128` | `0.000117` |

## Verdict

```json
{
  "supports_inferred_grounding_mode_authority": true,
  "supports_semantic_grounding_split": true,
  "supports_grounding_mode_prediction": true,
  "supports_self_anchor_authority": true,
  "supports_wrong_forced_mode_control": true,
  "no_context_control_hurts": true,
  "shuffled_context_control_hurts": true,
  "mode_leakage_low": true
}
```

## Claim Boundary

Toy evidence only. No consciousness, biology, quantum, production, or natural-language-understanding claim.

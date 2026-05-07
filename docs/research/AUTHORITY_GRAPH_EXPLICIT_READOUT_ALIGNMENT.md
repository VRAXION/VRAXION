# Authority Graph Explicit Readout Alignment

## Goal

Test whether explicit readout training can align output-causal authority with the existing hand-seeded route-state mechanism.

Route-state authority is diagnostic only. Verdicts prioritize explicit output influence and collapse guards.

## Run Configuration

```json
{
  "seeds": 1,
  "train_samples": 64,
  "validation_samples": 64,
  "final_test_samples": 128,
  "steps": 3,
  "epochs": 4,
  "learning_rate": 0.01,
  "checkpoint_every": 2,
  "max_runtime_hours": 0.25,
  "torch_threads": 2,
  "arms_requested": [
    "route_state_reference",
    "explicit_untrained",
    "CE_only",
    "combined_authority_readout_loss",
    "matched_random_combined_loss"
  ],
  "arms_completed": [
    "CE_only",
    "combined_authority_readout_loss",
    "explicit_untrained",
    "matched_random_combined_loss",
    "route_state_reference"
  ],
  "loss_type": "binary_cross_entropy_with_logits",
  "loss_reason": "The current authority graph exposes one binary positive-vs-negative readout logit, not a multiclass output vector.",
  "final_test_used_for_selection": false,
  "completed": true,
  "smoke": true
}
```

## Final-Test Results

| Arm | Success | Collapse | Accuracy | Temporal | Output Authority | Route-State Authority | Active | Inactive | Margin | Wrong Frame | Correct Conf | Wrong Conf | Mean Abs Logit | Near-Zero |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `CE_only` | `0.000000` | `0.000000` | `0.593750` | `0.500000` | `0.000396` | `0.608020` | `0.000516` | `0.000120` | `0.000396` | `0.000000` | `0.613391` | `0.613019` | `1.994646` | `0.000000` |
| `combined_authority_readout_loss` | `0.000000` | `0.000000` | `0.593750` | `0.500000` | `0.000396` | `0.608020` | `0.000516` | `0.000120` | `0.000396` | `0.000000` | `0.613391` | `0.613019` | `1.994646` | `0.000000` |
| `explicit_untrained` | `0.000000` | `0.000000` | `0.593750` | `0.500000` | `0.000298` | `0.608020` | `0.000389` | `0.000091` | `0.000298` | `0.000000` | `0.613310` | `0.613032` | `1.996073` | `0.000000` |
| `matched_random_combined_loss` | `0.000000` | `0.000000` | `0.406250` | `0.500000` | `-0.011341` | `-0.019756` | `0.039502` | `0.050843` | `-0.011341` | `0.000000` | `0.455699` | `0.451261` | `0.899267` | `0.018939` |
| `route_state_reference` | `0.000000` | `0.000000` | `0.950521` | `0.875000` | `null` | `0.383102` | `null` | `null` | `null` | `0.375000` | `null` | `null` | `null` | `null` |

## Train / Validation / Final-Test Gap

| Arm | Train Acc | Val Acc | Final Acc | Train Auth | Val Auth | Final Auth |
|---|---:|---:|---:|---:|---:|---:|
| `CE_only` | `0.593750` | `0.598958` | `0.593750` | `0.000409` | `0.000399` | `0.000396` |
| `combined_authority_readout_loss` | `0.593750` | `0.598958` | `0.593750` | `0.000409` | `0.000399` | `0.000396` |
| `explicit_untrained` | `0.593750` | `0.598958` | `0.593750` | `0.000308` | `0.000299` | `0.000298` |
| `matched_random_combined_loss` | `0.406250` | `0.401042` | `0.406250` | `-0.003699` | `-0.004025` | `-0.011341` |
| `route_state_reference` | `0.958333` | `0.942708` | `0.950521` | `null` | `null` | `null` |

## Verdict

```json
{
  "authority_alignment_improves_output_authority": false,
  "accuracy_preserved_under_alignment": false,
  "inactive_leakage_reduced": false,
  "wrong_frame_sensitivity_preserved": false,
  "combined_loss_beats_ce_only": false,
  "collapse_detected": false,
  "random_control_still_weak": true,
  "route_state_shortcut_still_diagnostic_only": true,
  "final_verdict_prioritizes_output_influence": true
}
```

## Interpretation Notes

- BCE is used because the graph exposes one binary positive-vs-negative readout logit.
- CE-only selects by validation accuracy/loss; authority-aligned arms select by validation combined authority objective.
- Each aligned arm also records the best-validation-authority checkpoint in JSON metadata.
- Collapse is flagged when authority appears to improve through low-confidence or near-zero logits.
- Matched random control uses the same readout capacity and training budget; only internal topology differs.

## Runtime Notes

- runtime seconds: `70.631232`
- interrupted by wall clock: `False`
- completed records: `5`

## Claim Boundary

Toy evidence only. Do not claim consciousness, biology, full VRAXION behavior, or production validation.

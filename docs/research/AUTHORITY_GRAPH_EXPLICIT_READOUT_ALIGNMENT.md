# Authority Graph Explicit Readout Alignment

## Goal

Test whether explicit readout training can align output-causal authority with the existing hand-seeded route-state mechanism.

Route-state authority is diagnostic only. Verdicts prioritize explicit output influence and collapse guards.

## Run Configuration

```json
{
  "seeds": 3,
  "train_samples": 256,
  "validation_samples": 256,
  "final_test_samples": 512,
  "steps": 5,
  "epochs": 200,
  "learning_rate": 0.005,
  "checkpoint_every": 25,
  "max_runtime_hours": 2.0,
  "torch_threads": 2,
  "arms_requested": [
    "route_state_reference",
    "explicit_untrained",
    "CE_only",
    "CE_plus_output_authority",
    "CE_plus_inactive_leakage_penalty",
    "CE_plus_wrong_frame_margin",
    "combined_authority_readout_loss",
    "matched_random_combined_loss"
  ],
  "arms_completed": [
    "CE_only",
    "CE_plus_inactive_leakage_penalty",
    "CE_plus_output_authority",
    "CE_plus_wrong_frame_margin",
    "combined_authority_readout_loss",
    "explicit_untrained",
    "matched_random_combined_loss",
    "route_state_reference"
  ],
  "loss_type": "binary_cross_entropy_with_logits",
  "loss_reason": "The current authority graph exposes one binary positive-vs-negative readout logit, not a multiclass output vector.",
  "final_test_used_for_selection": false,
  "completed": false,
  "smoke": false
}
```

## Final-Test Results

Important: this run hit the wall-clock limit. Seed 0 completed every arm. Seed 1 completed
`CE_only`, but most authority-aligned arms were interrupted early or evaluated before
training. The aggregate table below is useful for reproducibility, but the fairest loss
comparison is the completed seed-0 table in the next section.

| Arm | Success | Collapse | Accuracy | Temporal | Output Authority | Route-State Authority | Active | Inactive | Margin | Wrong Frame | Correct Conf | Wrong Conf | Mean Abs Logit | Near-Zero |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `CE_only` | `0.000000` | `0.000000` | `0.820312` | `0.500000` | `0.128543` | `0.654883` | `0.152199` | `0.023656` | `0.128543` | `0.337891` | `0.682892` | `0.571175` | `0.792476` | `0.068314` |
| `CE_plus_inactive_leakage_penalty` | `0.000000` | `0.000000` | `0.707031` | `0.500000` | `0.075492` | `0.654883` | `0.088700` | `0.013208` | `0.075492` | `0.181641` | `0.660532` | `0.595055` | `1.454225` | `0.032461` |
| `CE_plus_output_authority` | `0.000000` | `0.000000` | `0.711263` | `0.500000` | `0.086736` | `0.654883` | `0.102777` | `0.016041` | `0.086736` | `0.179688` | `0.668109` | `0.594274` | `1.298151` | `0.016957` |
| `CE_plus_wrong_frame_margin` | `0.000000` | `0.000000` | `0.710286` | `0.500000` | `0.074539` | `0.654883` | `0.088375` | `0.013835` | `0.074539` | `0.181641` | `0.659381` | `0.594370` | `1.448212` | `0.025194` |
| `combined_authority_readout_loss` | `0.000000` | `0.000000` | `0.705404` | `0.500000` | `0.073880` | `0.654883` | `0.087602` | `0.013722` | `0.073880` | `0.152344` | `0.658315` | `0.593894` | `1.442521` | `0.023740` |
| `explicit_untrained` | `0.000000` | `0.000000` | `0.592448` | `0.500000` | `0.000934` | `0.654883` | `0.001162` | `0.000228` | `0.000934` | `0.000000` | `0.610203` | `0.609458` | `1.989984` | `0.000000` |
| `matched_random_combined_loss` | `0.000000` | `0.000000` | `0.527995` | `0.500000` | `-0.019958` | `-0.049176` | `0.051189` | `0.071148` | `-0.019958` | `0.003906` | `0.504855` | `0.504384` | `1.061896` | `0.054264` |
| `route_state_reference` | `0.000000` | `0.000000` | `0.992188` | `1.000000` | `null` | `0.361312` | `null` | `null` | `null` | `0.359375` | `null` | `null` | `null` | `null` |

## Completed Seed-0 Comparison

| Arm | Accuracy | Temporal | Output Authority | Wrong Frame | Margin | Mean Abs Logit | Near-Zero |
|---|---:|---:|---:|---:|---:|---:|---:|
| `route_state_reference` | `0.991536` | `1.000000` | `null` | `0.359375` | `null` | `null` | `null` |
| `explicit_untrained` | `0.592448` | `0.500000` | `0.000932` | `0.000000` | `0.000932` | `1.990001` | `0.000000` |
| `CE_only` | `0.819010` | `0.500000` | `0.107910` | `0.304688` | `0.107910` | `0.663849` | `0.065891` |
| `CE_plus_output_authority` | `0.830078` | `0.500000` | `0.148166` | `0.359375` | `0.148166` | `0.900696` | `0.033915` |
| `CE_plus_inactive_leakage_penalty` | `0.821615` | `0.500000` | `0.150049` | `0.363281` | `0.150049` | `0.918483` | `0.064922` |
| `CE_plus_wrong_frame_margin` | `0.828125` | `0.500000` | `0.148144` | `0.363281` | `0.148144` | `0.906457` | `0.050388` |
| `combined_authority_readout_loss` | `0.818359` | `0.500000` | `0.146825` | `0.304688` | `0.146825` | `0.895075` | `0.047481` |
| `matched_random_combined_loss` | `0.606771` | `0.500000` | `-0.019040` | `0.007812` | `-0.019040` | `0.691123` | `0.046512` |

Completed-seed readout: authority-aligned losses give a modest output-authority lift
over `CE_only` (`~0.148-0.150` vs `0.108`) without logit collapse, and the matched
random control remains weak. This is not a strict success: temporal explicit-readout
accuracy stays at `0.500`, the combined loss does not beat the best individual losses,
and the output-authority lift does not cross the strict `+0.05` improvement threshold.

## Train / Validation / Final-Test Gap

| Arm | Train Acc | Val Acc | Final Acc | Train Auth | Val Auth | Final Auth |
|---|---:|---:|---:|---:|---:|---:|
| `CE_only` | `0.818359` | `0.820312` | `0.820312` | `0.140272` | `0.140970` | `0.128543` |
| `CE_plus_inactive_leakage_penalty` | `0.712240` | `0.708984` | `0.707031` | `0.083299` | `0.082560` | `0.075492` |
| `CE_plus_output_authority` | `0.717448` | `0.712891` | `0.711263` | `0.094925` | `0.094519` | `0.086736` |
| `CE_plus_wrong_frame_margin` | `0.716797` | `0.710938` | `0.710286` | `0.082055` | `0.081214` | `0.074539` |
| `combined_authority_readout_loss` | `0.711589` | `0.707031` | `0.705404` | `0.081029` | `0.080217` | `0.073880` |
| `explicit_untrained` | `0.600260` | `0.602214` | `0.592448` | `0.001009` | `0.001003` | `0.000934` |
| `matched_random_combined_loss` | `0.520833` | `0.533854` | `0.527995` | `-0.020388` | `-0.021888` | `-0.019958` |
| `route_state_reference` | `0.985677` | `0.981120` | `0.992188` | `null` | `null` | `null` |

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

- runtime seconds: `7462.895315`
- interrupted by wall clock: `True`
- completed records: `16`

## Claim Boundary

Toy evidence only. Do not claim consciousness, biology, full VRAXION behavior, or production validation.

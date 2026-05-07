# Authority Graph Explicit Readout Repair

## Goal

Test whether the hand-seeded authority graph can recover its route-state behavior through explicit route-to-readout edges.

The route-state readout is included only as a diagnostic reference. Verdicts prioritize explicit output influence.

## Run Configuration

```json
{
  "seeds": 1,
  "train_samples": 128,
  "validation_samples": 128,
  "final_test_samples": 256,
  "steps": 5,
  "epochs": 80,
  "learning_rate": 0.01,
  "checkpoint_every": 20,
  "max_runtime_hours": 0.5,
  "torch_threads": 2,
  "arms_requested": [
    "hand_seeded_route_state_reference",
    "hand_seeded_explicit_readout_untrained",
    "hand_seeded_train_readout_only",
    "hand_seeded_train_readout_plus_route_gains",
    "random_graph_train_readout_plus_route_gains"
  ],
  "arms_completed": [
    "hand_seeded_explicit_readout_untrained",
    "hand_seeded_route_state_reference",
    "hand_seeded_train_readout_only",
    "hand_seeded_train_readout_plus_route_gains",
    "random_graph_train_readout_plus_route_gains"
  ],
  "loss_type": "binary_cross_entropy_with_logits",
  "loss_reason": "The current authority graph exposes one binary positive-vs-negative readout logit, not a multiclass output vector.",
  "final_test_used_for_selection": false,
  "completed": true,
  "smoke": false
}
```

## Final-Test Results

| Arm | Success | Accuracy | Latent | Multi | Temporal | Output Authority | Route-State Authority | Wrong Frame | Recurrence | Route Spec | Edges | Readout Params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hand_seeded_explicit_readout_untrained` | `0.000000` | `0.592448` | `0.644531` | `0.632812` | `0.500000` | `0.001033` | `0.675403` | `0.000000` | `0.000000` | `0.001846` | `101.000000` | `12.000000` |
| `hand_seeded_route_state_reference` | `0.000000` | `0.992188` | `0.984375` | `0.992188` | `1.000000` | `null` | `0.376068` | `0.359375` | `0.500000` | `null` | `91.000000` | `0.000000` |
| `hand_seeded_train_readout_only` | `0.000000` | `0.992188` | `0.984375` | `0.992188` | `1.000000` | `0.104850` | `0.675403` | `0.359375` | `0.500000` | `0.045775` | `101.000000` | `12.000000` |
| `hand_seeded_train_readout_plus_route_gains` | `0.000000` | `0.897135` | `0.851562` | `0.839844` | `1.000000` | `0.070536` | `0.475943` | `0.132812` | `0.500000` | `0.008211` | `101.000000` | `12.000000` |
| `random_graph_train_readout_plus_route_gains` | `0.000000` | `0.700521` | `0.742188` | `0.734375` | `0.625000` | `-0.054994` | `-0.069182` | `0.000000` | `-0.125000` | `-0.182054` | `101.000000` | `12.000000` |

## Train / Validation / Final-Test Gap

| Arm | Train Acc | Val Acc | Final Acc | Train Output Auth | Val Output Auth | Final Output Auth |
|---|---:|---:|---:|---:|---:|---:|
| `hand_seeded_explicit_readout_untrained` | `0.591146` | `0.601562` | `0.592448` | `0.000869` | `0.000810` | `0.001033` |
| `hand_seeded_route_state_reference` | `1.000000` | `0.979167` | `0.992188` | `null` | `null` | `null` |
| `hand_seeded_train_readout_only` | `1.000000` | `0.979167` | `0.992188` | `0.086984` | `0.087204` | `0.104850` |
| `hand_seeded_train_readout_plus_route_gains` | `0.898438` | `0.898438` | `0.897135` | `0.049409` | `0.050633` | `0.070536` |
| `random_graph_train_readout_plus_route_gains` | `0.716146` | `0.695312` | `0.700521` | `-0.065594` | `-0.052431` | `-0.054994` |

## Verdict

```json
{
  "route_state_shortcut_confirmed": true,
  "explicit_readout_repair_successful": false,
  "readout_only_sufficient": false,
  "route_gain_training_required": false,
  "hand_topology_still_useful_under_explicit_readout": true,
  "damaged_hand_repair_under_explicit_readout": false,
  "random_graph_control_fails": true,
  "final_verdict_prioritizes_output_influence": true
}
```

## Interpretation Notes

- BCE is used because the current graph exposes one binary positive-vs-negative readout logit.
- Route-state authority is diagnostic only; explicit output-influence authority is the mechanism metric for verdicts.
- If readout-only fails but plus-route-gains works, interpret that as route-state calibration being required for explicit readout.
- Random graph readout capacity is matched to hand-seeded readout capacity; only internal topology differs.

## Runtime Notes

- runtime seconds: `362.460318`
- interrupted by wall clock: `False`
- completed records: `5`

## Claim Boundary

Toy evidence only. Do not claim consciousness, biology, full VRAXION behavior, or production validation.

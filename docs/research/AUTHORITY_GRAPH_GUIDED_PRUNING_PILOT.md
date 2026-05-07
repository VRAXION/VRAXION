# Authority Graph Guided Pruning Pilot

## Goal

Test whether an overcomplete explicit authority-graph scaffold can be trained through scalar gains and then pruned into a sparse authority-flow graph.

All arms use the same explicit readout-node scoring. The old route-state shortcut is not used for comparisons in this report.

## Run Configuration

```json
{
  "seeds": 3,
  "steps": 5,
  "train_samples": 128,
  "validation_samples": 128,
  "final_test_samples": 256,
  "epochs": 80,
  "fine_tune_epochs": 20,
  "mutation_steps": 30,
  "learning_rate": 0.0025,
  "checkpoint_every": 20,
  "max_runtime_hours": 1.5,
  "torch_threads": 2,
  "arms_requested": [
    "hand_seeded",
    "damaged_hand_seeded_50",
    "random_graph_baseline",
    "grammar_v2_untrained",
    "grammar_v2_mutation_only",
    "overcomplete_guided_train",
    "overcomplete_guided_prune_30",
    "overcomplete_guided_prune_50",
    "overcomplete_guided_prune_70",
    "overcomplete_guided_prune_50_repair"
  ],
  "arms_completed": [
    "damaged_hand_seeded_50",
    "grammar_v2_mutation_only",
    "grammar_v2_untrained",
    "hand_seeded",
    "overcomplete_guided_prune_30",
    "overcomplete_guided_prune_50",
    "overcomplete_guided_prune_50_repair",
    "overcomplete_guided_prune_70",
    "overcomplete_guided_train",
    "random_graph_baseline"
  ],
  "explicit_readout_policy": true,
  "completed": true,
  "smoke": false
}
```

## Final-Test Results

| Arm | Success | Strong | Accuracy | Latent | Multi | Temporal | Authority | Active | Inactive | Margin | Wrong Frame | Recurrence | Route Spec | Edges |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `damaged_hand_seeded_50` | `0.000000` | `0.000000` | `0.594184` | `0.644531` | `0.638021` | `0.500000` | `0.000162` | `0.000276` | `0.000114` | `0.000162` | `0.000000` | `0.000000` | `-0.000278` | `55.000000` |
| `grammar_v2_mutation_only` | `0.000000` | `0.000000` | `0.594184` | `0.644531` | `0.638021` | `0.500000` | `0.000455` | `0.001877` | `0.001421` | `0.000455` | `0.000000` | `0.000000` | `-0.010846` | `131.333333` |
| `grammar_v2_untrained` | `0.000000` | `0.000000` | `0.594184` | `0.644531` | `0.638021` | `0.500000` | `0.000264` | `0.001549` | `0.001285` | `0.000264` | `0.000000` | `0.000000` | `-0.026828` | `131.333333` |
| `hand_seeded` | `0.000000` | `0.000000` | `0.594184` | `0.644531` | `0.638021` | `0.500000` | `0.001039` | `0.001279` | `0.000240` | `0.001039` | `0.000000` | `0.000000` | `0.001800` | `101.000000` |
| `overcomplete_guided_prune_30` | `0.000000` | `0.000000` | `0.774306` | `0.800781` | `0.772135` | `0.750000` | `-0.004619` | `0.108727` | `0.113347` | `-0.004619` | `0.106771` | `0.250000` | `-0.165346` | `95.000000` |
| `overcomplete_guided_prune_50` | `0.000000` | `0.000000` | `0.736111` | `0.781250` | `0.760417` | `0.666667` | `-0.005696` | `0.098556` | `0.104252` | `-0.005696` | `0.080729` | `0.166667` | `-0.198533` | `70.333333` |
| `overcomplete_guided_prune_50_repair` | `0.000000` | `0.000000` | `0.765625` | `0.812500` | `0.776042` | `0.708333` | `0.007248` | `0.098015` | `0.090767` | `0.007248` | `0.138021` | `0.208333` | `-0.193655` | `70.333333` |
| `overcomplete_guided_prune_70` | `0.000000` | `0.000000` | `0.670139` | `0.730469` | `0.738281` | `0.541667` | `0.000925` | `0.075332` | `0.074407` | `0.000925` | `0.067708` | `0.041667` | `-0.247754` | `46.333333` |
| `overcomplete_guided_train` | `0.000000` | `0.000000` | `0.756944` | `0.772135` | `0.748698` | `0.750000` | `-0.005842` | `0.098441` | `0.104282` | `-0.005842` | `0.059896` | `0.250000` | `-0.201040` | `131.333333` |
| `random_graph_baseline` | `0.000000` | `0.000000` | `0.514323` | `0.524740` | `0.518229` | `0.500000` | `-0.049819` | `0.060498` | `0.110317` | `-0.049819` | `0.026042` | `0.041667` | `-0.273347` | `130.666667` |

## Train / Validation / Final-Test Gap

| Arm | Train Acc | Val Acc | Final Acc | Train Loss | Val Loss | Final Loss |
|---|---:|---:|---:|---:|---:|---:|
| `damaged_hand_seeded_50` | `0.593750` | `0.602431` | `0.594184` | `2.220537` | `2.210247` | `2.216972` |
| `grammar_v2_mutation_only` | `0.593750` | `0.602431` | `0.594184` | `2.355715` | `2.357748` | `2.358397` |
| `grammar_v2_untrained` | `0.593750` | `0.602431` | `0.594184` | `2.343481` | `2.345390` | `2.348613` |
| `hand_seeded` | `0.593750` | `0.602431` | `0.594184` | `2.115613` | `2.111880` | `2.113912` |
| `overcomplete_guided_prune_30` | `0.777778` | `0.783854` | `0.774306` | `1.520018` | `1.535083` | `1.525366` |
| `overcomplete_guided_prune_50` | `0.730903` | `0.742188` | `0.736111` | `1.609936` | `1.620025` | `1.612363` |
| `overcomplete_guided_prune_50_repair` | `0.763021` | `0.773438` | `0.765625` | `1.632636` | `1.643078` | `1.634416` |
| `overcomplete_guided_prune_70` | `0.675347` | `0.675347` | `0.670139` | `1.706161` | `1.715833` | `1.711356` |
| `overcomplete_guided_train` | `0.756076` | `0.764757` | `0.756944` | `1.587462` | `1.601151` | `1.592946` |
| `random_graph_baseline` | `0.514757` | `0.521701` | `0.514323` | `1.839744` | `1.857341` | `1.846932` |

## Pruning Audit

| Arm | Pruned edge type counts | <= hand edges | <= damaged edges | Still bloated |
|---|---|---:|---:|---:|
| `damaged_hand_seeded_50` | `{}` | `True` | `True` | `False` |
| `grammar_v2_mutation_only` | `{}` | `False` | `False` | `True` |
| `grammar_v2_untrained` | `{}` | `False` | `False` | `True` |
| `hand_seeded` | `{}` | `True` | `False` | `False` |
| `overcomplete_guided_prune_30` | `{"hub->route": 4, "route recurrent": 35, "suppressor->route": 4, "temporal role": 15, "token->hub": 17, "token->route": 34}` | `True` | `False` | `False` |
| `overcomplete_guided_prune_50` | `{"hub->route": 12, "route recurrent": 39, "suppressor->route": 6, "temporal role": 23, "token->hub": 38, "token->route": 65}` | `True` | `False` | `False` |
| `overcomplete_guided_prune_50_repair` | `{"hub->route": 12, "route recurrent": 39, "suppressor->route": 6, "temporal role": 23, "token->hub": 38, "token->route": 65}` | `True` | `False` | `False` |
| `overcomplete_guided_prune_70` | `{"hub->route": 16, "route recurrent": 47, "suppressor->route": 18, "temporal role": 30, "token->hub": 55, "token->route": 89}` | `True` | `True` | `False` |
| `overcomplete_guided_train` | `{}` | `False` | `False` | `True` |
| `random_graph_baseline` | `{}` | `False` | `False` | `True` |

## Best Prune-50 Grouped Ablation

```json
{
  "final_test_accuracy": 0.71224,
  "final_test_authority": 0.007432,
  "selected_seed": 0,
  "validation_grouped_ablation": {
    "hub->route": {
      "accuracy_drop": 0.065104,
      "authority_drop": -0.021262,
      "edge_count": 6,
      "temporal_drop": 0.125
    },
    "route recurrent": {
      "accuracy_drop": -0.023438,
      "authority_drop": 0.003014,
      "edge_count": 5,
      "temporal_drop": 0.0
    },
    "route->readout": {
      "accuracy_drop": 0.317708,
      "authority_drop": 0.000959,
      "edge_count": 10,
      "temporal_drop": 0.125
    },
    "suppressor->route": {
      "accuracy_drop": 0.289063,
      "authority_drop": 0.018769,
      "edge_count": 14,
      "temporal_drop": 0.125
    },
    "temporal role": {
      "accuracy_drop": 0.065104,
      "authority_drop": -0.002347,
      "edge_count": 4,
      "temporal_drop": 0.125
    },
    "token->hub": {
      "accuracy_drop": 0.132812,
      "authority_drop": -0.018892,
      "edge_count": 9,
      "temporal_drop": 0.375
    },
    "token->route": {
      "accuracy_drop": 0.114583,
      "authority_drop": 0.002004,
      "edge_count": 23,
      "temporal_drop": 0.125
    }
  }
}
```

## Verdict

```json
{
  "guided_training_helps": false,
  "pruning_preserves_authority": false,
  "sparse_graph_emerges": false,
  "repair_after_pruning_helps": false,
  "overcomplete_scaffold_viable": false,
  "hand_seeded_still_dominates": false,
  "mutation_only_still_insufficient": true,
  "final_verdict_uses_final_test": true
}
```

## Interpretation Notes

- Positive evidence requires authority/refraction and route specialization, not only raw accuracy.
- If pruning preserves accuracy but hurts authority, this report treats that as a mechanism failure.
- Suppressor pruning and inactive leakage are reported explicitly in the pruning audit and influence metrics.

## Runtime Notes

- runtime seconds: `2058.213584`
- interrupted by wall clock: `False`
- completed records: `30`

## Claim Boundary

Toy evidence only. Do not claim consciousness, biology, full VRAXION behavior, or production validation.

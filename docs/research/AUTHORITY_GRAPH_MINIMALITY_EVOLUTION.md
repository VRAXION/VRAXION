# Authority Graph Minimality + Evolution

## Goal

Separate necessary mechanism from hand-coded convenience in the explicit frame-gated authority graph. This study keeps the same toy tasks and does not use neural layers, backprop, new semantic concepts, biology, or wave/reservoir extensions.

## Run Configuration

```json
{
  "study": "minimality",
  "seeds": 3,
  "steps": 5,
  "samples": 96,
  "mutation_steps": 25,
  "mutation_scale": 0.18,
  "population_size": 8,
  "random_graphs": 5,
  "decay": 0.35,
  "smoke": false
}
```

## Baseline

| Metric | Mean | Std |
|---|---:|---:|
| Accuracy | `0.989583` | `0.002835` |
| Authority / Refraction | `0.376955` | `0.021401` |
| Temporal Order Accuracy | `1.000000` | `0.000000` |
| Nodes | `48.000000` | `0.000000` |
| Edges | `91.000000` | `0.000000` |

## Part 1: Component Necessity Ablations

| Ablation | Accuracy Drop | Authority Drop | Temporal Order Drop | Verdict |
|---|---:|---:|---:|---|
| `no_shared_hubs` | `0.000000` | `0.044019` | `0.000000` | `helpful_bias` |
| `no_frame_routes` | `0.416667` | `0.376955` | `0.000000` | `necessary_core` |
| `no_frame_gates` | `0.092593` | `0.216673` | `0.000000` | `necessary_core` |
| `no_recurrence` | `0.395833` | `0.057181` | `0.500000` | `necessary_core` |
| `no_suppressors` | `-0.004630` | `0.073527` | `0.000000` | `useful_but_replaceable` |
| `no_direct_token_to_readout` | `0.000000` | `0.049564` | `0.000000` | `helpful_bias` |
| `no_route_to_readout` | `0.395833` | `0.376955` | `0.500000` | `necessary_core` |

## Part 2: Minimal Graph Sweep

### Node Shrink

| Budget | Retained Nodes | Top Accuracy | Random Accuracy | Top Authority | Random Authority | Top Temporal |
|---|---:|---:|---:|---:|---:|---:|
| `100pct` | `48.000000` | `0.989583` | `0.989583` | `0.352829` | `0.344793` | `1.000000` |
| `75pct` | `36.000000` | `0.809028` | `0.913194` | `0.336167` | `0.270364` | `0.500000` |
| `50pct` | `24.000000` | `0.692130` | `0.719907` | `0.194306` | `0.127633` | `0.500000` |
| `33pct` | `16.000000` | `0.590278` | `0.652778` | `0.017722` | `0.077170` | `0.500000` |
| `25pct` | `12.000000` | `0.593750` | `0.665509` | `0.000000` | `0.043429` | `0.500000` |

### Edge Shrink

| Budget | Retained Edges | Top Accuracy | Random Accuracy | Top Authority | Random Authority | Top Temporal |
|---|---:|---:|---:|---:|---:|---:|
| `100pct` | `91.000000` | `0.989583` | `0.989583` | `0.321314` | `0.332551` | `1.000000` |
| `75pct` | `68.000000` | `0.989583` | `0.885417` | `0.309903` | `0.223952` | `1.000000` |
| `50pct` | `46.000000` | `0.993056` | `0.746528` | `0.276018` | `0.137767` | `1.000000` |
| `33pct` | `30.000000` | `0.791667` | `0.675926` | `0.259629` | `0.062935` | `0.500000` |
| `25pct` | `23.000000` | `0.760417` | `0.619213` | `0.222789` | `0.043926` | `0.500000` |

Minimality summary by seed:

```json
[
  {
    "minimal_node_count_for_90pct_accuracy": {
      "budget": "100pct",
      "node_count": 48,
      "accuracy": 0.986111
    },
    "minimal_edge_count_for_90pct_accuracy": {
      "budget": "50pct",
      "edge_count": 46,
      "accuracy": 0.989583
    },
    "minimal_graph_preserving_90pct_authority": null,
    "temporal_order_breaks_first_at": {
      "kind": "node_shrink",
      "budget": "75pct"
    }
  },
  {
    "minimal_node_count_for_90pct_accuracy": {
      "budget": "100pct",
      "node_count": 48,
      "accuracy": 0.989583
    },
    "minimal_edge_count_for_90pct_accuracy": {
      "budget": "50pct",
      "edge_count": 46,
      "accuracy": 0.993056
    },
    "minimal_graph_preserving_90pct_authority": null,
    "temporal_order_breaks_first_at": {
      "kind": "node_shrink",
      "budget": "75pct"
    }
  },
  {
    "minimal_node_count_for_90pct_accuracy": {
      "budget": "100pct",
      "node_count": 48,
      "accuracy": 0.993056
    },
    "minimal_edge_count_for_90pct_accuracy": {
      "budget": "50pct",
      "edge_count": 46,
      "accuracy": 0.996528
    },
    "minimal_graph_preserving_90pct_authority": {
      "kind": "edge_shrink",
      "budget": "75pct",
      "edge_count": 68,
      "authority": 0.336934
    },
    "temporal_order_breaks_first_at": {
      "kind": "node_shrink",
      "budget": "75pct"
    }
  }
]
```

## Part 3: Evolution From Weaker Priors

| Seed Graph | Start Fitness | Final Fitness | Start Accuracy | Final Accuracy | Final Authority | Final Edges |
|---|---:|---:|---:|---:|---:|---:|
| `random_graph` | `0.573948` | `0.604522` | `0.472222` | `0.495370` | `-0.038647` | `90.000000` |
| `hub_only_graph` | `0.728071` | `0.729271` | `0.593750` | `0.593750` | `-0.000835` | `31.333333` |
| `route_only_graph` | `0.708546` | `0.733500` | `0.590278` | `0.600694` | `-0.026528` | `21.000000` |
| `no_suppressor_seed` | `1.491335` | `1.516462` | `0.994213` | `0.993056` | `0.390795` | `76.000000` |
| `damaged_hand_seeded_30` | `1.206139` | `1.330839` | `0.840278` | `0.902778` | `0.258345` | `64.000000` |
| `damaged_hand_seeded_50` | `1.090715` | `1.172536` | `0.773148` | `0.804398` | `0.236929` | `44.666667` |
| `damaged_hand_seeded_70` | `1.063144` | `1.124308` | `0.771991` | `0.792824` | `0.199640` | `30.000000` |

### Recovered Structure Rates

| Seed Graph | Shared Hubs | Frame Routes | Suppressors | Recurrence |
|---|---:|---:|---:|---:|
| `random_graph` | `0.000000` | `1.000000` | `1.000000` | `0.000000` |
| `hub_only_graph` | `1.000000` | `0.000000` | `0.000000` | `1.000000` |
| `route_only_graph` | `0.000000` | `1.000000` | `0.000000` | `1.000000` |
| `no_suppressor_seed` | `1.000000` | `1.000000` | `0.333333` | `1.000000` |
| `damaged_hand_seeded_30` | `1.000000` | `1.000000` | `1.000000` | `1.000000` |
| `damaged_hand_seeded_50` | `1.000000` | `1.000000` | `1.000000` | `1.000000` |
| `damaged_hand_seeded_70` | `1.000000` | `1.000000` | `1.000000` | `0.666667` |

### Accepted Mutation Types

```json
{
  "random_graph": {
    "edge_add": {
      "mean": 3.0,
      "std": 1.414214
    },
    "edge_gain_perturb": {
      "mean": 1.333333,
      "std": 0.471405
    },
    "edge_remove": {
      "mean": 4.333333,
      "std": 1.699673
    },
    "edge_sign_flip": {
      "mean": 1.5,
      "std": 0.5
    },
    "frame_gate_strength": {
      "mean": 2.5,
      "std": 0.5
    },
    "recurrent_decay": {
      "mean": 2.5,
      "std": 1.5
    },
    "suppressor_strength": {
      "mean": 2.0,
      "std": 0.0
    }
  },
  "hub_only_graph": {
    "edge_add": {
      "mean": 2.0,
      "std": 0.0
    },
    "edge_gain_perturb": {
      "mean": 2.666667,
      "std": 0.471405
    },
    "edge_remove": {
      "mean": 3.0,
      "std": 0.816497
    },
    "edge_sign_flip": {
      "mean": 4.333333,
      "std": 1.247219
    },
    "frame_gate_strength": {
      "mean": 3.0,
      "std": 0.816497
    },
    "recurrent_decay": {
      "mean": 3.0,
      "std": 0.816497
    }
  },
  "route_only_graph": {
    "edge_add": {
      "mean": 2.333333,
      "std": 0.942809
    },
    "edge_gain_perturb": {
      "mean": 2.0,
      "std": 0.816497
    },
    "edge_remove": {
      "mean": 3.0,
      "std": 0.816497
    },
    "edge_sign_flip": {
      "mean": 2.666667,
      "std": 1.699673
    },
    "frame_gate_strength": {
      "mean": 2.333333,
      "std": 0.942809
    },
    "recurrent_decay": {
      "mean": 4.333333,
      "std": 2.054805
    },
    "suppressor_strength": {
      "mean": 1.666667,
      "std": 0.471405
    }
  },
  "no_suppressor_seed": {
    "edge_add": {
      "mean": 2.333333,
      "std": 0.471405
    },
    "edge_gain_perturb": {
      "mean": 3.5,
      "std": 1.5
    },
    "edge_remove": {
      "mean": 1.666667,
      "std": 0.471405
    },
    "edge_sign_flip": {
      "mean": 2.0,
      "std": 0.0
    },
    "frame_gate_strength": {
      "mean": 3.5,
      "std": 1.5
    },
    "recurrent_decay": {
      "mean": 2.0,
      "std": 0.0
    },
    "suppressor_strength": {
      "mean": 1.666667,
      "std": 0.471405
    }
  },
  "damaged_hand_seeded_30": {
    "edge_add": {
      "mean": 2.5,
      "std": 0.5
    },
    "edge_gain_perturb": {
      "mean": 2.0,
      "std": 0.816497
    },
    "edge_remove": {
      "mean": 1.5,
      "std": 0.5
    },
    "edge_sign_flip": {
      "mean": 3.666667,
      "std": 1.247219
    },
    "frame_gate_strength": {
      "mean": 4.333333,
      "std": 2.624669
    },
    "recurrent_decay": {
      "mean": 3.0,
      "std": 0.816497
    },
    "suppressor_strength": {
      "mean": 2.0,
      "std": 0.816497
    }
  },
  "damaged_hand_seeded_50": {
    "edge_add": {
      "mean": 3.0,
      "std": 0.0
    },
    "edge_gain_perturb": {
      "mean": 2.0,
      "std": 1.414214
    },
    "edge_remove": {
      "mean": 3.0,
      "std": 1.414214
    },
    "edge_sign_flip": {
      "mean": 2.0,
      "std": 0.816497
    },
    "frame_gate_strength": {
      "mean": 3.333333,
      "std": 1.885618
    },
    "recurrent_decay": {
      "mean": 2.666667,
      "std": 1.247219
    },
    "suppressor_strength": {
      "mean": 3.333333,
      "std": 1.699673
    }
  },
  "damaged_hand_seeded_70": {
    "edge_add": {
      "mean": 2.666667,
      "std": 1.247219
    },
    "edge_gain_perturb": {
      "mean": 1.333333,
      "std": 0.471405
    },
    "edge_remove": {
      "mean": 2.0,
      "std": 0.816497
    },
    "edge_sign_flip": {
      "mean": 2.666667,
      "std": 0.942809
    },
    "frame_gate_strength": {
      "mean": 1.5,
      "std": 0.5
    },
    "recurrent_decay": {
      "mean": 3.0,
      "std": 0.0
    },
    "suppressor_strength": {
      "mean": 3.0,
      "std": 0.0
    }
  }
}
```

## Part 4: Necessity Summary

| Component | Ablation Effect | Evolution Recovery | Verdict |
|---|---|---|---|
| `input_token_nodes` | required input ports by construction | damaged seeds partially tested; weak-prior recovery is reported above | `necessary_core` |
| `shared_hubs` | see no_shared_hubs | damaged seeds partially tested; weak-prior recovery is reported above | `helpful_bias` |
| `frame_route_nodes` | see no_frame_routes | damaged seeds partially tested; weak-prior recovery is reported above | `necessary_core` |
| `frame_gates` | see no_frame_gates | damaged seeds partially tested; weak-prior recovery is reported above | `necessary_core` |
| `recurrent_edges` | see no_recurrence | damaged seeds partially tested; weak-prior recovery is reported above | `necessary_core` |
| `suppressors` | see no_suppressors | damaged seeds partially tested; weak-prior recovery is reported above | `useful_but_replaceable` |
| `readout_ports` | see no_route_to_readout | damaged seeds partially tested; weak-prior recovery is reported above | `necessary_core` |
| `direct_shortcuts` | see no_direct_token_to_readout | damaged seeds partially tested; weak-prior recovery is reported above | `helpful_bias` |

## Verdict

```json
{
  "supports_frame_gates_as_necessary_core": true,
  "supports_route_structure_as_necessary_core": true,
  "supports_recurrence_as_temporal_core": true,
  "supports_suppressors_as_helpful": true,
  "supports_mutation_recovery_from_damage": true,
  "supports_mutation_recovery_from_weak_priors": false,
  "random_graph_remains_hard": true,
  "manual_structure_still_dominates": true
}
```

## Interpretation

- If damaged hand graphs recover but random or weak priors lag, the hand-coded mechanism is still doing substantial work.
- If route-only or hub-only priors recover, the architecture is more mutatable and less dependent on exact manual edges.
- Accuracy alone is not decisive; authority/refraction and wrong-frame controls are treated as mechanism metrics.

## Runtime Notes

- total runtime seconds: `352.010231`
- smoke mode: `False`

## Claim Boundary

Toy evidence only. Do not claim consciousness, biology, full VRAXION behavior, or production validation.

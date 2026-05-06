# RNN Teacher Authority Extraction

## Why This Was Tested

The explicit hand-seeded authority graph works, while developmental graph search still struggles to rediscover the sign/gain structure. This probe uses a tiny GRU as a backprop-trained teacher, then attempts to extract a smaller explicit authority graph from the teacher's behavior.

## Task Setup

The teacher sees toy token sequences, not natural language. Tasks are binary classifications over:

- `temporal_order_contrast`
- `query_cued_multi_aspect`
- `latent_refraction_sequence`

Query tokens are `danger_query`, `friendship_query`, `sound_query`, and `environment_query`. The same observation can appear under different queries with different labels.

## Run Configuration

```json
{
  "seeds": 2,
  "hidden_sizes": [
    32,
    64
  ],
  "embedding_dim": 24,
  "epochs": 80,
  "batch_size": 64,
  "train_size": 512,
  "validation_size": 256,
  "test_size": 512,
  "learning_rate": 0.002,
  "aux_frame_weight": 0.15,
  "query_position": "both",
  "extract_ridge": 0.15,
  "torch_threads": 2,
  "smoke": false
}
```

## Teacher Results By Hidden Size

| Hidden | Accuracy | Temporal | Query-cued | Frame pred | Zero recurrent | Randomized recurrent | Shuffled order | Query ablation | Query shuffle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `32` | `0.940430` | `1.000000` | `0.921392` | `0.988402` | `0.628906` | `0.660156` | `0.625977` | `0.774414` | `0.748047` |
| `64` | `0.970703` | `1.000000` | `0.961340` | `1.000000` | `0.607422` | `0.636719` | `0.666016` | `0.774414` | `0.763672` |

## Baselines

- bag-of-tokens MLP accuracy: `0.833008`
- static-with-position MLP accuracy: `0.962891`

## Authority Diagnostics

- active group influence: `0.441660`
- inactive group influence: `0.157744`
- authority/refraction score: `0.283917`

## Hidden Trajectory Diagnostics

- suffix divergence, `dog bite me` vs `dog bite snake`: `0.692222`
- order divergence, `dog bite me` vs `me bite dog`: `0.802098`

## Probe Diagnostics

- frame_probe_accuracy: `1.000000`
- actor_probe_accuracy: `0.612113`
- action_probe_accuracy: `0.579897`
- relation_probe_accuracy: `0.742268`
- sound_probe_accuracy: `0.744845`
- place_probe_accuracy: `0.731959`
- noise_probe_accuracy: `0.807990`

## Unit Saliency

- baseline accuracy before single-dim ablation: `0.970703`
- baseline authority/refraction before ablation: `0.266545`

Frame-related activation classifications, averaged across seeds:
- danger_related: `15.000000`
- environment_related: `14.000000`
- friendship_related: `19.500000`
- sound_related: `14.500000`
- unclear: `1.000000`

Example top hidden dimensions by single-dim accuracy drop:

```json
[
  {
    "dim": 2,
    "accuracy_drop": 0.005859
  },
  {
    "dim": 5,
    "accuracy_drop": 0.005859
  },
  {
    "dim": 38,
    "accuracy_drop": 0.005859
  },
  {
    "dim": 47,
    "accuracy_drop": 0.005859
  },
  {
    "dim": 62,
    "accuracy_drop": 0.005859
  },
  {
    "dim": 0,
    "accuracy_drop": 0.003906
  },
  {
    "dim": 3,
    "accuracy_drop": 0.003906
  },
  {
    "dim": 4,
    "accuracy_drop": 0.003906
  },
  {
    "dim": 8,
    "accuracy_drop": 0.003906
  },
  {
    "dim": 9,
    "accuracy_drop": 0.003906
  }
]
```


## Extracted Graph Attempt

The extracted graph is fitted from teacher logits using ridge least squares over token/role features. It does not use neural backprop internally. This is an approximation attempt, not a guaranteed faithful circuit recovery.

| Graph | Accuracy | Temporal | Authority | Wrong query/frame drop | Edges |
|---|---:|---:|---:|---:|---:|
| `extracted_authority_graph` | `0.767764` | `0.854839` | `0.123419` | `0.213918` | `116.000000` |
| `hand_seeded_authority_graph` | `0.908935` | `1.000000` | `0.337016` | `0.301546` | `91.000000` |
| `random_graph_baseline` | `0.419410` | `0.387097` | `-0.027839` | `-0.012887` | `91.000000` |

## Verdict

```json
{
  "gru_teacher_solves_tasks": true,
  "gru_teacher_uses_recurrence": true,
  "hidden_authority_structure_detected": true,
  "extraction_to_authority_graph_successful": true,
  "authority_graph_matches_teacher_partially": true,
  "rnn_teacher_useful_for_search": true,
  "shortcut_risk_detected": true,
  "hand_seeded_still_stronger_than_extracted": true
}
```

## Interpretation

A positive teacher result means backprop can find a recurrent solution for the toy tasks. Extraction success is judged separately: a GRU can solve the task while still being hard to compress into the explicit authority graph.

The teacher is strong and recurrence-sensitive, but there is a shortcut caveat: the position-aware static MLP is close to the GRU on raw accuracy. The useful GRU-specific evidence is therefore the combination of recurrence/randomization drops, query ablation drops, hidden trajectory divergence, and authority/influence geometry rather than accuracy alone.

The extracted graph is a meaningful partial transfer, but it is not yet compact: it uses more edges than the hand-seeded graph and remains well below the hand graph on authority/refraction. This supports using a GRU teacher as a search guide, not as a solved extraction pipeline.

## Runtime Notes

- runtime seconds: `38.607768`
- completed records: `2`

## Claim Boundary

Toy evidence only. Do not claim consciousness, biology, production validation, full VRAXION behavior, or natural-language understanding.

# E29 Real-Text Flow/Pocket vs MLP Unresolved Training Confirm Contract

## Purpose

E29 runs the user's requested 1v1 comparison: the E28 MLP baseline versus a small Flow/Pocket-matrix text model on the same weakly supervised FineWeb-Edu unresolved-action task.

This is not a full VRAXION deployment. It is a real-text proxy comparison.

## Systems

```text
flow_pocket_matrix_text_gradient
tiny_hash_mlp_real_text_gradient
keyword_regex_reference
majority_answer_baseline
random_control
```

The Flow/Pocket-matrix system uses:

```text
hashed text features
-> input_adapter
-> shared Flow[D]
-> router logits
-> action-specific pocket matrices
-> commit matrix
-> action readout
```

The MLP baseline uses the same hashed text features and a dense hidden layer.

## Metrics

- heldout action accuracy
- phrase-holdout action accuracy
- wrong confident answer on unresolved text
- false ask on answerable/neutral text
- non-answer justified rate
- parameter count
- deterministic replay
- checker failure count

## Decision Labels

```text
e29_flow_pocket_matrix_beats_mlp_on_real_text_unresolved
e29_flow_pocket_matrix_matches_mlp_with_better_abstention
e29_mlp_baseline_beats_flow_pocket_matrix
e29_real_text_needs_contrastive_bridge_for_both
e29_no_clear_real_text_winner
```

## Boundary

E29 compares two small learned systems on a mined real-text proxy. It does not prove open-ended chat behavior, raw language reasoning, AGI, consciousness, deployed-model readiness, or model-scale behavior.

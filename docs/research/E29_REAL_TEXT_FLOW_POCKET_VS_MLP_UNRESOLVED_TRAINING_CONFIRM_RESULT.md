# E29 Real-Text Flow/Pocket vs MLP Unresolved Training Confirm Result

Status: complete.

## Decision

```text
decision = e29_real_text_needs_contrastive_bridge_for_both
checker_failure_count = 0
target_checker_passed = true
sample_only_checker_passed = true
```

## Data

```text
parquet_root = S:\AI\MESSY TRAINING DATA - INPUT ONLY\Fineweb edu 10B
rows_seen = 288000
examples_selected = 6400
balanced actions:
  ANSWER = 1600
  ASK_FOR_EVIDENCE = 1600
  SEARCH_MORE = 1600
  HOLD_UNRESOLVED = 1600
```

## 1v1 Result

```text
flow_pocket_matrix_text_gradient:
  params = 440552
  train = 1.000000
  validation = 0.752518
  heldout = 0.780059
  phrase_holdout = 0.298651
  wrong_confident = 0.185417
  false_ask = 0.058125

tiny_hash_mlp_real_text_gradient:
  params = 656164
  train = 0.981823
  validation = 0.755396
  heldout = 0.785924
  phrase_holdout = 0.346821
  wrong_confident = 0.143750
  false_ask = 0.100625
```

## Interpretation

The Flow/Pocket-matrix model did not beat the MLP on this real-text mined task. It used fewer parameters and had a lower false-ask rate, but it overfit harder and had worse phrase-holdout generalization and a higher wrong-confident rate.

This means the current Flow/Pocket-matrix text version is not yet enough to turn FineWeb-mined weak labels into robust unresolved-state behavior. Both learned systems need a contrastive bridge: mined real examples plus controlled answerable/unresolved pairs, evidence-span supervision, and explicit pressure against wrong confident answers.

The keyword regex reference wins because the labels are regex-mined; that is a diagnostic upper/reference, not a valid learned reasoning result.

## Boundary

E29 compares two small learned systems on a mined real-text proxy. It does not prove open-ended chat behavior, raw language reasoning, deployed-model readiness, AGI, consciousness, or model-scale behavior.

## Naming Note

Future Flow/Pocket probes should use the canonical naming scheme in `FLOW_POCKET_NAMING_SCHEME.md`:

```text
Ground Field
Flow Field
Pocket Operator
Lens Pocket
Writer Pocket
Arbiter
Trace Ledger
Ingress Codec
Egress Codec
```

# E28 Real-Text Unresolved Information-Seeking Training Audit Result

Status: complete.

The run artifacts are expected under:

```text
target/pilot_wave/e28_real_text_unresolved_information_seeking_training_audit/
```

The committed sample pack is expected under:

```text
docs/research/artifact_samples/e28_real_text_unresolved_information_seeking_training/
```

## Decision

```text
decision = e28_real_text_signal_sparse_needs_synthetic_bridge
checker_failure_count = 0
target_checker_passed = true
sample_only_checker_passed = true
```

## Data Audit

```text
parquet_root = S:\AI\MESSY TRAINING DATA - INPUT ONLY\Fineweb edu 10B
rows_seen = 288000
row_groups_seen = 288
examples_selected = 6400

ANSWER = 1600
ASK_FOR_EVIDENCE = 1600
SEARCH_MORE = 1600
HOLD_UNRESOLVED = 1600
```

The mined labels are weak supervision from natural web text, not human-reviewed instruction/chat labels.

## Key Metrics

```text
tiny_hash_mlp_real_text_gradient:
  device = cuda
  parameters = 656164
  train_action_accuracy = 0.985863
  validation_action_accuracy = 0.769784
  heldout_action_accuracy = 0.809384
  phrase_holdout_action_accuracy = 0.321773
  wrong_confident_answer_on_unresolved = 0.177708
  false_ask_on_answerable = 0.063125

keyword_regex_reference:
  heldout_action_accuracy = 1.000000
  phrase_holdout_action_accuracy = 1.000000
  wrong_confident_answer_on_unresolved = 0.000000

majority_answer_baseline:
  heldout_action_accuracy = 0.322581
  wrong_confident_answer_on_unresolved = 1.000000
```

## Interpretation

FineWeb-Edu contains real natural uncertainty and information-seeking phrases, so the idea is viable as a data source. But the small gradient-trained text model over mined snippets overfit surface patterns: train reached 0.986 while validation stayed near 0.770, phrase-holdout fell to 0.322, and wrong confident answers on unresolved text remained 0.178.

Therefore, real webtext alone is not yet enough to produce the E27-style "do not answer when evidence is insufficient" behavior. The next step should mix mined real examples with controlled contrastive unresolved/answerable pairs and evidence-span supervision.

## Boundary

E28 is a real-text feasibility audit. It does not prove open-ended chat behavior, raw language reasoning, deployed-model readiness, AGI, consciousness, or model-scale behavior.

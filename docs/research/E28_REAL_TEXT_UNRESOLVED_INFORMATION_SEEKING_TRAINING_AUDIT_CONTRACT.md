# E28 Real-Text Unresolved Information-Seeking Training Audit Contract

## Purpose

E28 tests whether the local FineWeb-Edu parquet corpus contains enough natural uncertainty / missing-evidence / information-seeking signal to train a small real-text model to choose a non-answer action instead of making a confident answer.

This is a bridge from E27's controlled symbolic/text proxy toward real web text. It is not a chatbot test and not a model-scale claim.

## Data Source

Default source:

```text
S:\AI\MESSY TRAINING DATA - INPUT ONLY\Fineweb edu 10B
```

The runner mines weakly supervised snippets from parquet rows using mechanical text patterns such as:

- not enough evidence / insufficient information
- cannot determine / we do not know
- more research is needed / gather more data
- consult a doctor / ask an expert
- hard negatives containing evidence/information words without unresolved meaning

Labels are weak supervision and must not be treated as human-reviewed truth.

## Systems

Required systems:

```text
tiny_hash_mlp_real_text_gradient
keyword_regex_reference
majority_answer_baseline
random_control
```

## Actions

```text
ANSWER
ASK_FOR_EVIDENCE
SEARCH_MORE
HOLD_UNRESOLVED
```

## Required Artifacts

```text
backend_manifest.json
dataset_mining_report.json
mined_real_text_examples.jsonl
training_curve_report.json
system_results.json
aggregate_metrics.json
deterministic_replay.json
resource_usage_report.json
decision.json
summary.json
report.md
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
row_level_results.jsonl
```

Sample pack:

```text
docs/research/artifact_samples/e28_real_text_unresolved_information_seeking_training/
```

## Metrics

- heldout action accuracy
- phrase-holdout action accuracy
- wrong confident answer on unresolved text
- false ask on answerable/neutral text
- non-answer justified rate
- examples mined per action
- training curve
- deterministic replay hash match
- checker failure count

## Decision Labels

```text
e28_real_text_unresolved_training_signal_present
e28_real_text_regex_shortcut_risk_detected
e28_real_text_signal_sparse_needs_synthetic_bridge
e28_real_text_dataset_missing_or_invalid
```

## Boundary

E28 is a real-text feasibility audit. It does not prove open-ended language reasoning, production chatbot behavior, AGI, consciousness, or model-scale behavior.

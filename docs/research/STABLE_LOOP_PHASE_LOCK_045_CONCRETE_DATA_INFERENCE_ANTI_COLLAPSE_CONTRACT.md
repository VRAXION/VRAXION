# STABLE_LOOP_PHASE_LOCK_045_CONCRETE_DATA_INFERENCE_ANTI_COLLAPSE Contract

## Summary

044 completed the first bounded final-training candidate. 045 is the next
behavioral gate: concrete train -> inference on explicit examples.

Core question:

```text
Does the system learn input-conditioned outputs on explicit training data,
or does it collapse into static output behavior?
```

This is not scaleout. Do not enable production default training, promote public
beta, or claim full VRAXION, language grounding, or consciousness.

## Dataset Shape

Each row is explicit data:

```json
{
  "id": "train_000001",
  "split": "train|heldout|ood",
  "task_family": "route_answer|context_carry|symbolic_map|non_route_control",
  "input": "...",
  "expected_output": "...",
  "anti_shortcut_group": "..."
}
```

Required task families:

```text
route_answer
context_carry
symbolic_map
non_route_control
```

The route-answer family must vary all phase labels:

```text
phase_0
phase_1
phase_2
phase_3
```

## Required Arms

```text
NO_TRAIN_BASELINE
BASE_TRAIN_NO_ROUTE_GRAMMAR
FINAL_TRAINING_044_REFERENCE
ROUTE_GRAMMAR_TRAIN_AND_INFER
ROUTE_GRAMMAR_TRAIN_AND_INFER_ROLLBACK_GATED
ROUTE_GRAMMAR_INFERENCE_ONLY_ABLATION
ROUTE_GRAMMAR_TRAIN_ONLY_ABLATION
ROUTE_GRAMMAR_SHUFFLED_LABELS
ROUTE_GRAMMAR_SHUFFLED_INPUT_ORDER
NON_ROUTE_REGRESSION_CONTROL
ALWAYS_SPACE_CONTROL
ALWAYS_EMPTY_CONTROL
ALWAYS_MAJORITY_CONTROL
ALWAYS_PHASE_0_CONTROL
COPY_LAST_TOKEN_CONTROL
COPY_FIRST_TOKEN_CONTROL
ANSWER_ONLY_SHORTCUT_CONTROL
TRAIN_LABEL_PRIOR_CONTROL
RANDOM_LABEL_CONTROL
RANDOM_PHASE_RULE_CONTROL
```

## Required Outputs

```text
queue.json
progress.jsonl
metrics.jsonl
concrete_inference_metrics.jsonl
control_metrics.jsonl
train_examples_sample.jsonl
heldout_examples_sample.jsonl
ood_examples_sample.jsonl
inference_samples.jsonl
prediction_distribution.json
collapse_metrics.json
confusion_matrix.json
per_family_metrics.json
bad_cases.jsonl
summary.json
report.md
contract_snapshot.md
job_progress/*.jsonl
```

## Required Metrics

Accuracy:

```text
train_exact_accuracy
heldout_exact_accuracy
ood_exact_accuracy
per_family_accuracy
family_min_accuracy
```

Anti-collapse:

```text
unique_output_count
unique_output_rate
top_output_rate
space_only_rate
empty_output_rate
majority_output_rate
average_output_length
output_entropy
repetition_rate
copy_last_token_rate
copy_first_token_rate
static_output_score
```

Generalization:

```text
heldout_gap
ood_gap
new_key_value_accuracy
new_route_length_accuracy
new_template_accuracy
```

Regression:

```text
non_route_accuracy
non_route_regression_delta
false_route_activation_rate
route_api_overuse_rate
rollback_success
checkpoint_save_load_pass
```

## Positive Gate

`CONCRETE_INFERENCE_ANTI_COLLAPSE_POSITIVE` requires:

```text
heldout_exact_accuracy >= 0.90
ood_exact_accuracy >= 0.80
family_min_accuracy >= 0.75

top_output_rate <= 0.35
space_only_rate <= 0.01
empty_output_rate <= 0.01
majority_output_rate <= 0.35
output_entropy >= minimum_expected_entropy
unique_output_count >= expected_output_class_count

ALWAYS_SPACE fails
ALWAYS_EMPTY fails
ALWAYS_MAJORITY fails
COPY_LAST_TOKEN fails
SHUFFLED_LABELS fails

non_route_regression_delta >= -0.02
random controls fail
rollback_success = true
checkpoint_save_load_pass = true
```

## Verdicts

```text
CONCRETE_DATA_INFERENCE_POSITIVE
INPUT_CONDITIONED_OUTPUTS_LEARNED
STATIC_OUTPUT_COLLAPSE_REJECTED
SPACE_OUTPUT_COLLAPSE_REJECTED
MAJORITY_LABEL_SHORTCUT_REJECTED
COPY_SHORTCUT_REJECTED
HELDOUT_GENERALIZATION_PASSES
OOD_GENERALIZATION_PASSES
NON_ROUTE_REGRESSION_CLEAN
SHUFFLED_LABELS_FAIL
ALWAYS_SPACE_CONTROL_FAILS
ALWAYS_MAJORITY_CONTROL_FAILS
TRAINING_SIGNAL_COLLAPSES_TO_STATIC_OUTPUT
FINAL_TRAINING_NEEDS_MORE_BEHAVIORAL_WORK
PRODUCTION_API_NOT_READY
```

## Claim Boundary

045 can support:

```text
bounded concrete-data inference anti-collapse behavior
```

045 cannot support:

```text
scaleout final training
production default training
public beta promotion
production API readiness
full VRAXION
language grounding
consciousness
```

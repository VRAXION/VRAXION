# STABLE_LOOP_PHASE_LOCK_046_CONCRETE_INFERENCE_SCALEOUT Contract

## Summary

045 proved bounded concrete train -> inference anti-collapse behavior. 046 is
the next scaleout probe, still not production training.

Core question:

```text
Does input-conditioned concrete inference survive a larger and more diverse
suite with longer inputs, more output classes, more templates, and stronger
shortcut controls?
```

Do not enable production default training, promote public beta, or claim full
VRAXION, language grounding, or consciousness.

## Dataset Shape

Each row remains explicit concrete data:

```json
{
  "id": "heldout_000001",
  "split": "train|heldout|ood",
  "task_family": "route_answer|long_route_answer|context_carry|multi_memory|symbolic_map|compositional_map|arithmetic_transform|non_route_control",
  "input": "...",
  "expected_output": "...",
  "anti_shortcut_group": "..."
}
```

Scaleout additions over 045:

```text
long_route_answer: longer gate/phase chains
multi_memory: multiple key/value records with queried key
compositional_map: two-step symbolic map
arithmetic_transform: numeric symbolic transform
larger output vocabulary
OOD reordered templates
```

## Required Arms

```text
NO_TRAIN_BASELINE
BASE_TRAIN_NO_ROUTE_GRAMMAR
CONCRETE_INFERENCE_045_REFERENCE
ROUTE_GRAMMAR_SCALEOUT_TRAIN_AND_INFER
ROUTE_GRAMMAR_SCALEOUT_ROLLBACK_GATED
ROUTE_GRAMMAR_SCALEOUT_INFERENCE_ONLY_ABLATION
ROUTE_GRAMMAR_SCALEOUT_TRAIN_ONLY_ABLATION
ROUTE_GRAMMAR_SCALEOUT_SHUFFLED_LABELS
ROUTE_GRAMMAR_SCALEOUT_SHUFFLED_INPUT_ORDER
NON_ROUTE_SCALEOUT_REGRESSION_CONTROL
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

## Metrics

```text
train_exact_accuracy
heldout_exact_accuracy
ood_exact_accuracy
family_min_accuracy
per_family_accuracy
unique_output_count
expected_output_class_count
top_output_rate
space_only_rate
empty_output_rate
majority_output_rate
output_entropy
copy_last_token_rate
copy_first_token_rate
static_output_score
new_key_value_accuracy
new_route_length_accuracy
new_template_accuracy
non_route_accuracy
non_route_regression_delta
false_route_activation_rate
route_api_overuse_rate
rollback_success
checkpoint_save_load_pass
```

## Positive Gate

`CONCRETE_INFERENCE_SCALEOUT_POSITIVE` requires:

```text
heldout_exact_accuracy >= 0.92
ood_exact_accuracy >= 0.85
family_min_accuracy >= 0.80

top_output_rate <= 0.35
space_only_rate <= 0.01
empty_output_rate <= 0.01
majority_output_rate <= 0.35
output_entropy >= 2.25
unique_output_count >= expected_output_class_count

long route, multi-memory, compositional, arithmetic, and non-route families pass
ALWAYS_SPACE / ALWAYS_EMPTY / ALWAYS_MAJORITY / COPY controls fail
SHUFFLED_LABELS fails
random controls fail
rollback_success = true
checkpoint_save_load_pass = true
production_default_training_enabled = false
```

## Verdicts

```text
CONCRETE_INFERENCE_SCALEOUT_POSITIVE
SCALEOUT_INPUT_CONDITIONING_SURVIVES
LONG_SEQUENCE_GENERALIZATION_PASSES
MORE_OUTPUT_CLASSES_PASS
STATIC_OUTPUT_COLLAPSE_REJECTED
SPACE_OUTPUT_COLLAPSE_REJECTED
MAJORITY_LABEL_SHORTCUT_REJECTED
COPY_SHORTCUT_REJECTED
HELDOUT_GENERALIZATION_PASSES
OOD_GENERALIZATION_PASSES
NON_ROUTE_REGRESSION_CLEAN
SHUFFLED_LABELS_FAIL
SCALEOUT_NEEDS_MORE_BEHAVIORAL_WORK
PRODUCTION_API_NOT_READY
```

## Claim Boundary

046 can support:

```text
concrete inference behavior surviving a larger/diverser bounded suite
```

046 cannot support:

```text
production default training
public beta promotion
production API readiness
full VRAXION
language grounding
consciousness
```

# STABLE_LOOP_PHASE_LOCK_049_ADVERSARIAL_FROZEN_EVAL_SCALE Contract

## Summary

048 passed a small committed frozen eval corpus and caught a real exact-input
leakage issue before the corpus was fixed. 049 scales that idea into a larger
adversarial frozen corpus with hard distractors, longer OOD routes, more output
classes, and stricter leakage audits.

This is not a new capability probe and not production rollout. It does not
enable production default training, promote public beta, or claim full VRAXION,
language grounding, or consciousness.

## Core Question

```text
Does the 048 frozen-corpus input-conditioned behavior survive a larger,
adversarial frozen eval corpus without exact, near-duplicate, or semantic
train/eval leakage?
```

## Frozen Eval Source

```text
docs/research/STABLE_LOOP_PHASE_LOCK_049_ADVERSARIAL_FROZEN_EVAL_CORPUS.jsonl
```

The corpus is committed data and is loaded by the runner with `include_str!`.

## Required Arms

```text
NO_TRAIN_BASELINE
NO_ROUTE_GRAMMAR_ADVERSARIAL_FROZEN_BASELINE
FROZEN_EVAL_048_REFERENCE
ADVERSARIAL_FROZEN_ROUTE_GRAMMAR_TRAIN_AND_INFER
ADVERSARIAL_FROZEN_ROUTE_GRAMMAR_ROLLBACK_GATED
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

## Metrics

```text
train_exact_accuracy
heldout_exact_accuracy
ood_exact_accuracy
family_min_accuracy
template_holdout_accuracy
family_holdout_accuracy
hard_distractor_accuracy
long_ood_accuracy
frozen_eval_row_count
frozen_eval_unique_ids
train_eval_id_overlap_count
train_eval_input_overlap_count
train_eval_near_duplicate_count
train_eval_semantic_overlap_count
max_train_eval_token_jaccard
unique_output_count
expected_output_class_count
top_output_rate
majority_output_rate
space_only_rate
empty_output_rate
output_entropy
static_output_score
collapse_detected
copy_last_token_rate
copy_first_token_rate
non_route_regression_delta
checkpoint_save_load_pass
rollback_success
```

## Positive Gate

```text
ADVERSARIAL_FROZEN_EVAL_SCALE_POSITIVE if:

heldout_exact_accuracy >= 0.90
ood_exact_accuracy >= 0.85
family_min_accuracy >= 0.80
template_holdout_accuracy >= 0.85
family_holdout_accuracy >= 0.85
hard_distractor_accuracy >= 0.85
long_ood_accuracy >= 0.85
train_eval_id_overlap_count = 0
train_eval_input_overlap_count = 0
train_eval_near_duplicate_count = 0
train_eval_semantic_overlap_count = 0
unique_output_count >= expected_output_class_count
top_output_rate <= 0.35
majority_output_rate <= 0.35
space_only_rate <= 0.01
empty_output_rate <= 0.01
output_entropy >= minimum_expected_entropy
non_route_regression_delta >= -0.02
random / shuffled / static controls fail
rollback_success = true for rollback arm
checkpoint_save_load_pass = true for passing route grammar arms
```

## Required Outputs

```text
queue.json
progress.jsonl
metrics.jsonl
adversarial_frozen_metrics.jsonl
frozen_eval_manifest.json
leakage_audit.jsonl
prediction_distribution.json
collapse_metrics.json
confusion_matrix.json
per_family_metrics.json
inference_samples.jsonl
bad_cases.jsonl
summary.json
report.md
contract_snapshot.md
job_progress/*.jsonl
```

## Verdicts

```text
ADVERSARIAL_FROZEN_EVAL_SCALE_POSITIVE
ADVERSARIAL_FROZEN_INPUT_CONDITIONING_PASSES
ADVERSARIAL_FROZEN_NO_TRAIN_LEAKAGE
NEAR_DUPLICATE_LEAKAGE_AUDIT_PASSES
SEMANTIC_OVERLAP_AUDIT_PASSES
TEMPLATE_HOLDOUT_PASSES
FAMILY_HOLDOUT_PASSES
HARD_DISTRACTOR_PASSES
LONG_OOD_PASSES
STATIC_OUTPUT_COLLAPSE_REJECTED
MAJORITY_LABEL_SHORTCUT_REJECTED
COPY_SHORTCUT_REJECTED
SHUFFLED_LABELS_FAIL
RANDOM_LABEL_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
NON_ROUTE_REGRESSION_CLEAN
ADVERSARIAL_FROZEN_EVAL_FAILS
TRAIN_LEAKAGE_DETECTED
PRODUCTION_API_NOT_READY
```

## Claim Boundary

049 can support:

```text
bounded adversarial frozen eval scale evidence for input-conditioned inference behavior
```

049 cannot support:

```text
production default training
public beta promotion
production API readiness
full VRAXION
language grounding
consciousness
```

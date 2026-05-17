# STABLE_LOOP_PHASE_LOCK_047_LONG_RUN_BEHAVIOR_STABILITY Contract

## Summary

046 showed concrete inference scaleout positive. 047 tests whether that behavior
remains stable across checkpoint time.

This is not a new capability probe. It does not add task families as the main
goal. It does not enable production default training, promote public beta, or
claim full VRAXION, language grounding, or consciousness.

## Core Question

```text
Does the 046 concrete train -> inference behavior remain true across checkpoint
time, or does later drift / collapse / overfit return?
```

## Checkpoints

```text
checkpoint_000
checkpoint_010
checkpoint_025
checkpoint_050
checkpoint_100
checkpoint_200
```

## Required Arms

```text
ROUTE_GRAMMAR_LONG_RUN_TRAIN_AND_INFER
ROUTE_GRAMMAR_LONG_RUN_ROLLBACK_GATED
ROUTE_GRAMMAR_LONG_RUN_COST_CAPPED
CONCRETE_INFERENCE_046_REFERENCE
NO_ROUTE_GRAMMAR_LONG_RUN_BASELINE
ROUTE_GRAMMAR_SHUFFLED_LABELS
ROUTE_GRAMMAR_SHUFFLED_INPUT_ORDER
ALWAYS_SPACE_CONTROL
ALWAYS_MAJORITY_CONTROL
COPY_LAST_TOKEN_CONTROL
RANDOM_LABEL_CONTROL
RANDOM_PHASE_RULE_CONTROL
NON_ROUTE_REGRESSION_CONTROL
```

## Metrics At Every Checkpoint

```text
heldout_exact_accuracy
ood_exact_accuracy
family_min_accuracy
per_family_accuracy
unique_output_count
unique_output_rate
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
behavior_drift_score
output_distribution_drift
checkpoint_save_load_pass
rollback_success
```

## Stability Gate

Positive only if the 046 behavior gate remains stable at every measured
checkpoint:

```text
heldout_exact_accuracy >= 0.90
ood_exact_accuracy >= 0.85
family_min_accuracy >= 0.80
unique_output_count >= expected_output_class_count
top_output_rate <= 0.35
majority_output_rate <= 0.35
space_only_rate <= 0.01
empty_output_rate <= 0.01
output_entropy does not collapse by > 20%
behavior_drift_score within allowed band
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
checkpoint_metrics.jsonl
long_run_behavior_metrics.jsonl
output_distribution_by_checkpoint.jsonl
collapse_metrics_by_checkpoint.jsonl
per_family_metrics_by_checkpoint.jsonl
inference_samples_by_checkpoint.jsonl
bad_cases.jsonl
summary.json
report.md
contract_snapshot.md
job_progress/*.jsonl
```

## Verdicts

```text
LONG_RUN_BEHAVIOR_STABILITY_POSITIVE
INPUT_CONDITIONING_STABLE_OVER_TIME
OUTPUT_ENTROPY_STABLE
STATIC_OUTPUT_COLLAPSE_DOES_NOT_RETURN
MAJORITY_SHORTCUT_DOES_NOT_RETURN
COPY_SHORTCUT_DOES_NOT_RETURN
OOD_RETENTION_STABLE
NON_ROUTE_REGRESSION_CLEAN
BEHAVIOR_DRIFT_ACCEPTABLE
CHECKPOINT_SAVE_LOAD_STABLE
LONG_RUN_COLLAPSE_DETECTED
LONG_RUN_OVERFIT_DETECTED
PRODUCTION_API_NOT_READY
```

## Claim Boundary

047 can support:

```text
bounded long-run checkpoint-time stability for concrete inference behavior
```

047 cannot support:

```text
production default training
public beta promotion
production API readiness
full VRAXION
language grounding
consciousness
```

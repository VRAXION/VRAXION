# STABLE_LOOP_PHASE_LOCK_044_FINAL_TRAINING_LAUNCH Contract

## Summary

043 passed final-training preflight. 044 is the first bounded final-training
candidate run. It still does not enable production default training, promote
public beta, or claim production API readiness.

## Required Safeguards

```text
final_training_launched = true only after queue/report artifacts are initialized
production_default_training_enabled = false
public_beta_promoted = false
production_api_ready = false
rollback checkpoint written before training mutation loop
heartbeat progress every 30s
do not commit target/ outputs
```

## Required Arms

```text
FINAL_TRAINING_BASELINE_REFERENCE
FINAL_TRAINING_CHECKPOINT_SAVE_LOAD_PROOF
FINAL_TRAINING_ROUTE_GRAMMAR_ENABLED
FINAL_TRAINING_ROUTE_GRAMMAR_ROLLBACK_GATED
FINAL_TRAINING_LONG_HORIZON_MONITOR
FINAL_TRAINING_MULTI_SEED_MONITOR
FINAL_TRAINING_HARD_REGRESSION_MONITOR
FINAL_TRAINING_ARTIFACT_SAFETY_MONITOR
FINAL_TRAINING_COST_ENVELOPE_MONITOR
FINAL_TRAINING_OUTPUT_DISTRIBUTION_MONITOR
FINAL_TRAINING_NON_ROUTE_DRIFT_MONITOR
RANDOM_ROUTE_GRAMMAR_CONTROL
RANDOM_PHASE_RULE_CONTROL
```

## Required Metrics

```text
final_training_launched
final_training_completed
checkpoint_before_hash
checkpoint_after_hash
rollback_checkpoint_hash
checkpoint_save_load_pass
best_checkpoint_score
heldout_score
ood_score
context_carry_score
artifact_safety_score
hard_regression_pass_rate
output_distribution_drift
non_route_regression_delta
rollback_available
rollback_success
compute_overhead_ratio
memory_overhead_ratio
route_order_accuracy
missing_successor_count
family_min_accuracy
random_control_accuracy
production_default_training_enabled
```

## Required Outputs

```text
queue.json
progress.jsonl
metrics.jsonl
final_training_launch_metrics.jsonl
checkpoint_metrics.jsonl
training_progress_metrics.jsonl
hard_regression_metrics.jsonl
artifact_safety_metrics.jsonl
rollback_rehearsal_metrics.jsonl
launch_gate_metrics.jsonl
regression_metrics.jsonl
rollback_metrics.jsonl
control_metrics.jsonl
summary.json
report.md
contract_snapshot.md
examples_sample.jsonl
job_progress/*.jsonl
```

## Verdicts

```text
FINAL_TRAINING_LAUNCH_STARTED
FINAL_TRAINING_COMPLETED
FINAL_TRAINING_CHECKPOINT_WRITTEN
FINAL_TRAINING_IMPROVES_HELDOUT
FINAL_TRAINING_IMPROVES_OOD
FINAL_TRAINING_IMPROVES_CONTEXT_CARRY
HARD_REGRESSION_CORPUS_PASSES
ARTIFACT_SAFETY_PASSES
OUTPUT_DISTRIBUTION_DRIFT_ACCEPTABLE
ROLLBACK_CHECKPOINT_READY
RANDOM_ROUTE_GRAMMAR_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
PRODUCTION_DEFAULT_STILL_DISABLED
FINAL_TRAINING_LAUNCH_POSITIVE
PRODUCTION_API_NOT_READY
```

## Positive Gate

`FINAL_TRAINING_LAUNCH_POSITIVE` requires:

```text
final_training_launched = true
final_training_completed = true
checkpoint_after_hash != checkpoint_before_hash
checkpoint_save_load_pass = true
heldout/OOD/context-carry improve or remain above gate
hard_regression_pass_rate >= 0.95
artifact_safety_score does not regress
output_distribution_drift within allowed band
non_route_regression_delta >= -0.02
rollback checkpoint exists
rollback_success = true
random controls fail
production_default_training_enabled = false
public_beta_promoted = false
production_api_ready = false
```

## Claim Boundary

044 can support:

```text
first bounded final-training candidate completed under route-grammar launch gates
```

044 cannot support:

```text
production default training enablement
public beta promotion
production API readiness
full VRAXION
language grounding
consciousness
biological/FlyWire equivalence
physical quantum behavior
```

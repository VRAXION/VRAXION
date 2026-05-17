# STABLE_LOOP_PHASE_LOCK_043_FINAL_TRAINING_PREFLIGHT Contract

## Summary

042 showed the full-model-style route-grammar operator bridge is positive in a
bounded runner-local smoke. 043 is the final-training preflight. It does not
start final training. It decides whether the next milestone may be a final
training launch candidate.

## Required Arms

```text
BASE_CHECKPOINT_REFERENCE
BASE_CHECKPOINT_SAVE_LOAD_ROUNDTRIP
ROUTE_GRAMMAR_BRIDGE_042_REFERENCE
ROUTE_GRAMMAR_PREFLIGHT_LONG_HORIZON
ROUTE_GRAMMAR_PREFLIGHT_MULTI_SEED
ROUTE_GRAMMAR_PREFLIGHT_HARD_REGRESSION
ROUTE_GRAMMAR_PREFLIGHT_ARTIFACT_SAFETY
ROUTE_GRAMMAR_PREFLIGHT_ROLLBACK_REHEARSAL
ROUTE_GRAMMAR_PREFLIGHT_COST_ENVELOPE
ROUTE_GRAMMAR_PREFLIGHT_NON_ROUTE_DRIFT
ROUTE_GRAMMAR_PREFLIGHT_OUTPUT_DISTRIBUTION
RANDOM_ROUTE_GRAMMAR_CONTROL
RANDOM_PHASE_RULE_CONTROL
```

## Required Metrics

```text
checkpoint_save_load_pass
checkpoint_hash_stable
best_checkpoint_score
heldout_score
ood_score
context_carry_score
artifact_safety_score
output_distribution_drift
behavior_drift_score
non_route_regression_delta
rollback_success
rollback_time_steps
compute_overhead_ratio
memory_overhead_ratio
cost_envelope_pass
hard_regression_pass_rate
final_training_preflight_gate_pass
production_default_training_enabled
final_training_launched
```

## Required Outputs

```text
queue.json
progress.jsonl
metrics.jsonl
final_training_preflight_metrics.jsonl
checkpoint_handoff_metrics.jsonl
long_horizon_metrics.jsonl
hard_regression_metrics.jsonl
artifact_safety_metrics.jsonl
rollback_rehearsal_metrics.jsonl
preflight_gate_metrics.jsonl
regression_metrics.jsonl
control_metrics.jsonl
summary.json
report.md
contract_snapshot.md
examples_sample.jsonl
job_progress/*.jsonl
```

## Verdicts

```text
FINAL_TRAINING_PREFLIGHT_POSITIVE
CHECKPOINT_SAVE_LOAD_READY
ROUTE_GRAMMAR_BRIDGE_STABLE_LONG_HORIZON
HARD_REGRESSION_CORPUS_PASSES
ARTIFACT_SAFETY_PASSES
ROLLBACK_REHEARSAL_PASSES
COST_ENVELOPE_ACCEPTABLE
NON_ROUTE_DRIFT_CLEAN
OUTPUT_DISTRIBUTION_DRIFT_ACCEPTABLE
FINAL_TRAINING_READY_TO_LAUNCH
FINAL_TRAINING_NOT_READY
RANDOM_ROUTE_GRAMMAR_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
PRODUCTION_API_NOT_READY
```

## Positive Gate

`FINAL_TRAINING_PREFLIGHT_POSITIVE` requires:

```text
checkpoint save/load passes
checkpoint hash stable
route grammar bridge remains positive
long-horizon and multi-seed arms pass
hard regression corpus passes
artifact safety does not regress
non-route regression delta >= -0.02
output distribution drift within allowed band
rollback rehearsal passes
compute overhead <= 1.15
memory overhead <= 1.10
random controls fail
production_default_training_enabled = false
final_training_launched = false
```

`FINAL_TRAINING_READY_TO_LAUNCH` may be emitted only when the preflight gate
passes. It is still not the launch itself.

## Claim Boundary

043 can support:

```text
final training launch is preflight-ready under the tested runner-local gate
```

043 cannot support:

```text
final training has launched
production default training enablement
public beta promotion
production API readiness
full VRAXION
language grounding
consciousness
biological/FlyWire equivalence
physical quantum behavior
```

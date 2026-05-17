# STABLE_LOOP_PHASE_LOCK_036_EXTERNAL_ROUTE_GRAMMAR_CONSUMER Contract

## Summary

035 showed the experimental route-grammar API survives a non-toy hardening
suite. 036 tests whether less controlled external consumers can call that API
deterministically and safely.

This is API contract hardening, not a new route mechanism.

No production readiness, full VRAXION, language grounding, consciousness,
biological, FlyWire, or physical quantum claim.

## API Under Test

```text
instnct_core::experimental_route_grammar
```

## Required Arms

```text
HAND_PIPELINE_REFERENCE
EXTERNAL_CONSUMER_SMOKE
MULTI_CALL_API_STATE_ISOLATION
INVALID_INPUT_FUZZ
DETERMINISM_REPLAY
SERDE_ROUNDTRIP_TASKS
NO_GLOBAL_STATE_LEAK
CONCURRENT_CALLS_SAFE
REGRESSION_REACHABLE_SEED_BUG
NO_GRAMMAR_API_CONTROL
RANDOM_ROUTE_TASK_CONTROL
RANDOM_PHASE_RULE_CONTROL
```

## External Consumer Contract Checks

```text
external_consumer_smoke_pass
multi_call_state_isolation_pass
invalid_input_fuzz_pass
determinism_replay_pass
serde_roundtrip_pass
no_global_state_leak_pass
concurrent_calls_safe_pass
reachable_seed_bug_regression_pass
```

## Metrics

```text
sufficient_tick_final_accuracy
long_path_accuracy
family_min_accuracy
wrong_if_delivered_rate
route_order_accuracy
retained_successor_accuracy
missing_successor_count
duplicate_successor_count
branch_count
cycle_count
source_to_target_reachability
gate_shuffle_collapse
same_target_counterfactual_accuracy
random_control_accuracy
external_consumer_contract_pass
forbidden_private_field_leak
nonlocal_edge_count
direct_output_leak_rate
```

## Verdicts

```text
EXTERNAL_ROUTE_GRAMMAR_CONSUMER_POSITIVE
MULTI_CALL_API_STATE_ISOLATION_WORKS
INVALID_INPUT_FUZZ_WORKS
DETERMINISM_REPLAY_WORKS
SERDE_ROUNDTRIP_TASKS_WORK
NO_GLOBAL_STATE_LEAK_WORKS
CONCURRENT_CALLS_SAFE
REACHABLE_SEED_BUG_REGRESSION_WORKS
NO_GRAMMAR_API_CONTROL_FAILS
RANDOM_ROUTE_TASK_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
PRODUCTION_API_NOT_READY
```

## Decision Gate

```text
EXTERNAL_ROUTE_GRAMMAR_CONSUMER_POSITIVE if a non-hand consumer arm reaches:

sufficient_tick_final_accuracy >= 0.95
long_path_accuracy >= 0.95
family_min_accuracy >= 0.85
wrong_if_delivered_rate <= 0.10
route_order_accuracy >= 0.90
retained_successor_accuracy >= 0.90
missing_successor_count <= 0.05
branch/cycle near zero
same_target_counterfactual_accuracy >= 0.85
gate_shuffle_collapse >= 0.50
random controls fail
external consumer contract checks pass
wall/private/nonlocal/direct leaks = 0
```

## Required Outputs

```text
queue.json
progress.jsonl
metrics.jsonl
external_consumer_metrics.jsonl
external_consumer_contract_metrics.jsonl
api_metrics.jsonl
task_family_metrics.jsonl
loop_metrics.jsonl
grammar_metrics.jsonl
delivery_metrics.jsonl
routing_metrics.jsonl
family_metrics.jsonl
counterfactual_metrics.jsonl
control_metrics.jsonl
locality_audit.jsonl
mechanism_ranking.json
summary.json
report.md
contract_snapshot.md
examples_sample.jsonl
job_progress/*.jsonl
```

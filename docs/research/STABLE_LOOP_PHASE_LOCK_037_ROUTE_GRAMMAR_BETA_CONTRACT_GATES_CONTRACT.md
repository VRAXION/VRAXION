# STABLE_LOOP_PHASE_LOCK_037_ROUTE_GRAMMAR_BETA_CONTRACT_GATES Contract

## Summary

036 showed that external consumers can use the experimental route-grammar API
deterministically and safely in the current research scope.

037 is a release-gate probe:

```text
Should this remain experimental, or is it ready to be considered for a public beta contract?
```

This probe evaluates beta-contract gates only. It does not promote the API.

## API Under Test

```text
instnct_core::experimental_route_grammar
```

## Required Arms

```text
HAND_PIPELINE_REFERENCE
EXTERNAL_CONSUMER_REFERENCE_036
API_DOCUMENTATION_COMPLETENESS
BACKWARDS_COMPAT_TYPE_CONTRACT
DETERMINISTIC_REPLAY_CORPUS
INVALID_INPUT_FUZZ_EXPANDED
CONCURRENCY_STRESS_EXPANDED
EXTERNAL_CONSUMER_EXAMPLES
REGRESSION_CORPUS_ROUTE_GRAMMAR
NO_GRAMMAR_API_CONTROL
RANDOM_ROUTE_TASK_CONTROL
RANDOM_PHASE_RULE_CONTROL
```

## Gate Metrics

```text
api_documentation_completeness_pass
backwards_compatible_type_contract_pass
deterministic_replay_corpus_pass
invalid_input_fuzz_expanded_pass
concurrency_stress_expanded_pass
external_consumer_examples_pass
regression_corpus_route_grammar_pass
experimental_to_beta_boundary_documented
public_beta_promoted
beta_contract_gate_pass
```

## Verdicts

```text
ROUTE_GRAMMAR_BETA_CONTRACT_GATES_POSITIVE
API_DOCUMENTATION_COMPLETENESS_PASS
BACKWARDS_COMPAT_TYPE_CONTRACT_PASS
DETERMINISTIC_REPLAY_CORPUS_PASS
INVALID_INPUT_FUZZ_EXPANDED_PASS
CONCURRENCY_STRESS_EXPANDED_PASS
EXTERNAL_CONSUMER_EXAMPLES_PASS
REGRESSION_CORPUS_ROUTE_GRAMMAR_PASS
NO_GRAMMAR_API_CONTROL_FAILS
RANDOM_ROUTE_TASK_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
PRODUCTION_API_NOT_READY
```

## Decision Gate

```text
ROUTE_GRAMMAR_BETA_CONTRACT_GATES_POSITIVE if:

sufficient_tick_final_accuracy >= 0.95
long_path_accuracy >= 0.95
family_min_accuracy >= 0.85
wrong_if_delivered_rate <= 0.10
route_order_accuracy >= 0.90
retained_successor_accuracy >= 0.90
missing_successor_count <= 0.05
same_target_counterfactual_accuracy >= 0.85
gate_shuffle_collapse >= 0.50
random controls fail
all beta gate metrics pass
public_beta_promoted = false
```

## Required Outputs

```text
queue.json
progress.jsonl
metrics.jsonl
beta_contract_metrics.jsonl
beta_contract_gate_metrics.jsonl
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

## Claim Boundary

037 can support that the experimental API has enough gate evidence to consider a
public beta contract later. It cannot itself claim production readiness, full
VRAXION, language grounding, consciousness, biological equivalence, FlyWire
wiring, or physical quantum behavior.

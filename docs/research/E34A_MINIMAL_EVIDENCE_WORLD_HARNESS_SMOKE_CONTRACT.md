# E34A Minimal Evidence World Harness Smoke Contract

## Purpose

`E34A_MINIMAL_EVIDENCE_WORLD_HARNESS_SMOKE` tests a minimal active-evidence world:

```text
limited initial evidence
-> choose evidence acquisition actions
-> update Flow Field candidate state
-> answer only when the hidden cause is evidence-backed
```

The point is to check whether a gradientless mutation/rollback policy can reach for useful evidence instead of passively consuming a preloaded stream.

## Boundary

E34A is a deterministic symbolic/numeric smoke probe. It is not a raw-language reasoning proof, chatbot benchmark, AGI claim, consciousness claim, deployed-model claim, or model-scale claim.

## World

Each episode contains:

```text
hidden cause/state
binary evidence features
one verified initial observation
one untrusted rumor observation
INSPECT(feature) actions
ANSWER(cause) action
trace ledger of actions and candidate-state reductions
```

The system must use visible inspected evidence, not hidden oracle truth, before answering.

## Systems

```text
learned_mutation_policy
forced_initial_answer
random_action_control
ask_all_until_unique
oracle_info_gain_reference
```

`oracle_info_gain_reference` is a ceiling/control, not a valid learned system.

## Required Metrics

```text
closed_loop_success
answer_correct
trace_exact
wrong_confident_answer
false_ask
redundant_actions
average_steps_to_answer
first_useful_evidence_action
accepted/rejected/rollback mutation counts
parameter diff/hash
deterministic replay
checker failure count
```

## Decisions

```text
e34a_active_evidence_world_confirmed
e34a_active_policy_no_advantage
e34a_mutation_policy_failed
e34a_artifact_invalid
```

Positive requires:

```text
learned closed-loop success >= 0.98
learned trace exact >= 0.98
wrong confident answer <= 0.01
learned average steps < ask-all baseline
random control substantially worse
forced initial answer fails as expected
accepted and rejected mutations both present
deterministic replay passes
checker failure count = 0
```

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
policy_initial_state.json
policy_final_state.json
parameter_diff.json
mutation_history.jsonl
row_level_results.jsonl
system_results.json
aggregate_metrics.json
deterministic_replay.json
resource_usage_report.json
decision.json
summary.json
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
report.md
```

The run must write progress and heartbeat artifacts during execution. No black-box long run is allowed.

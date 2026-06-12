# E34B Active Evidence World With Noisy Text Observations Contract

## Purpose

`E34B_ACTIVE_EVIDENCE_WORLD_WITH_NOISY_TEXT_OBSERVATIONS` extends E34A:

```text
clean binary evidence feature
-> short noisy text observation about that feature
```

The probe asks whether the active evidence-seeking loop still works when the
system must parse visible noisy text observations before updating the Flow Field
candidate state.

## Boundary

E34B is a deterministic controlled text-mediated evidence probe. It is not a
chatbot benchmark, raw language understanding proof, AGI claim, consciousness
claim, deployed-model claim, or model-scale claim.

## World

Each episode contains:

```text
hidden cause/state
binary evidence features
one verified initial text observation
one untrusted rumor text observation
INSPECT_TEXT(feature) actions
ANSWER(cause) action
trace ledger of text observations, parsed values, and candidate-state changes
```

The visible observations include paraphrase, weak source language, stale hints,
OOD aliases, and adversarial contrast clauses such as:

```text
rumor says marker 3 is absent, but verified reading says marker 3 is present
```

The primary system must use the visible observation text. Hidden truth is only
used by the world simulator and oracle/reference controls.

## Systems

```text
learned_mutation_text_policy
forced_initial_text_answer
random_text_action_control
ask_all_text_until_unique
keyword_shortcut_text_control
oracle_text_policy_reference
```

`oracle_text_policy_reference` is a ceiling/control, not a valid learned system.
`keyword_shortcut_text_control` is a danger control: it should fail on adversarial
contrast text if the task is not shortcutable.

## Required Metrics

```text
closed_loop_success
answer_correct
trace_exact
text_extraction_accuracy
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
e34b_noisy_text_active_evidence_confirmed
e34b_text_extraction_bottleneck_detected
e34b_active_policy_no_efficiency_advantage
e34b_noisy_text_active_evidence_failed
e34b_artifact_invalid
```

Positive requires:

```text
learned closed-loop success >= 0.95
learned trace exact >= 0.95
text extraction accuracy >= 0.96
wrong confident answer <= 0.03
learned average steps < ask-all baseline
random control substantially worse
forced initial answer fails as expected
keyword shortcut adversarial control fails relative to learned policy
accepted and rejected mutations both present
deterministic replay passes
checker failure count = 0
```

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
text_observation_report.json
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

The run must write progress and heartbeat artifacts during execution. No
black-box long run is allowed.

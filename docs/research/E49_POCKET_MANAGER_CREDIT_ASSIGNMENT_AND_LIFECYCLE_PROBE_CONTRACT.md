# E49 Pocket Manager Credit Assignment And Lifecycle Probe Contract

## Purpose

E49 tests the Pocket Manager / Pocket Ecology Runtime layer.

Core question:

```text
Can call-level PocketEvaluationEvents, delayed outcome, counterfactual
ablation, and harm/trace feedback correctly drive pocket lifecycle decisions?
```

This probe follows the modular architecture lock:

```text
Pocket -> Proposal Field -> Agency + Trace -> Commit/Reject/Defer
```

## Boundary

This is a controlled symbolic/numeric lifecycle probe. It does not test raw
language reasoning, deployed assistant behavior, AGI, consciousness, or
model-scale behavior.

## Systems

```text
no_manager_random_reuse
final_answer_only_score
immediate_score_only
call_count_popularity_score
full_event_credit_manager
oracle_lifecycle_reference
```

## Pocket Lifecycle States

```text
candidate
active
core
specialist
quarantine
deprecated
banned
```

## Required Event

Every pocket call must emit a `PocketEvaluationEvent` with at least:

```text
pocket_id
pocket_version
call_id
cycle_id
route_id
caller_node
edge_id_if_any
input_footprint
output_proposal_hash
proposal_type
proposal_target
proposal_confidence
agency_decision
commit_id_if_any
reject_reason_if_any
defer_reason_if_any
trace_ref
ground_ref
evidence_refs
cost
immediate_outcome
delayed_outcome
counterfactual_without_pocket
downstream_harm
failure_mode
```

## Required Credit Signals

The full manager may use:

```text
immediate outcome
delayed outcome
counterfactual_without_pocket
downstream harm
trace mismatch
OOD/adversarial survival
reuse
cost
novelty
```

Controls must isolate weaker credit policies:

```text
final answer only
immediate score only
call-count popularity only
random/no manager
```

## Metrics

```text
lifecycle_accuracy
weighted_lifecycle_credit
promote_correct_core
quarantine_dangerous_specialist
avoid_credit_hijack
delayed_harm_detection
cost_adjusted_utility
OOD_survival
adversarial_survival
route_quality_delta
wrong_commit_delta
prune_false_positive
```

## Decisions

Allowed decisions:

```text
e49_pocket_manager_credit_lifecycle_positive
e49_final_answer_only_sufficient
e49_immediate_score_sufficient
e49_call_count_popularity_sufficient
e49_counterfactual_credit_required
e49_invalid_artifact_detected
```

Positive requires:

```text
full_event_credit_manager lifecycle_accuracy >= 0.90
avoid_credit_hijack = 1.0
delayed_harm_detection = 1.0
quarantine_dangerous_specialist >= 0.95
wrong_commit_delta <= 0.05
prune_false_positive <= 0.05
final_answer_only_score does not also pass the credit-hijack gate
deterministic replay passes
checker failure_count = 0
```

## Required Artifacts

```text
backend_manifest.json
pocket_event_schema.json
pocket_evaluation_events.jsonl
pocket_feature_report.json
credit_assignment_report.json
lifecycle_decision_report.json
counterfactual_ablation_report.json
delayed_credit_report.json
manager_mutation_history.jsonl
system_results.json
lifecycle_decision_rows.jsonl
aggregate_metrics.json
deterministic_replay.json
decision.json
summary.json
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
results_table.md
report.md
```

## Sample Pack

The sample pack must live under:

```text
docs/research/artifact_samples/e49_pocket_manager_credit_assignment_and_lifecycle_probe/
```

## Hard Requirements

```text
no gradient descent
no optimizer/backprop
row-level/event-level eval
real mutation/rollback for the full event manager feature selection
accepted/rejected mutation evidence
rollback count equals rejected count
target checker passes with failure_count = 0
sample-only checker passes
```

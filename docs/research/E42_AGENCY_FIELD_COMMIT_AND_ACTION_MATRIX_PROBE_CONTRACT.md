# E42 Agency Field Commit And Action Matrix Probe Contract

## Summary

E42 is the first minimal test for the proposed central decision layer:

```text
Agency Field = central action / commit decision field
Action Matrix = ALU / Logic Atom genome implementation
```

It tests whether a multi-input Logic Atom genome can decide actions from
mechanical views without turning into a monolithic black box.

When multiple atoms fire, the Agency Field uses specificity-first arbitration:
the atom with more matched conditions wins; action priority only breaks ties.
This prevents one broad action atom from suppressing narrower decision lanes.
Positive action lanes require negative guards too: e.g. a COMMIT/ANSWER atom
must encode not only the requested action but also visible proposal-valid,
trace-valid, not-unresolved, and no-ground-contradiction guards.
The mutation fitness repairs trace failures as well as action failures, so an
action-only shortcut is not enough. Decoy/noise conditions are allowed as an
audit surface but penalized in the genome score; a stable Agency Field should
not depend on random decoy bits.

## Core Question

Can an Agency Field using Flow, Ground, Proposal, Trace, and Cost views make
better COMMIT/REJECT/DEFER/ASK/CALL/ANSWER decisions than:

- direct pocket action,
- simple priority Arbiter,
- Agency Field without Ground,
- random action,
- invalid monolith/oracle controls?

## Systems

```text
oracle_agency_reference_only
direct_pocket_action_baseline
simple_priority_arbiter
agency_field_without_ground
agency_field_full_views_grow_shrink
fixed_direct_decision_lanes_reference
full_monolith_oracle_control
random_action_control
```

## Input Views

The Agency Field sees mechanical views only:

```text
Flow view:
  unresolved, conflict, answer_ready

Ground view:
  stable, contradiction

Proposal view:
  valid, wants_commit, wants_answer, wants_call

Trace view:
  valid, support

Cost view:
  cheap, pocket_health, evidence_available

Noise view:
  decoy bits
```

No semantic lanes such as truth/confidence/memory-result labels are used.

## Output Actions

```text
COMMIT
REJECT
DEFER
ASK
CALL
ANSWER
```

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
agency_field_report.json
decision_lane_report.json
system_results.json
mutation_report.json
row_level_results.jsonl
aggregate_metrics.json
deterministic_replay.json
decision.json
summary.json
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
report.md
```

Sample pack:

```text
docs/research/artifact_samples/e42_agency_field_commit_and_action_matrix_probe/
```

## Metrics

```text
action_accuracy
wrong_commit_rate
missed_commit_rate
correct_defer_rate
correct_ask_rate
conflict_resolution_rate
trace_exact_rate
unnecessary_call_rate
accepted/rejected/rollback mutation counts
parameter diff/hash
deterministic replay
checker failure_count
```

## Decisions

```text
e42_agency_field_positive
e42_simple_arbiter_sufficient
e42_ground_trace_not_needed
e42_agency_field_growth_failed
e42_monolith_control_required
e42_invalid_artifact_detected
```

Positive requires:

```text
agency_field_full_views_grow_shrink action_accuracy >= 0.95
wrong_commit_rate <= 0.03
missed_commit_rate <= 0.03
correct_defer_rate >= 0.95
correct_ask_rate >= 0.95
conflict_resolution_rate >= 0.95
trace_exact_rate >= 0.90
simple/direct controls remain below 0.85 action accuracy
without-ground ablation does not also pass
checker failure_count = 0
sample-only checker passes
deterministic replay passes
```

## Boundary

E42 is a controlled symbolic/numeric Agency Field proxy. It does not prove raw
language reasoning, AGI, consciousness, deployed-model behavior, or model-scale
behavior.

# E93 Agency/Guard Commit-Safety Expansion Contract

## Purpose

Expand the Operator Library with commit-safety Guards around the Proposal Field
-> Agency -> Flow/Ground boundary.

This probe does not add a new architecture. It validates scoped Operators that
help decide whether visible proposals should be committed, rejected, deferred,
or used to ask for more evidence.

Boundary:

```text
controlled Proposal Field commit-safety proxy
not open-domain model behavior
direct Flow write remains disallowed
```

## Required Operator Candidates

```text
proposal_collision_guard
ground_conflict_guard
evidence_recency_guard
trace_dependency_coverage_guard
cycle_freshness_guard
local_scope_exit_t_stab
agency_commit_quorum_guard
safe_commit_action_scribe
```

## Required Controls

```text
first_proposal_committer
majority_without_trace_committer
stale_cycle_committer
ground_overwrite_committer
local_scope_leak_committer
always_reject_control
quorum_guard_clone
```

## Positive Decision

```text
decision = e93_agency_guard_commit_safety_expansion_confirmed
```

Requires:

```text
validation_commit_safety_success_min = 1.0
adversarial_commit_safety_success_min = 1.0
validation_trace_validity_min = 1.0
adversarial_wrong_commit_max = 0.0
adversarial_false_commit_max = 0.0
validation_missed_commit_max = 0.0
checker failure_count = 0
sample-only checker failure_count = 0
```

## Commands

```text
python private_probe_runner_removed
python private_probe_runner_removed --out target/pilot_wave/e93_agency_guard_commit_safety_expansion --write-summary
python private_probe_runner_removed --sample-only archived_public_artifact_sample_removed --write-summary
```

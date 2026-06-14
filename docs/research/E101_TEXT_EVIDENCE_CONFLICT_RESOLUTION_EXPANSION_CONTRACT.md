# E101 Text Evidence Conflict Resolution Expansion Contract

## Purpose

E101 expands the Operator Library with controlled text-evidence conflict
resolution skills. These Operators choose among multiple already-extracted text
evidence spans by detecting conflicts, applying source priority, preferring the
latest verified evidence when valid, asking for missing dependencies, and
blocking unsafe commits when contradiction remains unresolved.

This is not natural-language reasoning or open-domain fact checking.

## Required Boundary

```text
controlled text evidence conflict resolution proxy
not natural-language reasoning
not open-domain fact checking
not direct span commit
not model-scale claim
```

## Stable Candidate Targets

```text
evidence_conflict_detector_lens
source_priority_resolver_lens
temporal_latest_span_t_stab
multi_span_consistency_guard
contradiction_to_defer_guard
missing_dependency_question_scribe
clarified_query_focus_lens
conflict_resolved_proposal_scribe
```

## Controls

```text
first_span_wins_control
latest_without_source_control
keyword_strength_picker
contradiction_ignoring_committer
always_ask_control
source_rank_blind_control
stale_span_committer
conflict_detector_echo_clone
```

## Required Artifacts

```text
run_manifest.json
operator_library_manifest.json
task_generation_report.json
progress.jsonl
partial_aggregate_snapshot.json
seed_results.json
aggregate_metrics.json
selection_frequency_report.json
counterfactual_report.json
operator_lifecycle_report.json
mutation_summary.json
deterministic_replay.json
decision.json
summary.json
report.md
row_level_samples.jsonl
operator_evolution_history.jsonl
```

Sample pack:

```text
docs/research/artifact_samples/e101_text_evidence_conflict_resolution_expansion/
```

## Metrics

```text
resolution_success
conflict_detection_validity
source_priority_validity
temporal_latest_validity
ask_question_validity
trace_resolution_validity
unsafe_conflict_commit_rate
false_ask_rate
stale_commit_rate
counterfactual resolution loss
accepted/rejected/rollback mutation counts
deterministic replay hash match
checker failure count
```

## Pass Requirements

```text
validation_resolution_success_min = 1.000000
adversarial_resolution_success_min = 1.000000
validation_conflict_detection_validity_min = 1.000000
validation_source_priority_validity_min = 1.000000
validation_temporal_latest_validity_min = 1.000000
validation_ask_question_validity_min = 1.000000
validation_trace_resolution_validity_min = 1.000000
adversarial_unsafe_conflict_commit_rate_max = 0.000000
adversarial_false_ask_rate_max = 0.000000
adversarial_stale_commit_rate_max = 0.000000
checker_failure_count = 0
sample_only_checker_failure_count = 0
deterministic replay passes
```

## Decisions

```text
e101_text_evidence_conflict_resolution_expansion_confirmed
e101_text_evidence_conflict_resolution_incomplete
e101_source_priority_failure
e101_temporal_latest_failure
e101_unresolved_conflict_commit_failure
e101_missing_dependency_ask_failure
e101_artifact_or_replay_failure
```

## Interpretation Rule

A positive result means the library gained scoped text-evidence conflict
resolution Operators.

It does not mean the system performs general textual fact checking.

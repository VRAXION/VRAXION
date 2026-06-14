# E101 Text Evidence Conflict Resolution Expansion Result

```text
decision = e101_text_evidence_conflict_resolution_expansion_confirmed
checker_failure_count = 0
sample_only_checker_failure_count = 0
```

Boundary:

```text
controlled text evidence conflict resolution proxy
not natural-language reasoning
not open-domain fact checking
```

## Key Metrics

```text
seeds = 16
validation_resolution_success_min = 1.000000
validation_resolution_success_mean = 1.000000
adversarial_resolution_success_min = 1.000000
adversarial_resolution_success_mean = 1.000000
validation_conflict_detection_validity_min = 1.000000
validation_source_priority_validity_min = 1.000000
validation_temporal_latest_validity_min = 1.000000
validation_ask_question_validity_min = 1.000000
validation_trace_resolution_validity_min = 1.000000
adversarial_unsafe_conflict_commit_rate_max = 0.000000
adversarial_false_ask_rate_max = 0.000000
adversarial_stale_commit_rate_max = 0.000000
accepted_mutations_total = 128
rejected_mutations_total = 512
rollback_count_total = 512
```

## Stable Operator Candidates

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

## Rejected Controls

```text
first_span_wins_control           -> Quarantine
latest_without_source_control     -> Quarantine
keyword_strength_picker           -> Quarantine
contradiction_ignoring_committer  -> Quarantine
always_ask_control                -> Deprecated
source_rank_blind_control         -> Quarantine
stale_span_committer              -> Quarantine
conflict_detector_echo_clone      -> Redundant
```

## Interpretation

E101 adds scoped text-evidence conflict resolution Operators. The skills detect
span conflicts, resolve by trusted source priority, stabilize latest verified
evidence, require multi-span consistency, defer unresolved contradictions, ask
for missing dependencies, clarify the active query focus, and render the final
conflict-resolved proposal.

This is not natural-language reasoning or open-domain fact checking.

## Artifacts

```text
target/pilot_wave/e101_text_evidence_conflict_resolution_expansion/
docs/research/artifact_samples/e101_text_evidence_conflict_resolution_expansion/
```

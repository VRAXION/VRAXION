# E105 Conversation Memory Summary And Context Compression Expansion Contract

## Purpose

E105 expands the Operator Library with controlled context-compression skills.
These Operators compress a long evidence-state trace into a smaller summary
while preserving required facts, unresolved dependencies, citation pointers,
stale-fact exclusions, and safe re-entry state.

This is not open-domain summarization.

## Required Boundary

```text
controlled context compression proxy
not open-domain summarization
not direct summary without evidence
not general dialogue memory
not model-scale claim
```

## Stable Candidate Targets

```text
context_window_pressure_lens
summary_relevance_span_selector_lens
required_fact_preservation_guard
unresolved_dependency_preservation_t_stab
citation_pointer_compaction_scribe
obsolete_turn_prune_guard
summary_drift_detection_guard
compressed_context_reentry_scribe
```

## Controls

```text
last_turn_only_summary_control
keyword_frequency_summary_control
drop_unresolved_dependency_control
drop_citation_pointer_control
stale_fact_summary_control
overcompressed_summary_control
hallucinated_bridge_summary_control
summary_guard_echo_clone
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
docs/research/artifact_samples/e105_conversation_memory_summary_and_context_compression_expansion/
```

## Metrics

```text
context_compression_success
required_fact_preservation
unresolved_dependency_preservation
citation_pointer_validity
stale_fact_exclusion
context_reentry_success
compression_ratio_validity
hallucinated_summary_rate
lost_dependency_rate
stale_summary_rate
cross_thread_bleed_rate
overcompression_rate
counterfactual context compression loss
accepted/rejected/rollback mutation counts
deterministic replay hash match
checker failure count
```

## Pass Requirements

```text
validation_context_compression_success_min = 1.000000
adversarial_context_compression_success_min = 1.000000
validation_required_fact_preservation_min = 1.000000
validation_unresolved_dependency_preservation_min = 1.000000
validation_citation_pointer_validity_min = 1.000000
validation_stale_fact_exclusion_min = 1.000000
validation_context_reentry_success_min = 1.000000
validation_compression_ratio_validity_min = 1.000000
adversarial_hallucinated_summary_rate_max = 0.000000
adversarial_lost_dependency_rate_max = 0.000000
adversarial_stale_summary_rate_max = 0.000000
adversarial_cross_thread_bleed_rate_max = 0.000000
adversarial_overcompression_rate_max = 0.000000
checker_failure_count = 0
sample_only_checker_failure_count = 0
deterministic replay passes
```

## Decisions

```text
e105_context_compression_summary_expansion_confirmed
e105_context_compression_summary_incomplete
e105_required_fact_loss_failure
e105_unresolved_dependency_loss_failure
e105_citation_pointer_loss_failure
e105_summary_drift_failure
e105_artifact_or_replay_failure
```

## Interpretation Rule

A positive result means the library gained scoped context-compression Operators
for controlled evidence-state traces.

It does not mean open-domain summarization works.

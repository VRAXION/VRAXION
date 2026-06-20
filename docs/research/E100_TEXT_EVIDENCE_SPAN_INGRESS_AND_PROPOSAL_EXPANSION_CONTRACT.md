# E100 Text Evidence Span Ingress And Proposal Expansion Contract

## Purpose

E100 expands the Operator Library with controlled text-ingress skills. These
Operators convert short, noisy text observations into mechanical evidence-span
proposals while preserving source attribution and blocking negation, quote,
irrelevant-span, and weak-claim failure modes.

This is not natural-language understanding. It is a controlled evidence-span
ingress proxy for the current Text Field -> Proposal Field path.

## Required Boundary

```text
controlled text evidence-span ingress proxy
not natural-language understanding
not open-domain reasoning
not direct text answer
not model-scale claim
```

## Stable Candidate Targets

```text
text_frame_boundary_lens
evidence_span_locator_lens
source_attribution_lens
negation_contrast_scope_guard
quote_scope_boundary_guard
irrelevant_span_filter_guard
weak_claim_uncertainty_t_stab
text_evidence_proposal_scribe
```

## Controls

```text
keyword_only_span_picker
first_number_committer
negation_blind_extractor
quote_bleed_extractor
source_blind_extractor
whole_sentence_dump_control
answer_from_text_without_evidence_control
span_locator_echo_clone
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
archived_public_artifact_sample_removed
```

## Metrics

```text
span_ingress_success
span_accuracy
source_attribution_validity
negation_contrast_safety
quote_scope_safety
irrelevant_filter_safety
uncertainty_safety
unsafe_text_commit_rate
irrelevant_commit_rate
counterfactual span ingress loss
accepted/rejected/rollback mutation counts
deterministic replay hash match
checker failure count
```

## Pass Requirements

```text
validation_span_ingress_success_min = 1.000000
adversarial_span_ingress_success_min = 1.000000
validation_span_accuracy_min = 1.000000
validation_source_attribution_validity_min = 1.000000
validation_negation_contrast_safety_min = 1.000000
validation_quote_scope_safety_min = 1.000000
validation_irrelevant_filter_safety_min = 1.000000
validation_uncertainty_safety_min = 1.000000
adversarial_unsafe_text_commit_rate_max = 0.000000
adversarial_irrelevant_commit_rate_max = 0.000000
checker_failure_count = 0
sample_only_checker_failure_count = 0
deterministic replay passes
```

## Decisions

```text
e100_text_evidence_span_ingress_expansion_confirmed
e100_text_evidence_span_ingress_incomplete
e100_negation_scope_failure
e100_quote_scope_failure
e100_source_attribution_failure
e100_irrelevant_span_failure
e100_uncertainty_safety_failure
e100_artifact_or_replay_failure
```

## Interpretation Rule

A positive result means the library gained scoped text-ingress Operators for
mechanical evidence-span proposals.

It does not mean the system understands natural language generally.

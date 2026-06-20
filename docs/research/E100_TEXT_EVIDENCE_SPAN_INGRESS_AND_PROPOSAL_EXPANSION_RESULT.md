# E100 Text Evidence Span Ingress And Proposal Expansion Result

```text
decision = e100_text_evidence_span_ingress_expansion_confirmed
checker_failure_count = 0
sample_only_checker_failure_count = 0
```

Boundary:

```text
controlled text evidence-span ingress proxy
not natural-language understanding
not direct text answer
```

## Key Metrics

```text
seeds = 16
validation_span_ingress_success_min = 1.000000
validation_span_ingress_success_mean = 1.000000
adversarial_span_ingress_success_min = 1.000000
adversarial_span_ingress_success_mean = 1.000000
validation_span_accuracy_min = 1.000000
validation_source_attribution_validity_min = 1.000000
validation_negation_contrast_safety_min = 1.000000
validation_quote_scope_safety_min = 1.000000
validation_irrelevant_filter_safety_min = 1.000000
validation_uncertainty_safety_min = 1.000000
adversarial_unsafe_text_commit_rate_max = 0.000000
adversarial_irrelevant_commit_rate_max = 0.000000
accepted_mutations_total = 128
rejected_mutations_total = 512
rollback_count_total = 512
```

## Stable Operator Candidates

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

## Rejected Controls

```text
keyword_only_span_picker                  -> Quarantine
first_number_committer                    -> Quarantine
negation_blind_extractor                  -> Quarantine
quote_bleed_extractor                     -> Quarantine
source_blind_extractor                    -> Quarantine
whole_sentence_dump_control               -> Quarantine
answer_from_text_without_evidence_control -> Quarantine
span_locator_echo_clone                   -> Redundant
```

## Interpretation

E100 adds scoped text-ingress Operators. The skills find valid text/frame
boundaries, locate minimal evidence spans, attach source attribution, block
negation/contrast and quote-scope failures, reject irrelevant spans, stabilize
weak claims into ASK/DEFER behavior, and render normalized evidence proposals
for Agency review.

This is not natural-language understanding. It is a controlled text
evidence-span ingress proxy.

## Artifacts

```text
target/pilot_wave/e100_text_evidence_span_ingress_and_proposal_expansion/
archived_public_artifact_sample_removed
```

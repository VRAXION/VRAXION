# E95 Active Evidence Request/Search Expansion Contract

## Purpose

Expand the Operator Library with Lens/Guard/Scribe/T-Stab skills that choose a
targeted evidence request/search action when the Flow state is unresolved.

Boundary:

```text
controlled active evidence selection proxy
not open-domain retrieval
not chatbot behavior
not hidden answer solving
```

## Required Operator Candidates

```text
missing_dependency_locator_lens
targeted_evidence_request_scribe
source_reliability_rank_guard
redundant_request_guard
search_budget_guard
adversarial_decoy_source_guard
retrieved_evidence_integrator_t_stab
answer_ready_after_evidence_scribe
```

## Required Controls

```text
broad_search_spammer
random_evidence_picker
rumor_source_committer
decoy_surface_match_committer
budgetless_search_runner
always_ask_all_control
request_locator_clone
```

## Positive Decision

```text
decision = e95_active_evidence_request_search_expansion_confirmed
```

Requires:

```text
validation_evidence_action_success_min = 1.0
adversarial_evidence_action_success_min = 1.0
validation_targeted_request_accuracy_min = 1.0
validation_answer_ready_accuracy_min = 1.0
adversarial_wrong_confident_max = 0.0
adversarial_false_search_max = 0.0
adversarial_over_budget_max = 0.0
checker failure_count = 0
sample-only checker failure_count = 0
```

## Commands

```text
python scripts/probes/run_e95_active_evidence_request_search_expansion.py
python scripts/probes/run_e95_active_evidence_request_search_expansion_check.py --out target/pilot_wave/e95_active_evidence_request_search_expansion --write-summary
python scripts/probes/run_e95_active_evidence_request_search_expansion_check.py --sample-only docs/research/artifact_samples/e95_active_evidence_request_search_expansion --write-summary
```

# E95 Active Evidence Request/Search Expansion Result

```text
decision = e95_active_evidence_request_search_expansion_confirmed
checker_failure_count = 0
sample_only_checker_failure_count = 0
```

Boundary:

```text
controlled active evidence selection proxy
not open-domain retrieval
not chatbot behavior
not hidden answer solving
```

## Key Metrics

```text
seeds = 16
validation_evidence_action_success_min = 1.000000
validation_evidence_action_success_mean = 1.000000
adversarial_evidence_action_success_min = 1.000000
adversarial_evidence_action_success_mean = 1.000000
validation_targeted_request_accuracy_min = 1.000000
validation_answer_ready_accuracy_min = 1.000000
adversarial_wrong_confident_max = 0.000000
adversarial_false_search_max = 0.000000
adversarial_over_budget_max = 0.000000
accepted_mutations_total = 128
rejected_mutations_total = 448
rollback_count_total = 448
```

## Stable Operator Candidates

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

## Rejected Controls

```text
broad_search_spammer            -> Quarantine
random_evidence_picker          -> Quarantine
rumor_source_committer          -> Quarantine
decoy_surface_match_committer   -> Quarantine
budgetless_search_runner        -> Quarantine
always_ask_all_control          -> Deprecated
request_locator_clone           -> Redundant
```

## Interpretation

E95 adds scoped active-evidence Operators. These Operators do not perform
open-domain retrieval. They decide which visible dependency should be requested,
which source class is safe, whether a request is redundant, whether search would
exceed budget, and when retrieved evidence makes the state answer-ready.

This extends E27/E34-style "do not answer without enough evidence" behavior into
a more explicit request/search skill layer.

## Artifacts

```text
target/pilot_wave/e95_active_evidence_request_search_expansion/
docs/research/artifact_samples/e95_active_evidence_request_search_expansion/
```

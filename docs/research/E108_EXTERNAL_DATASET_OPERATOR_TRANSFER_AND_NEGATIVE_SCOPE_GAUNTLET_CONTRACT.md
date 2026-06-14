# E108 External Dataset Operator Transfer And Negative Scope Gauntlet Contract

## Purpose

E108 tests whether E107 role-assigned Operators transfer to deterministic
external-style dataset families without negative transfer, wrong-scope calls,
unsafe commits, unsupported answers, or cost blowup.

This is not Core or TrueGolden promotion.

## Required Boundary

```text
external transfer/no-harm qualification
not Golden promotion
not Core promotion
not final training
not raw web benchmark claim
not open-domain reasoning claim
```

## Input

```text
docs/research/artifact_samples/e107_operator_library_e90_e106_survival_role_and_regression_gauntlet/operator_lifecycle_report.json
```

E107 roles are frozen for Phase 1.

## Dataset Zones

```text
external_structured_heldout
real_like_noisy_text
negative_scope_corpus
adversarial_scope_collision
external_progress_state
external_clarification_state
long_composition_no_harm
unrelated_open_domain_text
```

The generated dataset must write a manifest with source/license/hash/splits.

## Compared Policies

```text
no_operator_baseline
e107_frozen_role_policy
full_library_scan_control
popularity_selector_control
scope_blind_selector_control
e107_frozen_plus_tiny_adapter
oracle_reference_invalid_control
```

## Metrics

```text
external_validation_success
external_adversarial_success
negative_scope_success
activated_gain_mean
ablation_loss_mean
negative_transfer_rate
wrong_scope_call_rate
false_commit_rate
false_answer_rate
unsupported_answer_rate
no_harm_rate
cost_adjusted_utility_mean
role_stability
full_library_scan_negative_transfer_rate
external_transfer_candidate_count
scoped_transfer_candidate_count
internal_only_count
quarantine_count
deprecated_count
deterministic replay hash match
checker failure count
```

## Output Statuses

```text
ExternalTransferCandidate
ScopedTransferCandidate
InternalOnly
Quarantine
Deprecated
ChallengerReplaced
```

## Pass Requirements

```text
seed_count >= 16
case_count >= 10000
external_validation_success >= 0.980000
external_adversarial_success >= 0.980000
negative_scope_success = 1.000000
negative_transfer_rate = 0.000000
wrong_scope_call_rate = 0.000000
false_commit_rate = 0.000000
false_answer_rate = 0.000000
unsupported_answer_rate = 0.000000
no_harm_rate = 1.000000
activated_gain_mean > 0
ablation_loss_mean > 0
full_library_scan_negative_transfer_rate > 0
checker_failure_count = 0
sample_only_checker_failure_count = 0
deterministic replay passes
```

## Decisions

```text
e108_external_transfer_no_harm_positive
e108_scoped_specialist_transfer_positive
e108_internal_only_operator_detected
e108_negative_transfer_detected
e108_scope_overreach_detected
e108_challenger_replaces_current_operator
e108_external_transfer_no_harm_incomplete
```

## Interpretation Rule

A positive result qualifies Operators as external transfer candidates or scoped
transfer candidates. It does not promote anything to Core, TrueGolden, or
general-purpose reasoning.

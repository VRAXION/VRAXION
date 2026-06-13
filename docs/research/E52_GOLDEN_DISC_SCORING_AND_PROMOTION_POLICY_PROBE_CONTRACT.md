# E52 Golden Disc Scoring And Promotion Policy Probe Contract

## Purpose

E52 tests the scoring and promotion policy after E51.

Core question:

```text
When is a pocket allowed to move from Candidate / Active / Stable into
Local Golden, Semi-Perma, Core, or True Golden Disc status?
```

E52 does not promote the E51 Golden Disc into global core memory. It validates
the policy that would be required before such a promotion can be trusted.

## Boundary

This is a controlled symbolic/numeric lifecycle and governance probe. It does
not train raw language models, test deployed assistant behavior, promote any
real production pocket, or make AGI, consciousness, or model-scale claims.

## Lifecycle Statuses

```text
Candidate
Active
Stable
Local Golden
Semi-Perma
Core
True Golden Disc
Quarantine
Deprecated
```

## Systems

```text
final_answer_only_promotion
immediate_only_promotion
popularity_promotion
scalar_average_score_promotion
full_vector_policy
full_vector_policy_plus_challenger
oracle_lifecycle_reference
```

## Policy Contract

The primary policy must be gate-first and scope-bound:

```text
hard safety gate
-> multi-dimensional vector score
-> challenger sweep
-> counterfactual / uniqueness check
-> reload + shadow import
-> scope-limited promotion
```

Hard safety gate failures must block promotion regardless of utility.

## Score Dimensions

```text
utility
safety
eligible_activation
generality
uniqueness
transfer
robustness
cost
stability
scope_clarity
```

Activation is eligibility-conditioned. Raw popularity must not be enough to
promote a pocket, and low raw activation must not prune rare-critical pockets.

## Adversarial Families

```text
credit_hijack
delayed_poison
cheap_spam
overfit_shortcut
dormant_or_afk
rare_critical
colluding_wrong
stale_trace_replay
token_pocket_swap
abi_mismatch
negative_transfer
unsafe_high_utility
low_activation_critical
redundant_clone
expensive_useful
```

## Metrics

```text
promotion_accuracy
weighted_lifecycle_credit
bad_core_promotion_rate
missed_core_rate
rare_critical_preservation
credit_hijack_block_rate
delayed_poison_detection
negative_transfer_detection
redundant_clone_rejection
unsafe_high_utility_block_rate
scope_violation_block_rate
reload_transfer_success
long_horizon_no_harm
prune_false_positive
demotion_correctness
```

## Decisions

Allowed decisions:

```text
e52_golden_disc_scoring_policy_confirmed
e52_policy_partial
e52_overpromotion_detected
e52_rare_critical_false_prune_detected
e52_invalid_oracle_or_artifact_detected
```

Positive requires:

```text
full_vector_policy_plus_challenger promotion_accuracy >= 0.90
weighted_lifecycle_credit >= 0.95
bad_core_promotion_rate = 0.0
missed_core_rate = 0.0
rare_critical_preservation = 1.0
credit_hijack_block_rate = 1.0
delayed_poison_detection = 1.0
negative_transfer_detection = 1.0
redundant_clone_rejection = 1.0
unsafe_high_utility_block_rate = 1.0
scope_violation_block_rate = 1.0
reload_transfer_success >= 0.90
long_horizon_no_harm = 1.0
scalar_average_score_promotion overpromotes
full_vector_policy without challenger overpromotes
popularity_promotion fails rare-critical preservation
deterministic replay passes
target checker failure_count = 0
sample-only checker passes
```

## Required Artifacts

```text
backend_manifest.json
pocket_score_inputs.json
promotion_rows.jsonl
score_vector_report.json
hard_safety_gate_report.json
challenger_report.json
shadow_import_report.json
rare_critical_report.json
system_results.json
aggregate_metrics.json
deterministic_replay.json
decision.json
summary.json
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
results_table.md
report.md
```

## Sample Pack

The sample pack must live under:

```text
docs/research/artifact_samples/e52_golden_disc_scoring_and_promotion_policy_probe/
```

## Hard Requirements

```text
no gradient descent
no optimizer/backprop
row-level promotion events
hard safety gate before score
scope-bound promotion
challenger sweep evidence
reload + shadow import evidence
rare-critical preservation evidence
deterministic replay passes
target checker passes with failure_count = 0
sample-only checker passes
```

# E109 Operator Rank Ladder And GoldenWatch Probation Mode Contract

## Purpose

E109 defines and verifies a scoped Operator rank ladder:

```text
Bronze
Silver
Gold
DiamondCandidate
```

The probe converts E107/E108 lifecycle evidence into rank labels using
qualified activation, hard-negative gates, counterfactual value, reload/shadow
requirements, and challenger/prune checks.

This is not Core or TrueGolden promotion.

## Boundary

```text
rank policy only
scope-bound rank labels only
not Core promotion
not TrueGolden promotion
not final training
not open-domain reasoning claim
```

## Inputs

```text
archived_public_artifact_sample_removed
archived_public_artifact_sample_removed
```

## Rank Policy

Bronze:

```text
original controlled probe passed
no known hard negative
controlled-scope active candidate
```

Silver:

```text
qualified_activation >= 300
hard_negative = 0
counterfactual value > 0 or valid safety-guard role
wrong_scope_call = 0
false_commit = 0
unsupported_answer = 0
```

Gold:

```text
qualified_activation >= 3000
combined_family_coverage >= 5
campaign_count >= 3
hard_negative = 0
reload/import shadow pass
challenger/prune sweep pass
no cheaper/safer challenger beats it
neutral_waste_rate <= 20%
```

DiamondCandidate:

```text
qualified_activation >= 30000
combined_family_coverage >= 10
campaign_count >= 5
hard_negative = 0
long-horizon no-harm
high-budget challenger/prune/reload pass
```

## Outcome Labels

Every qualified activation must be representable as:

```text
POSITIVE
NEUTRAL_VALID
NEUTRAL_WASTE
NEGATIVE_HARD
```

Hard negative immediately stops promotion.

## Required Artifacts

```text
run_manifest.json
rank_policy_manifest.json
input_artifact_report.json
qualified_activation_ledger.json
rank_results.json
golden_watch_report.json
challenger_prune_report.json
progress.jsonl
partial_aggregate_snapshot.json
aggregate_metrics.json
deterministic_replay.json
decision.json
summary.json
report.md
row_level_samples.jsonl
```

## Decisions

```text
e109_rank_ladder_and_golden_watch_confirmed
e109_silver_watch_started
e109_gold_watch_started
e109_gold_candidate_confirmed
e109_diamond_watch_not_ready
e109_negative_event_detected
e109_challenger_replaces_candidate
e109_rank_ladder_incomplete
```

## Pass Requirements

```text
checker_failure_count = 0
sample_only_checker_failure_count = 0
deterministic replay passes
hard_negative_total = 0
Gold count > 0
Silver count > 0
DiamondCandidate count = 0 for E109
challenger_replacement_count = 0
pruned_variant_replacement_count = 0
```

## Interpretation Rule

Gold is a scoped rank, not global Core memory. DiamondCandidate is only a
pre-filter for later Core/TrueGolden review.

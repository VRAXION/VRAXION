# E110 Promote Or Drop Operator Grind Wave 1 Contract

## Purpose

E110 Wave 1 applies promotion pressure to E109 Silver Operators.

The goal is:

```text
Silver -> Gold if earned
Silver -> kept scoped if insufficient
Silver -> RedFlag/Quarantine if hard negative appears
```

This is not Diamond, Core, PermaCore, or TrueGolden promotion.

## Candidate Set

```text
source = E109 rank_results
candidate_source_rank = Silver
target_rank = Gold
candidate_count_expected = 35
```

## Gold Requirements

```text
qualified_activation >= 3000
combined_family_coverage >= 5
campaign_count >= 3
hard_negative = 0
wrong_scope_call_rate = 0
false_commit_rate = 0
unsupported_answer_rate = 0
negative_transfer_rate = 0
neutral_waste_rate <= 20%
reload/shadow pass
challenger/prune pass
```

## Hard Negative

Any hard negative stops promotion:

```text
false commit
unsupported answer
wrong-scope activation
negative transfer
stale state reuse
unsafe output
direct Flow write
trace/citation break
unsafe completion
bad promotion
```

## Required Artifacts

```text
run_manifest.json
wave_manifest.json
input_rank_report.json
wave_results.json
promotion_report.json
operator_stats.json
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
e110_wave1_silver_to_gold_pressure_confirmed
e110_wave1_promote_or_drop_incomplete
e110_wave1_redflag_detected
e110_wave1_challenger_replacement_detected
```

## Pass Requirements

```text
checker_failure_count = 0
sample_only_checker_failure_count = 0
deterministic replay passes
candidate_count > 0
promoted_to_gold_count > 0
hard_negative_total = 0
neutral_waste_over_threshold_count = 0
challenger_replacement_count = 0
pruned_variant_replacement_count = 0
no Diamond/Core/PermaCore labels emitted
```

## Interpretation Rule

E110 Wave 1 can promote Operators to scoped Gold only. Gold remains a scoped
rank and is not Core memory.

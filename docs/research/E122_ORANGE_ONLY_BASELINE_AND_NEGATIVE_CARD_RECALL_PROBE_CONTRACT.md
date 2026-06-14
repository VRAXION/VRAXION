# E122 Orange Only Baseline And Negative Card Recall Probe Contract

## Goal

Create a clean active-library baseline where every non-deprecated Operator is
ranked as scoped `OrangeLegendaryCandidate`, then save rejected/bad mutation
patterns as planner-only Negative Knowledge Cards and measure whether those
cards are recalled/useful.

This is not Core, PermaCore, TrueGolden, final training, or Gemma-style text
generation.

## Core Questions

```text
1. Can the active Operator library be represented as an orange-only baseline?
2. Can rejected/bad mutation shapes be saved as explicit negative cards?
3. Are those negative cards recalled by the mutation planner?
4. Do they prevent repeated bad attempts without false-blocking useful ones?
```

## Input

Use the existing E109-E121 dashboard payload:

```text
active = rank not in {Deprecated, RedFlag}
inactive = rank in {Deprecated, RedFlag}
```

E122 must not revive deprecated or red-flagged Operators.

## Negative Knowledge Cards

Negative cards are not normal callable skills.

They are planner-only priors:

```text
runtime_callable = false
visible_to_router = false
visible_to_mutation_planner = true
```

Each card records:

```text
negative_card_id
operator_id
failed_mutation_type
failure_mode
why_failed
trigger_pattern
severity
planner_response
hit_count
prevented_bad_attempts
false_block_count
manual_note_allowed
replay_ref
```

## Required Artifacts

```text
run_manifest.json
input_operator_report.json
orange_only_results.json
negative_knowledge_cards.json
negative_card_usage_report.json
mutation_summary.json
row_level_samples.jsonl
progress.jsonl
partial_aggregate_snapshot.json
aggregate_metrics.json
deterministic_replay.json
decision.json
summary.json
report.md
checker_summary.json
negative_card_registry/*.json
```

## Required Metrics

```text
active_operator_count
orange_only_active_count
non_orange_active_count
deprecated_count
qualified_activation_min
hard_negative_total
wrong_scope_call_total
false_commit_total
unsupported_answer_total
direct_flow_write_total
negative_card_count
recalled_card_count
negative_card_recall_event_count
prevented_repeat_failure_count
false_block_count
negative_card_recall_rate
negative_card_precision
mutation_attempts_total
accepted_mutations_total
rollback_count_total
deterministic replay hash match
```

## Pass Requirements

```text
orange_only_active_count == active_operator_count
non_orange_active_count == 0
deprecated_count == 3
qualified_activation_min >= 300000
hard_negative_total == 0
wrong_scope_call_total == 0
false_commit_total == 0
unsupported_answer_total == 0
direct_flow_write_total == 0
negative_card_count > 0
negative_card_recall_event_count > 0
prevented_repeat_failure_count > 0
false_block_count == 0
negative cards are planner-only
checker failure_count == 0
deterministic replay passes
```

## Decision Labels

```text
e122_orange_only_baseline_and_negative_cards_confirmed
e122_negative_cards_unused
e122_false_block_detected
e122_orange_baseline_incomplete
```

# E122 Orange Only Baseline And Negative Card Recall Probe Result

```text
decision = e122_orange_only_baseline_and_negative_cards_confirmed
checker_failure_count = 0
```

Boundary:

```text
Orange-only active baseline
mutation-planner negative cards only
not Core
not PermaCore
not TrueGolden
not final training
not Gemma-style generation
```

## Key Metrics

```text
source_operator_count = 147
active_operator_count = 144
inactive_operator_count = 3

pre_orange_active_count = 8
newly_oranged_active_count = 136
orange_only_active_count = 144
non_orange_active_count = 0
deprecated_count = 3
red_flag_count = 0

qualified_activation_min = 300857
qualified_activation_total = 43564286

hard_negative_total = 0
wrong_scope_call_total = 0
false_commit_total = 0
unsupported_answer_total = 0
negative_transfer_total = 0
direct_flow_write_total = 0

negative_card_count = 576
recalled_card_count = 487
useful_card_count = 449
dormant_card_count = 89
negative_card_recall_event_count = 1895
prevented_repeat_failure_count = 1696
false_block_count = 0
negative_card_recall_rate = 0.845486
negative_card_precision = 1.000000

mutation_attempts_total = 314171
accepted_mutations_total = 1564
rejected_mutations_total = 312607
rollback_count_total = 312607

mean_selected_prune_ratio = 0.735903
mean_rule_of_three_upper_failure_bound = 0.00000992
```

## Interpretation

E122 establishes a clean active-library baseline: all 144 non-deprecated
Operators are now represented as scoped `OrangeLegendaryCandidate` rows, while
the 3 deprecated Operators remain inactive.

E122 also adds Negative Knowledge Cards for rejected/bad mutation shapes. These
cards are not normal callable skills and are not shown to the normal Router as
answering capability. They are planner-only negative priors:

```text
runtime_callable = false
visible_to_router = false
visible_to_mutation_planner = true
```

The card layer was actually used in the probe:

```text
negative_card_recall_event_count = 1895
prevented_repeat_failure_count = 1696
false_block_count = 0
```

So the result supports keeping bad/rejected mutation patterns as a structured
mutation-memory layer, as long as they stay planner-only and are measured for
false blocks.

## Artifacts

```text
target/pilot_wave/e122_orange_only_baseline_and_negative_card_recall_probe/
target/pilot_wave/e122_orange_only_baseline_and_negative_card_recall_probe/negative_card_registry/
```

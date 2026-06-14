# E125 Broad Text Understanding Candidate Expansion Wave Result

```text
decision = e125_broad_text_understanding_15plus_gold_positive
checker_failure_count = 0
target_checker_passed = true
```

## Summary

E125 widened the E124 text-understanding candidate pool and successfully
promoted more than fifteen additional scoped Gold operators from the E122
orange-only baseline.

This is scoped operator farming only. It is not Core, PermaCore, TrueGolden,
final training, Gemma-level generation, or open-domain text reasoning.

## Key Metrics

```text
rows_seen = 40000
candidate_pool_count = 42
farmable_candidate_count = 20
selected_candidate_count = 20
target_gold_count = 15
promoted_to_gold_count = 20
supply_gap_to_15 = 0

mutation_lanes = 6
mutation_attempts_total = 23478
accepted_mutations_total = 447
rollback_count_total = 23031

hard_negative_total = 0
false_commit_total = 0
unsupported_answer_total = 0
negative_transfer_total = 0
negative_card_false_block_count = 0
normal_router_callable_cards = 0

max_clean_batch_size = 16
mean_selected_prune_ratio = 0.62
```

The 24/32 batch rows did not pass only because the run found 20 selected
farmable candidates, not because of unsafe behavior.

## Gold Operators

```text
multi_sentence_reference_bridge_lens
modal_strength_guard
sentence_clause_boundary_lens
condition_consequence_lens
data_record_field_lens
scope_limiter_lens
coreference_pointer_lens
parenthetical_qualifier_lens
enumeration_choice_lens
definition_scope_boundary_guard
discourse_relation_lens
exception_contrast_lens
instruction_warning_split_guard
example_boundary_lens
requirement_exception_guard
math_expression_span_lens
comparison_normalization_guard
url_email_reference_lens
definition_example_split_guard
code_command_block_lens
```

All promoted operators selected the `negative_card_guided` mutation lane with
`selected_prune_ratio = 0.62`.

## Interpretation

The current orange-only library can still farm new scoped text-understanding
operators from FineWeb-style text. The bottleneck was not safety or mutation
capacity in this wave: the system found 20 clean Gold candidates against a
target of 15 with zero hard negatives.

The dominant winning lane was negative-card-guided mutation, which supports the
current policy of keeping negative knowledge cards as non-callable governance
memory. They blocked unsafe random variants without becoming normal router
targets.

## Next Recommendation

Run a follow-up stress/ranking wave before treating these as durable library
members:

```text
E126_TEXT_GOLD_NEW_OPERATOR_SURVIVAL_AND_ROLE_GAUNTLET
```

The next wave should test the 20 E125 Gold operators against unseen text
families, negative scope, redundancy with existing orange operators, and
counterfactual ablation value.

# E126 E125 Gold To Orange Legendary Probation Gauntlet Result

```text
decision = e126_orange_legendary_probation_confirmed
checker_failure_count = 0
target_checker_passed = true
```

## Summary

E126 pushed the twenty scoped Gold operators from E125 through the
Orange/Legendary probation gate. All twenty survived.

This is a scoped Orange/LegendaryCandidate result only. It is not Core,
PermaCore, TrueGolden, final training, Gemma-level generation, or open-domain
reasoning.

## Key Metrics

```text
candidate_count = 20
orange_legendary_candidate_count = 20
qualified_activation_min = 300719
qualified_activation_total = 6031177
qualified_activation_add_total = 5828091
family_coverage_min = 12
campaign_count_min = 8

hard_negative_total = 0
false_commit_total = 0
wrong_scope_call_total = 0
unsupported_answer_total = 0
negative_transfer_total = 0
direct_flow_write_total = 0

reload_match_rate = 1.0
negative_scope_pass_rate = 1.0
challenger_pass_rate = 1.0
prune_pass_rate = 1.0

mutation_attempts_total = 95474
accepted_mutations_total = 614
rollback_count_total = 94860
prune_attempts_total = 1026
challenger_attempts_total = 467
mean_selected_prune_ratio = 0.7335
```

## Orange / Legendary Operators

```text
code_command_block_lens
comparison_normalization_guard
condition_consequence_lens
coreference_pointer_lens
data_record_field_lens
definition_example_split_guard
definition_scope_boundary_guard
discourse_relation_lens
enumeration_choice_lens
example_boundary_lens
exception_contrast_lens
instruction_warning_split_guard
math_expression_span_lens
modal_strength_guard
multi_sentence_reference_bridge_lens
parenthetical_qualifier_lens
requirement_exception_guard
scope_limiter_lens
sentence_clause_boundary_lens
url_email_reference_lens
```

## Interpretation

The E125 broad candidate wave did not only produce Gold candidates; all twenty
could be pushed to scoped Orange/LegendaryCandidate status under the same
probation style used previously for E120/E121.

The result strengthens the current text-understanding library, but the scope is
still narrow: these are mechanical lenses/guards for text structure, evidence,
scope, and formatting. They are not free-form language generation or general
reasoning weights.

## Next Recommendation

Continue the overnight cycle:

```text
candidate discovery
-> Gold farm
-> Orange/Legendary probation
-> survival/regression check
-> repeat
```

Every cycle must keep progress artifacts and must not promote beyond
Orange/LegendaryCandidate without a separate Core/PermaCore policy.

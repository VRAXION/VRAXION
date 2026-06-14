# E121 E120 Gold To Orange Legendary Probation Gauntlet Result

```text
decision = e121_orange_legendary_probation_confirmed
checker_failure_count = 0
```

Boundary:

```text
OrangeLegendaryCandidate only
not Core
not PermaCore
not TrueGolden
not Gemma-style generation
not final training
```

## Key Metrics

```text
candidate_count = 8
orange_legendary_candidate_count = 8

qualified_activation_total = 2415626
qualified_activation_add_total = 2314672
qualified_activation_min = 300873
family_coverage_min = 12
campaign_count_min = 8

hard_negative_total = 0
wrong_scope_call_total = 0
false_commit_total = 0
unsupported_answer_total = 0
negative_transfer_total = 0
direct_flow_write_total = 0

reload_match_rate = 1.000000
negative_scope_pass_rate = 1.000000
challenger_pass_rate = 1.000000
prune_pass_rate = 1.000000

mutation_attempts_total = 40111
accepted_mutations_total = 264
rejected_mutations_total = 39847
rollback_count_total = 39847
prune_attempts_total = 391
challenger_attempts_total = 194

mean_selected_prune_ratio = 0.756250
mean_rule_of_three_upper_failure_bound = 0.00000993
```

## Promoted Operators

```text
causal_relation_lens
comparison_quantifier_guard
date_entity_timeline_lens
definition_term_anchor_lens
named_entity_anchor_lens
procedure_step_parser_lens
quote_speaker_attribution_lens
safety_domain_caution_guard
```

Each of the eight E120 scoped Gold FineWeb Operators reached
`OrangeLegendaryCandidate` status with at least 300k clean scoped qualified
activation and zero hard negatives.

## Interpretation

E121 moves the E120 FineWeb Operators from scoped Gold to scoped
Orange/LegendaryCandidate status. This means they have substantially more
probation evidence, cleaner pruned forms, and stricter no-harm evidence than
their E120 state.

This does not make them Core, PermaCore, TrueGolden, open-domain language
reasoners, or Gemma-style generation modules. They remain scoped text-grounding
/ guard / lens Operators.

## Artifacts

```text
target/pilot_wave/e121_e120_gold_to_orange_legendary_probation_gauntlet/
target/pilot_wave/e121_e120_gold_to_orange_legendary_probation_gauntlet/operator_registry/
```

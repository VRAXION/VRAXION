# E120 FineWeb Skill Farm To Gold Wave Result

```text
decision = e120_fineweb_skill_farm_gold_positive
checker_failure_count = 0
```

Boundary:

```text
scoped Gold promotion only
not Core
not PermaCore
not TrueGolden
not Gemma-style generation
not final training
```

## Key Metrics

```text
candidate_count = 8
saved_operator_count = 8
promoted_to_gold_count = 8
kept_silver_count = 0

qualified_activation_total = 100954
qualified_activation_min = 12035
family_coverage_min = 8
campaign_count_min = 4

hard_negative_total = 0
wrong_scope_call_total = 0
false_commit_total = 0
unsupported_answer_total = 0
negative_transfer_total = 0

reload_match_rate = 1.000000
negative_scope_pass_rate = 1.000000
challenger_pass_rate = 1.000000
prune_pass_rate = 1.000000
mean_selected_prune_ratio = 0.641250
```

Mutation/prune pressure:

```text
mutation_attempts_total = 3333
accepted_mutations_total = 95
rejected_mutations_total = 3238
rollback_count_total = 3238
prune_attempts_total = 224
challenger_attempts_total = 121
selected_variant_type = pruned_gold for all 8
```

## New Scoped Gold Operators

```text
definition_term_anchor_lens
named_entity_anchor_lens
causal_relation_lens
date_entity_timeline_lens
comparison_quantifier_guard
procedure_step_parser_lens
safety_domain_caution_guard
quote_speaker_attribution_lens
```

Per-operator evidence:

```text
definition_term_anchor_lens
  rank = Gold
  qualified_activation = 12908
  family_coverage = 8
  campaign_count = 4

named_entity_anchor_lens
  rank = Gold
  qualified_activation = 12988
  family_coverage = 8
  campaign_count = 5

causal_relation_lens
  rank = Gold
  qualified_activation = 12035
  family_coverage = 9
  campaign_count = 6

date_entity_timeline_lens
  rank = Gold
  qualified_activation = 12759
  family_coverage = 9
  campaign_count = 5

comparison_quantifier_guard
  rank = Gold
  qualified_activation = 12212
  family_coverage = 11
  campaign_count = 6

procedure_step_parser_lens
  rank = Gold
  qualified_activation = 12779
  family_coverage = 8
  campaign_count = 5

safety_domain_caution_guard
  rank = Gold
  qualified_activation = 13032
  family_coverage = 11
  campaign_count = 5

quote_speaker_attribution_lens
  rank = Gold
  qualified_activation = 12241
  family_coverage = 8
  campaign_count = 5
```

## Interpretation

E120 converts the E119 FineWeb FarmCandidates into saved scoped Gold Operators.
The selected form for every new Operator was `pruned_gold`, which means the
candidate had enough support to train a scoped ABI/Proposal Field contract and
then pass a simplification/challenger gate.

These are not Core memory and not natural-language generation modules. They are
FineWeb text-grounding / guard / lens Operators that can now be used as scoped
skills in later curriculum and no-harm gauntlets.

## Artifacts

```text
target/pilot_wave/e120_fineweb_skill_farm_to_gold_wave/
target/pilot_wave/e120_fineweb_skill_farm_to_gold_wave/operator_registry/
```

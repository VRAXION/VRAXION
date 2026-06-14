# E124 Text Understanding Mass Mutation Farm Capacity Probe Result

```text
decision = e124_mass_mutation_text_farm_capacity_positive
checker_failure_count = 0
```

Boundary:

```text
scoped mass Operator farming only
not Core
not PermaCore
not TrueGolden
not final training
not Gemma-style generation
```

## Key Metrics

```text
rows_seen = 40000
active_operator_count = 144
orange_only_confirmed = true
negative_card_count = 576

candidate_pool_count = 16
farmable_candidate_count = 5
selected_candidate_count = 5
promoted_to_gold_count = 5
kept_silver_count = 0

mutation_lane_count = 6
mutation_attempts_total = 5824
accepted_mutations_total = 117
rejected_mutations_total = 5707
rollback_count_total = 5707

hard_negative_total = 0
wrong_scope_call_total = 0
false_commit_total = 0
unsupported_answer_total = 0
negative_transfer_total = 0

negative_card_blocked_variant_count = 5
negative_card_false_block_count = 0
normal_router_callable_cards = 0

mean_selected_prune_ratio = 0.620000
max_clean_batch_size = 4
```

## Gold Operators

```text
sentence_clause_boundary_lens
  variant = negative_card_guided
  qualified_activation = 13172
  prune = 0.62

coreference_pointer_lens
  variant = negative_card_guided
  qualified_activation = 12551
  prune = 0.62

definition_scope_boundary_guard
  variant = negative_card_guided
  qualified_activation = 13206
  prune = 0.62

discourse_relation_lens
  variant = negative_card_guided
  qualified_activation = 11085
  prune = 0.62

code_command_block_lens
  variant = negative_card_guided
  qualified_activation = 3000
  prune = 0.62
```

## Batch Capacity

```text
batch 2  -> 2/2 Gold, pass
batch 4  -> 4/4 Gold, pass
batch 8  -> 5/5 evaluated, not pass because only 5 farmable candidates existed
batch 12 -> 5/5 evaluated, not pass because only 5 farmable candidates existed
batch 16 -> 5/5 evaluated, not pass because only 5 farmable candidates existed
```

Interpretation:

```text
The current mass-add limit was candidate supply, not safety.
All 5 farmable candidates became scoped Gold.
The clean guaranteed batch size is 4.
```

## Mutation Lane Result

Every selected winner came from:

```text
negative_card_guided
```

This is useful: the E122 negative-card memory is not only passive bookkeeping.
It steered mass farming toward safer variants while keeping:

```text
negative_card_false_block_count = 0
normal_router_callable_cards = 0
```

## Recommended Next

```text
E125_TEXT_UNDERSTANDING_GOLD_TO_ORANGE_PROBATION_WAVE
```

Move the five E124 scoped Gold text-understanding Operators into an Orange
probation wave, then rerun a broader discovery scan to see whether the next
limiting layer is semantic roles, paraphrase, topic shifts, or entity
attribute binding.

## Artifacts

```text
target/pilot_wave/e124_text_understanding_mass_mutation_farm_capacity_probe/
target/pilot_wave/e124_text_understanding_mass_mutation_farm_capacity_probe/operator_registry/
```

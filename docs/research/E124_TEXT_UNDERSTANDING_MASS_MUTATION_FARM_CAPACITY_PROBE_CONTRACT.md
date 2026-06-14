# E124 Text Understanding Mass Mutation Farm Capacity Probe Contract

## Goal

Test how many scoped text-understanding Operators can be farmed safely in one
wave when multiple mutation lanes are kept alive in parallel.

This follows E122/E123:

```text
E122 = orange-only active baseline + planner-only negative cards
E123 = FineWeb discovery found remaining under-covered text patterns
E124 = mass farm capacity test over a broader text-understanding candidate pool
```

This is not Core, PermaCore, TrueGolden, final training, or Gemma-style
generation.

## Candidate Pool

The candidate pool targets low/mid-level text understanding layers:

```text
unit_dimension_guard
code_command_block_lens
sentence_clause_boundary_lens
coreference_pointer_lens
semantic_role_frame_lens
paraphrase_equivalence_lens
topic_shift_boundary_lens
discourse_relation_lens
entity_attribute_binding_lens
quote_attribution_scope_lens
table_list_structure_lens
evidence_quality_source_tier_guard
negation_scope_guard
acronym_expansion_anchor_lens
timeline_event_sequence_lens
definition_scope_boundary_guard
```

## Mutation Lanes

Each selected candidate keeps several mutation lanes alive:

```text
random_seed_mutation
guided_existing_neighbor
prune_heavy_contract
negative_card_guided
sibling_challenger
compact_contract_variant
```

The selected variant must pass reload, negative-scope, challenger, and prune
gates.

## Capacity Sweep

Report whether the system can safely run candidate batches:

```text
2
4
8
12
16
```

The key output is:

```text
max_clean_batch_size
```

## Required Artifacts

```text
run_manifest.json
candidate_pool_report.json
mass_mutation_lane_report.json
operator_cards.json
operator_gold_results.json
variant_report.json
batch_capacity_report.json
negative_card_interaction_report.json
mutation_summary.json
row_level_samples.jsonl
candidate_examples.jsonl
progress.jsonl
partial_aggregate_snapshot.json
aggregate_metrics.json
deterministic_replay.json
decision.json
summary.json
report.md
checker_summary.json
operator_registry/*.json
```

## Pass Requirements

```text
rows_seen >= 10000
active_operator_count = 144
orange_only_confirmed = true
mutation_lane_count >= 4
promoted_to_gold_count >= 2
hard_negative_total = 0
wrong_scope_call_total = 0
false_commit_total = 0
unsupported_answer_total = 0
negative_transfer_total = 0
negative_card_false_block_count = 0
normal_router_callable_cards = 0
negative_card_blocked_variant_count > 0
deterministic replay passes
checker failure_count = 0
```

## Decision Labels

```text
e124_mass_mutation_text_farm_capacity_positive
e124_mass_farm_capacity_limited
e124_no_mass_farm_capacity_detected
e124_mass_farm_redflag_detected
```

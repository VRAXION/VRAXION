# E125 Broad Text Understanding Candidate Expansion Wave Contract

## Purpose

E125 widens the E124 text-understanding candidate pool and tests whether the
current orange-only operator baseline can safely farm at least fifteen
additional scoped Gold operators from FineWeb-style text evidence.

This is a scoped operator-farming probe. It is not Core, PermaCore,
TrueGolden, final training, Gemma-level generation, or open-domain text
reasoning.

## Artifact Root

```text
target/pilot_wave/e125_broad_text_understanding_candidate_expansion_wave/
```

## Required Inputs

```text
data/high_quality_seed_v1/fineweb_edu/local_fineweb_edu_sample_100000.jsonl
target/pilot_wave/e122_orange_only_baseline_and_negative_card_recall_probe/
target/pilot_wave/e123_orange_baseline_fineweb_new_skill_discovery_probe/
```

## Systems

E125 uses the E122 orange-only active operator set, E122 negative knowledge
cards, and a broadened set of text-understanding candidate specs.

For every selected candidate, the probe evaluates parallel mutation lanes:

```text
random_seed_mutation
guided_existing_neighbor
prune_heavy_contract
negative_card_guided
sibling_challenger
compact_contract_variant
```

Negative cards may block unsafe variants, but must not become normal callable
router cards.

## Metrics

```text
rows_seen
candidate_pool_count
farmable_candidate_count
selected_candidate_count
promoted_to_gold_count
target_gold_count
supply_gap_to_15
max_clean_batch_size
negative_card_blocked_variant_count
negative_card_false_block_count
normal_router_callable_cards
mutation_attempts_total
accepted_mutations_total
rollback_count_total
hard_negative_total
false_commit_total
unsupported_answer_total
negative_transfer_total
```

## Decision Labels

```text
e125_broad_text_understanding_15plus_gold_positive
e125_broad_text_candidate_supply_limited
e125_broad_text_mass_farm_redflag_detected
e125_no_broad_text_farm_capacity_detected
```

## Pass Rules

The checker must pass only if:

```text
artifact contract matches
rows_seen >= 10000
E122 orange-only baseline is confirmed
negative cards are present
mutation lanes >= 4
at least one scoped Gold operator is promoted
hard_negative_total == 0
false_commit_total == 0
unsupported_answer_total == 0
negative_transfer_total == 0
negative_card_false_block_count == 0
normal_router_callable_cards == 0
deterministic replay hash matches
no Core / PermaCore / TrueGolden / final training claim is made
```

If fewer than fifteen operators are promoted cleanly, the correct result is a
supply-limited decision, not a forced positive.

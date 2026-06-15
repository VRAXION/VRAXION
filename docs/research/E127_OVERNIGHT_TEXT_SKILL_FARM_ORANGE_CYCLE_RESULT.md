# E127 Overnight Text Skill Farm Orange Cycle Result

```text
decision = e127_overnight_cycle_positive
cycle_count = 30
orange_legendary_candidate_total = 268
hard_negative_total = 0
false_commit_total = 0
wrong_scope_call_total = 0
unsupported_answer_total = 0
```

## Summary

E127 starts the unattended cyclic loop requested after E125/E126:

```text
candidate discovery
-> scoped Gold farm
-> Orange/Legendary probation
-> repeat with already-promoted operators excluded
```

The supervised overnight loop has completed 30 checkpointed cycles so far. Several candidate
spec packs were added as the currently visible candidate space was exhausted;
each pack was validated by running the next cycle with active already-promoted
operators excluded.

This is scoped operator farming only. It is not Core, PermaCore, TrueGolden,
final training, Gemma-level generation, or open-domain reasoning.

## Aggregate Metrics

```text
selected_candidate_total = 268
orange_legendary_candidate_total = 268
mutation_attempts_total = 1297730
accepted_mutations_total = 8517
rollback_count_total = 1289213

hard_negative_total = 0
false_commit_total = 0
wrong_scope_call_total = 0
unsupported_answer_total = 0
```

## Latest Successful Cycle 028 Metrics

```text
selected_candidate_count = 9
orange_legendary_candidate_count = 9
candidate_pool_count = 58
farmable_candidate_count = 9
qualified_activation_min = 300791
mutation_attempts_total = 40747
accepted_mutations_total = 307
rollback_count_total = 40440
mean_selected_prune_ratio = 0.62

reload_match_rate = 1.0
negative_scope_pass_rate = 1.0
prune_pass_rate = 1.0
challenger_pass_rate = 1.0

hard_negative_total = 0
false_commit_total = 0
wrong_scope_call_total = 0
unsupported_answer_total = 0
```

## Latest Successful Cycle 028 Orange Operators

```text
e128_observation_reported_by_lens
e128_staleness_due_to_date_guard
e128_negated_causal_guard
e128_exact_vs_approximate_guard
e128_method_assumption_result_lens
e128_candidate_promotion_reason_lens
e128_known_unknown_split_lens
e128_question_scope_guard
e128_partial_success_failure_lens
```

## Latest Boundary Cycle 029 Metrics

Cycle 029 found no additional farmable candidates in the current candidate
spec pack. This is a clean exhaustion boundary, not a regression.

```text
decision = e127_cycle_no_candidates
rows_seen = 40000
candidate_pool_count = 49
farmable_candidate_count = 0
selected_candidate_count = 0
orange_legendary_candidate_count = 0

hard_negative_total = 0
false_commit_total = 0
wrong_scope_call_total = 0
unsupported_answer_total = 0
```

## Latest Successful Cycle 030 Metrics

Cycle 030 used the fresh `e129_` candidate spec pack added after the cycle 029
exhaustion boundary.

```text
selected_candidate_count = 16
orange_legendary_candidate_count = 16
candidate_pool_count = 107
farmable_candidate_count = 48
qualified_activation_min = 300816
mutation_attempts_total = 77578
accepted_mutations_total = 507
rollback_count_total = 77071
mean_selected_prune_ratio = 0.62

reload_match_rate = 1.0
negative_scope_pass_rate = 1.0
prune_pass_rate = 1.0
challenger_pass_rate = 1.0

hard_negative_total = 0
false_commit_total = 0
wrong_scope_call_total = 0
unsupported_answer_total = 0
```

## Latest Successful Cycle 030 Orange Operators

```text
e129_range_endpoint_inclusive_lens
e129_pronoun_antecedent_guard
e129_scope_creep_request_lens
e129_objective_subjective_split_lens
e129_timeline_chain_lens
e129_private_public_boundary_guard
e129_inequality_direction_guard
e129_unit_conversion_context_guard
e129_aggregation_granularity_lens
e129_tradeoff_preference_lens
e129_rank_order_tie_guard
e129_metaphor_literal_boundary_guard
e129_user_preference_constraint_lens
e129_list_item_negative_scope_guard
e129_step_numbering_sequence_lens
e129_batch_stream_boundary_lens
```

Cycle 031 started after this checkpoint. Later cycle results should be treated
as provisional until their cycle artifacts are checkpointed.

## Current Boundary

The current state is still scoped Operator farming only:

```text
Orange/LegendaryCandidate = yes
Core = no
PermaCore = no
TrueGolden = no
Gemma-level text generation = no
open-domain reasoning claim = no
```

## Progress Safety

The runner writes:

```text
target/pilot_wave/e127_overnight_text_skill_farm_orange_cycle/progress.jsonl
target/pilot_wave/e127_overnight_text_skill_farm_orange_cycle/partial_aggregate_snapshot.json
target/pilot_wave/e127_overnight_text_skill_farm_orange_cycle/cycles/cycle_###/
```

Each cycle has its own candidate pool, selected operator cards, orange results,
variant report, row-level samples, deterministic replay hash, decision, and
report.

## Stop Mechanism

Create this file to stop the loop at the next cycle boundary:

```text
target/pilot_wave/e127_overnight_text_skill_farm_orange_cycle/STOP
```

# E127 Overnight Text Skill Farm Orange Cycle Result

```text
decision = e127_overnight_cycle_positive
cycle_count = 29
orange_legendary_candidate_total = 252
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

The supervised overnight loop has completed 29 checkpointed cycles so far. Several candidate
spec packs were added as the currently visible candidate space was exhausted;
each pack was validated by running the next cycle with active already-promoted
operators excluded.

This is scoped operator farming only. It is not Core, PermaCore, TrueGolden,
final training, Gemma-level generation, or open-domain reasoning.

## Aggregate Metrics

```text
selected_candidate_total = 252
orange_legendary_candidate_total = 252
mutation_attempts_total = 1220152
accepted_mutations_total = 8010
rollback_count_total = 1212142

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

Cycle 030 should only be started after adding a fresh candidate spec pack or
changing the data/curriculum source. Later cycle results should be treated as
provisional until their cycle artifacts are checkpointed.

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

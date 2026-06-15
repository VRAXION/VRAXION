# E127 Overnight Text Skill Farm Orange Cycle Result

```text
decision = e127_overnight_cycle_positive
cycle_count = 31
orange_legendary_candidate_total = 284
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

The supervised overnight loop has completed 31 checkpointed cycles so far. Several candidate
spec packs were added as the currently visible candidate space was exhausted;
each pack was validated by running the next cycle with active already-promoted
operators excluded.

This is scoped operator farming only. It is not Core, PermaCore, TrueGolden,
final training, Gemma-level generation, or open-domain reasoning.

## Aggregate Metrics

```text
selected_candidate_total = 284
orange_legendary_candidate_total = 284
mutation_attempts_total = 1374205
accepted_mutations_total = 9016
rollback_count_total = 1365189

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

## Latest Successful Cycle 031 Metrics

Cycle 031 continued the fresh `e129_` candidate spec pack after cycle 030
promoted the first 16 candidates from that pack.

```text
selected_candidate_count = 16
orange_legendary_candidate_count = 16
candidate_pool_count = 91
farmable_candidate_count = 32
qualified_activation_min = 300759
mutation_attempts_total = 76475
accepted_mutations_total = 499
rollback_count_total = 75976
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

## Latest Successful Cycle 031 Orange Operators

```text
e129_alias_identity_resolution_lens
e129_security_permission_boundary_guard
e129_assumption_dependency_guard
e129_exception_to_general_rule_lens
e129_temporal_order_causality_guard
e129_promise_capability_guard
e129_data_license_usage_guard
e129_hypothetical_scenario_guard
e129_action_reversibility_guard
e129_absolute_relative_delta_lens
e129_evidence_quality_hierarchy_lens
e129_intervention_causality_lens
e129_budget_time_constraint_lens
e129_significance_claim_guard
e129_quote_question_answer_split_lens
e129_outlier_exception_lens
```

Cycle 032 started after this checkpoint. Later cycle results should be treated
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

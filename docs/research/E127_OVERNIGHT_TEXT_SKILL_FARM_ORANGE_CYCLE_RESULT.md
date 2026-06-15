# E127 Overnight Text Skill Farm Orange Cycle Result

```text
decision = e127_overnight_cycle_positive
cycle_count = 27
orange_legendary_candidate_total = 243
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

The supervised overnight loop has completed 27 checkpointed cycles so far. Several candidate
spec packs were added as the currently visible candidate space was exhausted;
each pack was validated by running the next cycle with active already-promoted
operators excluded.

This is scoped operator farming only. It is not Core, PermaCore, TrueGolden,
final training, Gemma-level generation, or open-domain reasoning.

## Aggregate Metrics

```text
selected_candidate_total = 243
orange_legendary_candidate_total = 243
mutation_attempts_total = 1179405
accepted_mutations_total = 7703
rollback_count_total = 1171702

hard_negative_total = 0
false_commit_total = 0
wrong_scope_call_total = 0
unsupported_answer_total = 0
```

## Latest Successful Cycle 027 Metrics

```text
selected_candidate_count = 16
orange_legendary_candidate_count = 16
candidate_pool_count = 74
farmable_candidate_count = 25
qualified_activation_min = 300740
mutation_attempts_total = 76286
accepted_mutations_total = 545
rollback_count_total = 75741
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

## Latest Successful Cycle 027 Orange Operators

```text
e128_progress_ledger_status_lens
e128_answer_format_constraint_lens
e128_multi_condition_rule_lens
e128_external_side_effect_guard
e128_correlation_causation_guard
e128_official_unofficial_source_lens
e128_user_intent_background_split_lens
e128_multi_turn_continuity_lens
e128_tool_output_status_lens
e128_future_prediction_guard
e128_followup_dependency_lens
e128_before_after_state_update_lens
e128_active_evidence_request_lens
e128_claim_source_date_lens
e128_non_actionable_note_guard
e128_unresolved_reference_guard
```

Cycle 028 started after this checkpoint. Later cycle results should be treated
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

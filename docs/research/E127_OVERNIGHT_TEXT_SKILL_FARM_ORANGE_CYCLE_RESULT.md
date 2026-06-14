# E127 Overnight Text Skill Farm Orange Cycle Result

```text
decision = e127_overnight_cycle_positive
cycle_count = 16
orange_legendary_candidate_total = 143
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

The supervised overnight loop has completed 16 cycles so far. Several candidate
spec packs were added as the currently visible candidate space was exhausted;
each pack was validated by running the next cycle with active already-promoted
operators excluded.

This is scoped operator farming only. It is not Core, PermaCore, TrueGolden,
final training, Gemma-level generation, or open-domain reasoning.

## Aggregate Metrics

```text
selected_candidate_total = 143
orange_legendary_candidate_total = 143
mutation_attempts_total = 698424
accepted_mutations_total = 4540
rollback_count_total = 693884

hard_negative_total = 0
false_commit_total = 0
wrong_scope_call_total = 0
unsupported_answer_total = 0
```

## Latest Successful Cycle 015 Metrics

```text
selected_candidate_count = 12
orange_legendary_candidate_count = 12
candidate_pool_count = 27
farmable_candidate_count = 12
qualified_activation_min = 300694
mutation_attempts_total = 58225
accepted_mutations_total = 371
rollback_count_total = 57854
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

## Latest Successful Cycle 015 Orange Operators

```text
unit_conversion_phrase_lens
survey_response_scale_lens
benchmark_leakage_guard
privacy_personal_data_lens
dataset_provenance_lens
file_permission_mode_lens
model_checkpoint_lens
mitigation_action_lens
chart_axis_legend_lens
failure_mode_phrase_lens
reproducibility_seed_lens
table_row_value_lens
```

Cycle 016 then scanned 40,000 rows and stopped cleanly with no farmable
candidate left in the current candidate set.

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

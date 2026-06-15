# E127 Overnight Text Skill Farm Orange Cycle Result

```text
decision = e127_overnight_cycle_positive
cycle_count = 26
orange_legendary_candidate_total = 227
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

The supervised overnight loop completed 26 cycles in this checkpoint. Several candidate
spec packs were added as the currently visible candidate space was exhausted;
each pack was validated by running the next cycle with active already-promoted
operators excluded.

This is scoped operator farming only. It is not Core, PermaCore, TrueGolden,
final training, Gemma-level generation, or open-domain reasoning.

## Aggregate Metrics

```text
selected_candidate_total = 227
orange_legendary_candidate_total = 227
mutation_attempts_total = 1103119
accepted_mutations_total = 7158
rollback_count_total = 1095961

hard_negative_total = 0
false_commit_total = 0
wrong_scope_call_total = 0
unsupported_answer_total = 0
```

## Latest Successful Cycle 025 Metrics

```text
selected_candidate_count = 3
orange_legendary_candidate_count = 3
candidate_pool_count = 32
farmable_candidate_count = 3
qualified_activation_min = 301217
mutation_attempts_total = 14802
accepted_mutations_total = 99
rollback_count_total = 14703
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

## Latest Successful Cycle 025 Orange Operators

```text
hypothesis_test_result_lens
observation_inference_split_lens
root_cause_evidence_lens
```

Cycle 026 was an exhaustion boundary: it scanned 40,000 rows, saw 29 candidate
patterns, found 0 farmable candidates, and stopped cleanly with no safety
failures. Continuing the overnight loop now requires adding a new candidate
spec pack and restarting from this checkpoint.

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

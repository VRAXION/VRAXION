# E127 Overnight Text Skill Farm Orange Cycle Result

```text
decision = e127_overnight_cycle_positive
cycle_count = 1
orange_legendary_candidate_total = 16
hard_negative_total = 0
false_commit_total = 0
```

## Summary

E127 starts the unattended cyclic loop requested after E125/E126:

```text
candidate discovery
-> scoped Gold farm
-> Orange/Legendary probation
-> repeat with already-promoted operators excluded
```

The first supervised full cycle completed cleanly.

This is scoped operator farming only. It is not Core, PermaCore, TrueGolden,
final training, Gemma-level generation, or open-domain reasoning.

## First Cycle Metrics

```text
selected_candidate_total = 16
orange_legendary_candidate_total = 16
mutation_attempts_total = 81235
accepted_mutations_total = 500
rollback_count_total = 80735

hard_negative_total = 0
false_commit_total = 0
wrong_scope_call_total = 0
unsupported_answer_total = 0
```

## First Cycle Orange Operators

```text
location_address_span_lens
entity_attribute_binding_lens
task_deadline_lens
sequence_ordering_lens
timeline_event_sequence_lens
negation_scope_guard
semantic_role_frame_lens
hedge_uncertainty_strength_t_stab
imperative_action_request_lens
absolute_claim_guard
acronym_expansion_anchor_lens
appositive_alias_lens
cause_vs_correlation_guard
topic_shift_boundary_lens
ambiguous_reference_defer_guard
evidence_quality_source_tier_guard
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

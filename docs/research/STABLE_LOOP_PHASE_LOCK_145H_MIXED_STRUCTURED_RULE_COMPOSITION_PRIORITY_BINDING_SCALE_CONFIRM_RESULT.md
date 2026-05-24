# STABLE_LOOP_PHASE_LOCK_145H_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_SCALE_CONFIRM Result

145H is a scale-confirm milestone for the 145A mixed structured-rule composition priority binding prototype.

Boundary: 145H is constrained helper/backend evidence only: mixed structured-rule composition with explicit priority over block types only, not natural-language rule reasoning, not open-ended arbitration, not GPT-like/open-domain capability, not production readiness, and not architecture superiority.

## Expected Decision

```text
decision = mixed_structured_rule_composition_priority_binding_scale_confirmed
verdict = INSTNCT_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_SCALE_CONFIRMED
next = 145Z_MIXED_STRUCTURED_RULE_COMPOSITION_NEXT_DECISION_PLAN
```

## What 145H Tests

145A established the prototype:

```text
multiple canonical structured rule blocks
-> per-block candidate pocket derivation
-> explicit priority over block types
-> final selected pocket id
-> existing static marker binding
-> same-line value extraction
```

145H repeats that behavior at larger coverage without modifying the helper. It uses the already-added manifest-gated decoder:

```text
deterministic_pocket_gated_mixed_structured_rule_composition_binding_decoder
```

It also confirms the older decoders remain present and unchanged:

```text
deterministic_pocket_gated_rule_selected_pocket_binding_decoder
deterministic_pocket_gated_structured_rule_metadata_binding_decoder
```

## Scale Evidence

The default smoke scale is:

```text
seeds = 5301,5302,5303,5304
families = 25
groups_per_family = 24
group_size = 4
main_eval_rows = 9600
```

Generated artifacts are written only under:

```text
target/pilot_wave/stable_loop_phase_lock_145h_mixed_structured_rule_composition_priority_binding_scale_confirm/
```

The runner writes `queue.json` and `progress.jsonl` immediately and continues heartbeat/progress writes during generation, so the run is not a black-box run.

## Denominator Guard

145H separates positive composition rows from expected-fallback rows:

```text
positive rows:
  end_to_end_answer_accuracy
  final_selected_pocket_derivation_accuracy
  selected_pocket_to_marker_binding_accuracy
  same_line_value_extraction_accuracy

fallback rows:
  fallback_control_subset_accuracy
  missing/duplicate/unknown priority rejection
  structural invalid prompt fallback
  priority=pocket_* oracle rejection
```

The checker rejects the run if expected-fallback rows are included in the positive answer denominator.

## Coverage Guard

145H requires coverage of:

```text
recency>quorum>tie_break
quorum>recency>tie_break
tie_break>quorum>recency
each block type winning by priority
each pocket id winning through each supported block type where applicable
distinct candidates in SAME_BLOCKS_DIFFERENT_PRIORITY
priority-only changes winner
```

## Interpretation

If positive, 145H confirms only the constrained helper/backend mixed structured-rule composition primitive at scale. It does not show natural-language rule reasoning, open-ended arbitration, GPT-like/open-domain capability, production readiness, or architecture superiority.

## Status

The executable result is produced by:

```text
scripts/probes/run_stable_loop_phase_lock_145h_mixed_structured_rule_composition_priority_binding_scale_confirm.py
```

The checker is:

```text
scripts/probes/run_stable_loop_phase_lock_145h_mixed_structured_rule_composition_priority_binding_scale_confirm_check.py
```

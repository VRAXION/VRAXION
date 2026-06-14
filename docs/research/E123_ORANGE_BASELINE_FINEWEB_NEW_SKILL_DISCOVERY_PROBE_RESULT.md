# E123 Orange Baseline FineWeb New Skill Discovery Probe Result

```text
decision = e123_new_skill_candidates_found_after_orange_baseline
checker_failure_count = 0
```

Boundary:

```text
discovery only
no promotion
not Core
not PermaCore
not TrueGolden
not final training
not Gemma-style generation
```

## Key Metrics

```text
rows_seen = 40000
active_operator_count = 144
orange_only_confirmed = true
negative_card_count = 576

candidate_count = 11
new_farm_candidate_count = 2
covered_candidate_count = 9
watch_candidate_count = 0

negative_card_blocked_bad_variant_count = 3168
negative_card_false_block_count = 0
normal_router_callable_cards = 0
```

## New Farm Candidates

```text
unit_dimension_guard
  support = 5564
  avg_orange_coverage = 0.667
  gap_score = 3705.624

code_command_block_lens
  support = 309
  avg_orange_coverage = 0.500
  gap_score = 309.000
```

Interpretation:

```text
The orange baseline covers most candidate space tested here,
but two repeated FineWeb patterns remain under-covered:

1. number/unit/dimension claims
2. code or command-like spans
```

## Covered By Orange Baseline

```text
acronym_expansion_anchor_lens
citation_reference_span_lens
conditional_requirement_guard
contrast_exception_scope_guard
definition_term_anchor_lens
duration_frequency_lens
evidence_quality_source_tier_guard
layout_table_list_lens
negation_scope_guard
```

## Negative Card Layer

Negative cards stayed planner-only:

```text
normal_router_callable_cards = 0
negative_card_false_block_count = 0
```

They did block unsafe candidate variants:

```text
negative_card_blocked_bad_variant_count = 3168
```

So the E122 negative memory layer is being used as intended during candidate
analysis, without becoming normal callable answer capability.

## Recommended Next

```text
E124_UNIT_DIMENSION_AND_CODE_COMMAND_OPERATOR_FARM_TO_GOLD
```

Farm the two E123 candidates separately, because their scopes are different:

```text
unit_dimension_guard:
  preserve number + unit + dimension bundles
  reject dimension mixups

code_command_block_lens:
  detect code / command spans
  prevent ordinary-claim handling of executable text
```

## Artifacts

```text
target/pilot_wave/e123_orange_baseline_fineweb_new_skill_discovery_probe/
```

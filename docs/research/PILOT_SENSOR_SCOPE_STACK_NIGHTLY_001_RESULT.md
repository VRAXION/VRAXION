# PILOT_SENSOR_SCOPE_STACK_NIGHTLY_001 Result

## Goal

Decide whether learned scope-aware text-to-evidence failure is mainly data/architecture, or whether an explicit scope/event parser is needed.

Pipeline under test: `text -> scope flags -> evidence -> fixed guard -> action`.

## Setup

- Fixed guard: `evidence_strength_margin_guard` from `PILOT_TOPK_GUARD_001`.
- Teacher: `structured_rule_sensor` from `PILOT_SENSOR_001`.
- Direct arms predict evidence bands directly.
- Scope-stack arms predict scope flags, then deterministic mapper produces evidence.
- No direct action head is used for verdict.

## Gate Summary

```json
{
  "confirmed": [
    "direct_evidence_char_ngram_mlp",
    "scope_stack_char_ngram_mlp"
  ],
  "matrix_promoted": [
    "direct_evidence_char_ngram_mlp",
    "scope_stack_char_ngram_mlp"
  ],
  "smoke_viable": [
    "direct_evidence_char_ngram_mlp",
    "direct_evidence_word_ngram_mlp",
    "scope_stack_char_ngram_mlp",
    "scope_stack_word_ngram_mlp"
  ],
  "stopped": ""
}
```

## Aggregate Metrics

| Stage:Model | Seeds | Score | Action | Scope | Neg | Corr | Weak/Amb | False Commit | Trap False | Scope Flags | Evidence Band | Strict Syn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `confirm:direct_evidence_char_ngram_mlp` | `10` | `0.811` | `0.937` | `1.000` | `1.000` | `1.000` | `0.450` | `0.063` | `0.000` | `0.000` | `0.897` | `0.200` |
| `confirm:scope_stack_char_ngram_mlp` | `10` | `0.814` | `0.934` | `1.000` | `1.000` | `0.950` | `0.475` | `0.060` | `0.000` | `0.909` | `0.909` | `0.200` |
| `matrix:direct_evidence_char_ngram_mlp` | `3` | `0.829` | `0.943` | `1.000` | `1.000` | `1.000` | `0.500` | `0.057` | `0.000` | `0.000` | `0.895` | `0.200` |
| `matrix:direct_evidence_word_ngram_mlp` | `3` | `0.743` | `0.914` | `1.000` | `1.000` | `1.000` | `0.250` | `0.086` | `0.000` | `0.000` | `0.886` | `0.000` |
| `matrix:scope_stack_char_ngram_mlp` | `3` | `0.800` | `0.933` | `1.000` | `1.000` | `1.000` | `0.417` | `0.067` | `0.000` | `0.914` | `0.914` | `0.200` |
| `matrix:scope_stack_word_ngram_mlp` | `3` | `0.643` | `0.895` | `1.000` | `1.000` | `0.750` | `0.333` | `0.076` | `0.000` | `0.857` | `0.857` | `0.000` |
| `reference:keyword_sensor` | `1` | `-3.819` | `0.486` | `0.200` | `0.333` | `0.000` | `0.500` | `0.286` | `0.833` | `0.000` | `0.400` | `0.000` |
| `reference:oracle_flags_mapper` | `1` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` | `0.000` | `1.000` | `1.000` | `0.000` |
| `reference:structured_rule_sensor_teacher` | `1` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` | `0.000` | `0.000` | `1.000` | `0.000` |
| `smoke:direct_evidence_char_ngram_mlp` | `1` | `0.829` | `0.943` | `1.000` | `1.000` | `1.000` | `0.500` | `0.057` | `0.000` | `0.000` | `0.886` | `0.200` |
| `smoke:direct_evidence_word_ngram_mlp` | `1` | `0.743` | `0.914` | `1.000` | `1.000` | `1.000` | `0.250` | `0.086` | `0.000` | `0.000` | `0.886` | `0.000` |
| `smoke:scope_stack_char_ngram_mlp` | `1` | `0.829` | `0.943` | `1.000` | `1.000` | `1.000` | `0.500` | `0.057` | `0.000` | `0.914` | `0.914` | `0.200` |
| `smoke:scope_stack_word_ngram_mlp` | `1` | `0.700` | `0.914` | `1.000` | `1.000` | `0.750` | `0.500` | `0.057` | `0.000` | `0.857` | `0.857` | `0.000` |

## Phenomenon Metrics

| Stage:Model | Phenomenon | Action | False Commit | Evidence Band | Scope Flag |
|---|---|---:|---:|---:|---:|
| `confirm:direct_evidence_char_ngram_mlp` | `ambiguous` | `0.800` | `0.200` | `0.333` | `0.000` |
| `confirm:direct_evidence_char_ngram_mlp` | `correction` | `1.000` | `0.000` | `1.000` | `0.000` |
| `confirm:direct_evidence_char_ngram_mlp` | `known` | `1.000` | `0.000` | `1.000` | `0.000` |
| `confirm:direct_evidence_char_ngram_mlp` | `mention_trap` | `1.000` | `0.000` | `1.000` | `0.000` |
| `confirm:direct_evidence_char_ngram_mlp` | `multi_step_unsupported` | `1.000` | `0.000` | `1.000` | `0.000` |
| `confirm:direct_evidence_char_ngram_mlp` | `negation` | `1.000` | `0.000` | `1.000` | `0.000` |
| `confirm:direct_evidence_char_ngram_mlp` | `strict_unseen_synonym` | `0.200` | `0.000` | `0.780` | `0.000` |
| `confirm:direct_evidence_char_ngram_mlp` | `substring_trap` | `1.000` | `0.000` | `1.000` | `0.000` |
| `confirm:direct_evidence_char_ngram_mlp` | `surface_variation` | `1.000` | `0.000` | `1.000` | `0.000` |
| `confirm:direct_evidence_char_ngram_mlp` | `unknown` | `1.000` | `0.000` | `1.000` | `0.000` |
| `confirm:direct_evidence_char_ngram_mlp` | `weak` | `0.467` | `0.533` | `0.467` | `0.000` |
| `confirm:scope_stack_char_ngram_mlp` | `ambiguous` | `0.633` | `0.367` | `0.333` | `0.333` |
| `confirm:scope_stack_char_ngram_mlp` | `correction` | `0.960` | `0.000` | `0.960` | `0.960` |
| `confirm:scope_stack_char_ngram_mlp` | `known` | `1.000` | `0.000` | `1.000` | `1.000` |
| `confirm:scope_stack_char_ngram_mlp` | `mention_trap` | `1.000` | `0.000` | `1.000` | `1.000` |
| `confirm:scope_stack_char_ngram_mlp` | `multi_step_unsupported` | `1.000` | `0.000` | `1.000` | `1.000` |
| `confirm:scope_stack_char_ngram_mlp` | `negation` | `1.000` | `0.000` | `1.000` | `1.000` |
| `confirm:scope_stack_char_ngram_mlp` | `strict_unseen_synonym` | `0.200` | `0.000` | `0.520` | `0.520` |
| `confirm:scope_stack_char_ngram_mlp` | `substring_trap` | `1.000` | `0.000` | `1.000` | `1.000` |
| `confirm:scope_stack_char_ngram_mlp` | `surface_variation` | `1.000` | `0.000` | `1.000` | `1.000` |
| `confirm:scope_stack_char_ngram_mlp` | `unknown` | `1.000` | `0.000` | `1.000` | `1.000` |
| `confirm:scope_stack_char_ngram_mlp` | `weak` | `0.667` | `0.333` | `0.667` | `0.667` |
| `matrix:direct_evidence_char_ngram_mlp` | `ambiguous` | `0.889` | `0.111` | `0.333` | `0.000` |
| `matrix:direct_evidence_char_ngram_mlp` | `correction` | `1.000` | `0.000` | `1.000` | `0.000` |
| `matrix:direct_evidence_char_ngram_mlp` | `known` | `1.000` | `0.000` | `1.000` | `0.000` |
| `matrix:direct_evidence_char_ngram_mlp` | `mention_trap` | `1.000` | `0.000` | `1.000` | `0.000` |
| `matrix:direct_evidence_char_ngram_mlp` | `multi_step_unsupported` | `1.000` | `0.000` | `1.000` | `0.000` |
| `matrix:direct_evidence_char_ngram_mlp` | `negation` | `1.000` | `0.000` | `1.000` | `0.000` |
| `matrix:direct_evidence_char_ngram_mlp` | `strict_unseen_synonym` | `0.200` | `0.000` | `0.800` | `0.000` |
| `matrix:direct_evidence_char_ngram_mlp` | `substring_trap` | `1.000` | `0.000` | `1.000` | `0.000` |
| `matrix:direct_evidence_char_ngram_mlp` | `surface_variation` | `1.000` | `0.000` | `1.000` | `0.000` |
| `matrix:direct_evidence_char_ngram_mlp` | `unknown` | `1.000` | `0.000` | `1.000` | `0.000` |
| `matrix:direct_evidence_char_ngram_mlp` | `weak` | `0.444` | `0.556` | `0.444` | `0.000` |
| `matrix:direct_evidence_word_ngram_mlp` | `ambiguous` | `0.667` | `0.333` | `0.333` | `0.000` |
| `matrix:direct_evidence_word_ngram_mlp` | `correction` | `1.000` | `0.000` | `1.000` | `0.000` |
| `matrix:direct_evidence_word_ngram_mlp` | `known` | `1.000` | `0.000` | `1.000` | `0.000` |
| `matrix:direct_evidence_word_ngram_mlp` | `mention_trap` | `1.000` | `0.000` | `1.000` | `0.000` |
| `matrix:direct_evidence_word_ngram_mlp` | `multi_step_unsupported` | `1.000` | `0.000` | `1.000` | `0.000` |
| `matrix:direct_evidence_word_ngram_mlp` | `negation` | `1.000` | `0.000` | `1.000` | `0.000` |
| `matrix:direct_evidence_word_ngram_mlp` | `strict_unseen_synonym` | `0.000` | `0.000` | `1.000` | `0.000` |
| `matrix:direct_evidence_word_ngram_mlp` | `substring_trap` | `1.000` | `0.000` | `1.000` | `0.000` |
| `matrix:direct_evidence_word_ngram_mlp` | `surface_variation` | `1.000` | `0.000` | `1.000` | `0.000` |
| `matrix:direct_evidence_word_ngram_mlp` | `unknown` | `1.000` | `0.000` | `1.000` | `0.000` |
| `matrix:direct_evidence_word_ngram_mlp` | `weak` | `0.333` | `0.667` | `0.333` | `0.000` |
| `matrix:scope_stack_char_ngram_mlp` | `ambiguous` | `0.556` | `0.444` | `0.333` | `0.333` |
| `matrix:scope_stack_char_ngram_mlp` | `correction` | `1.000` | `0.000` | `1.000` | `1.000` |
| `matrix:scope_stack_char_ngram_mlp` | `known` | `1.000` | `0.000` | `1.000` | `1.000` |
| `matrix:scope_stack_char_ngram_mlp` | `mention_trap` | `1.000` | `0.000` | `1.000` | `1.000` |
| `matrix:scope_stack_char_ngram_mlp` | `multi_step_unsupported` | `1.000` | `0.000` | `1.000` | `1.000` |
| `matrix:scope_stack_char_ngram_mlp` | `negation` | `1.000` | `0.000` | `1.000` | `1.000` |
| `matrix:scope_stack_char_ngram_mlp` | `strict_unseen_synonym` | `0.200` | `0.000` | `0.267` | `0.267` |
| `matrix:scope_stack_char_ngram_mlp` | `substring_trap` | `1.000` | `0.000` | `1.000` | `1.000` |
| `matrix:scope_stack_char_ngram_mlp` | `surface_variation` | `1.000` | `0.000` | `1.000` | `1.000` |
| `matrix:scope_stack_char_ngram_mlp` | `unknown` | `1.000` | `0.000` | `1.000` | `1.000` |
| `matrix:scope_stack_char_ngram_mlp` | `weak` | `0.667` | `0.333` | `0.667` | `0.667` |
| `matrix:scope_stack_word_ngram_mlp` | `ambiguous` | `0.778` | `0.222` | `0.333` | `0.333` |
| `matrix:scope_stack_word_ngram_mlp` | `correction` | `0.800` | `0.000` | `0.800` | `0.800` |
| `matrix:scope_stack_word_ngram_mlp` | `known` | `1.000` | `0.000` | `1.000` | `1.000` |
| `matrix:scope_stack_word_ngram_mlp` | `mention_trap` | `1.000` | `0.000` | `1.000` | `1.000` |
| `matrix:scope_stack_word_ngram_mlp` | `multi_step_unsupported` | `1.000` | `0.000` | `1.000` | `1.000` |
| `matrix:scope_stack_word_ngram_mlp` | `negation` | `1.000` | `0.000` | `1.000` | `1.000` |
| `matrix:scope_stack_word_ngram_mlp` | `strict_unseen_synonym` | `0.000` | `0.000` | `1.000` | `1.000` |
| `matrix:scope_stack_word_ngram_mlp` | `substring_trap` | `1.000` | `0.000` | `1.000` | `1.000` |
| `matrix:scope_stack_word_ngram_mlp` | `surface_variation` | `1.000` | `0.000` | `1.000` | `1.000` |
| `matrix:scope_stack_word_ngram_mlp` | `unknown` | `1.000` | `0.000` | `1.000` | `1.000` |
| `matrix:scope_stack_word_ngram_mlp` | `weak` | `0.333` | `0.667` | `0.333` | `0.333` |
| `reference:keyword_sensor` | `ambiguous` | `1.000` | `0.000` | `0.667` | `0.000` |
| `reference:keyword_sensor` | `correction` | `0.000` | `0.000` | `0.000` | `0.000` |
| `reference:keyword_sensor` | `known` | `1.000` | `0.000` | `1.000` | `0.000` |
| `reference:keyword_sensor` | `mention_trap` | `0.000` | `1.000` | `0.000` | `0.000` |
| `reference:keyword_sensor` | `multi_step_unsupported` | `1.000` | `0.000` | `1.000` | `0.000` |
| `reference:keyword_sensor` | `negation` | `0.286` | `0.286` | `0.000` | `0.000` |
| `reference:keyword_sensor` | `strict_unseen_synonym` | `0.000` | `0.000` | `1.000` | `0.000` |
| `reference:keyword_sensor` | `substring_trap` | `0.500` | `0.500` | `0.500` | `0.000` |
| `reference:keyword_sensor` | `surface_variation` | `1.000` | `0.000` | `1.000` | `0.000` |
| `reference:keyword_sensor` | `unknown` | `1.000` | `0.000` | `1.000` | `0.000` |
| `reference:keyword_sensor` | `weak` | `0.000` | `1.000` | `0.000` | `0.000` |
| `reference:oracle_flags_mapper` | `ambiguous` | `1.000` | `0.000` | `1.000` | `1.000` |
| `reference:oracle_flags_mapper` | `correction` | `1.000` | `0.000` | `1.000` | `1.000` |
| `reference:oracle_flags_mapper` | `known` | `1.000` | `0.000` | `1.000` | `1.000` |
| `reference:oracle_flags_mapper` | `mention_trap` | `1.000` | `0.000` | `1.000` | `1.000` |
| `reference:oracle_flags_mapper` | `multi_step_unsupported` | `1.000` | `0.000` | `1.000` | `1.000` |
| `reference:oracle_flags_mapper` | `negation` | `1.000` | `0.000` | `1.000` | `1.000` |
| `reference:oracle_flags_mapper` | `strict_unseen_synonym` | `0.000` | `0.000` | `1.000` | `1.000` |
| `reference:oracle_flags_mapper` | `substring_trap` | `1.000` | `0.000` | `1.000` | `1.000` |
| `reference:oracle_flags_mapper` | `surface_variation` | `1.000` | `0.000` | `1.000` | `1.000` |
| `reference:oracle_flags_mapper` | `unknown` | `1.000` | `0.000` | `1.000` | `1.000` |
| `reference:oracle_flags_mapper` | `weak` | `1.000` | `0.000` | `1.000` | `1.000` |
| `reference:structured_rule_sensor_teacher` | `ambiguous` | `1.000` | `0.000` | `1.000` | `0.000` |
| `reference:structured_rule_sensor_teacher` | `correction` | `1.000` | `0.000` | `1.000` | `0.000` |
| `reference:structured_rule_sensor_teacher` | `known` | `1.000` | `0.000` | `1.000` | `0.000` |
| `reference:structured_rule_sensor_teacher` | `mention_trap` | `1.000` | `0.000` | `1.000` | `0.000` |
| `reference:structured_rule_sensor_teacher` | `multi_step_unsupported` | `1.000` | `0.000` | `1.000` | `0.000` |
| `reference:structured_rule_sensor_teacher` | `negation` | `1.000` | `0.000` | `1.000` | `0.000` |
| `reference:structured_rule_sensor_teacher` | `strict_unseen_synonym` | `0.000` | `0.000` | `1.000` | `0.000` |
| `reference:structured_rule_sensor_teacher` | `substring_trap` | `1.000` | `0.000` | `1.000` | `0.000` |
| `reference:structured_rule_sensor_teacher` | `surface_variation` | `1.000` | `0.000` | `1.000` | `0.000` |
| `reference:structured_rule_sensor_teacher` | `unknown` | `1.000` | `0.000` | `1.000` | `0.000` |
| `reference:structured_rule_sensor_teacher` | `weak` | `1.000` | `0.000` | `1.000` | `0.000` |
| `smoke:direct_evidence_char_ngram_mlp` | `ambiguous` | `1.000` | `0.000` | `0.333` | `0.000` |
| `smoke:direct_evidence_char_ngram_mlp` | `correction` | `1.000` | `0.000` | `1.000` | `0.000` |
| `smoke:direct_evidence_char_ngram_mlp` | `known` | `1.000` | `0.000` | `1.000` | `0.000` |
| `smoke:direct_evidence_char_ngram_mlp` | `mention_trap` | `1.000` | `0.000` | `1.000` | `0.000` |
| `smoke:direct_evidence_char_ngram_mlp` | `multi_step_unsupported` | `1.000` | `0.000` | `1.000` | `0.000` |
| `smoke:direct_evidence_char_ngram_mlp` | `negation` | `1.000` | `0.000` | `1.000` | `0.000` |
| `smoke:direct_evidence_char_ngram_mlp` | `strict_unseen_synonym` | `0.200` | `0.000` | `0.800` | `0.000` |
| `smoke:direct_evidence_char_ngram_mlp` | `substring_trap` | `1.000` | `0.000` | `1.000` | `0.000` |
| `smoke:direct_evidence_char_ngram_mlp` | `surface_variation` | `1.000` | `0.000` | `1.000` | `0.000` |
| `smoke:direct_evidence_char_ngram_mlp` | `unknown` | `1.000` | `0.000` | `1.000` | `0.000` |
| `smoke:direct_evidence_char_ngram_mlp` | `weak` | `0.333` | `0.667` | `0.333` | `0.000` |
| `smoke:direct_evidence_word_ngram_mlp` | `ambiguous` | `0.667` | `0.333` | `0.333` | `0.000` |
| `smoke:direct_evidence_word_ngram_mlp` | `correction` | `1.000` | `0.000` | `1.000` | `0.000` |
| `smoke:direct_evidence_word_ngram_mlp` | `known` | `1.000` | `0.000` | `1.000` | `0.000` |
| `smoke:direct_evidence_word_ngram_mlp` | `mention_trap` | `1.000` | `0.000` | `1.000` | `0.000` |
| `smoke:direct_evidence_word_ngram_mlp` | `multi_step_unsupported` | `1.000` | `0.000` | `1.000` | `0.000` |
| `smoke:direct_evidence_word_ngram_mlp` | `negation` | `1.000` | `0.000` | `1.000` | `0.000` |
| `smoke:direct_evidence_word_ngram_mlp` | `strict_unseen_synonym` | `0.000` | `0.000` | `1.000` | `0.000` |
| `smoke:direct_evidence_word_ngram_mlp` | `substring_trap` | `1.000` | `0.000` | `1.000` | `0.000` |
| `smoke:direct_evidence_word_ngram_mlp` | `surface_variation` | `1.000` | `0.000` | `1.000` | `0.000` |
| `smoke:direct_evidence_word_ngram_mlp` | `unknown` | `1.000` | `0.000` | `1.000` | `0.000` |
| `smoke:direct_evidence_word_ngram_mlp` | `weak` | `0.333` | `0.667` | `0.333` | `0.000` |
| `smoke:scope_stack_char_ngram_mlp` | `ambiguous` | `0.667` | `0.333` | `0.333` | `0.333` |
| `smoke:scope_stack_char_ngram_mlp` | `correction` | `1.000` | `0.000` | `1.000` | `1.000` |
| `smoke:scope_stack_char_ngram_mlp` | `known` | `1.000` | `0.000` | `1.000` | `1.000` |
| `smoke:scope_stack_char_ngram_mlp` | `mention_trap` | `1.000` | `0.000` | `1.000` | `1.000` |
| `smoke:scope_stack_char_ngram_mlp` | `multi_step_unsupported` | `1.000` | `0.000` | `1.000` | `1.000` |
| `smoke:scope_stack_char_ngram_mlp` | `negation` | `1.000` | `0.000` | `1.000` | `1.000` |
| `smoke:scope_stack_char_ngram_mlp` | `strict_unseen_synonym` | `0.200` | `0.000` | `0.200` | `0.200` |
| `smoke:scope_stack_char_ngram_mlp` | `substring_trap` | `1.000` | `0.000` | `1.000` | `1.000` |
| `smoke:scope_stack_char_ngram_mlp` | `surface_variation` | `1.000` | `0.000` | `1.000` | `1.000` |
| `smoke:scope_stack_char_ngram_mlp` | `unknown` | `1.000` | `0.000` | `1.000` | `1.000` |
| `smoke:scope_stack_char_ngram_mlp` | `weak` | `0.667` | `0.333` | `0.667` | `0.667` |
| `smoke:scope_stack_word_ngram_mlp` | `ambiguous` | `1.000` | `0.000` | `0.333` | `0.333` |
| `smoke:scope_stack_word_ngram_mlp` | `correction` | `0.800` | `0.000` | `0.800` | `0.800` |
| `smoke:scope_stack_word_ngram_mlp` | `known` | `1.000` | `0.000` | `1.000` | `1.000` |
| `smoke:scope_stack_word_ngram_mlp` | `mention_trap` | `1.000` | `0.000` | `1.000` | `1.000` |
| `smoke:scope_stack_word_ngram_mlp` | `multi_step_unsupported` | `1.000` | `0.000` | `1.000` | `1.000` |
| `smoke:scope_stack_word_ngram_mlp` | `negation` | `1.000` | `0.000` | `1.000` | `1.000` |
| `smoke:scope_stack_word_ngram_mlp` | `strict_unseen_synonym` | `0.000` | `0.000` | `1.000` | `1.000` |
| `smoke:scope_stack_word_ngram_mlp` | `substring_trap` | `1.000` | `0.000` | `1.000` | `1.000` |
| `smoke:scope_stack_word_ngram_mlp` | `surface_variation` | `1.000` | `0.000` | `1.000` | `1.000` |
| `smoke:scope_stack_word_ngram_mlp` | `unknown` | `1.000` | `0.000` | `1.000` | `1.000` |
| `smoke:scope_stack_word_ngram_mlp` | `weak` | `0.333` | `0.667` | `0.333` | `0.333` |

## Verdict

```json
{
  "confirm:direct_evidence_char_ngram_mlp": [
    "DIRECT_EVIDENCE_STILL_WEAK",
    "WEAK_AMBIGUOUS_BOTTLENECK",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED",
    "DISTILL_RULE_TEACHER_ONLY",
    "LEARNED_SCOPE_SENSOR_BOTTLENECK"
  ],
  "confirm:scope_stack_char_ngram_mlp": [
    "SCOPE_STACK_NEAR_PASS_WEAK_AMBIGUOUS_BOTTLENECK",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED",
    "DISTILL_RULE_TEACHER_ONLY",
    "LEARNED_SCOPE_SENSOR_BOTTLENECK"
  ],
  "global": [
    "LEARNED_SENSOR_NEAR_PASS",
    "WEAK_AMBIGUOUS_FALSE_COMMIT_BOTTLENECK",
    "SCOPE_STACK_NO_BETTER_THAN_DIRECT"
  ],
  "matrix:direct_evidence_char_ngram_mlp": [
    "DIRECT_EVIDENCE_STILL_WEAK",
    "WEAK_AMBIGUOUS_BOTTLENECK",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED",
    "DISTILL_RULE_TEACHER_ONLY",
    "LEARNED_SCOPE_SENSOR_BOTTLENECK"
  ],
  "matrix:direct_evidence_word_ngram_mlp": [
    "DIRECT_EVIDENCE_STILL_WEAK",
    "WEAK_AMBIGUOUS_BOTTLENECK",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED",
    "DISTILL_RULE_TEACHER_ONLY",
    "LEARNED_SCOPE_SENSOR_BOTTLENECK"
  ],
  "matrix:scope_stack_char_ngram_mlp": [
    "SCOPE_STACK_NEAR_PASS_WEAK_AMBIGUOUS_BOTTLENECK",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED",
    "DISTILL_RULE_TEACHER_ONLY",
    "LEARNED_SCOPE_SENSOR_BOTTLENECK"
  ],
  "matrix:scope_stack_word_ngram_mlp": [
    "STRICT_UNSEEN_SYNONYM_UNSOLVED",
    "DISTILL_RULE_TEACHER_ONLY",
    "LEARNED_SCOPE_SENSOR_BOTTLENECK"
  ],
  "reference:keyword_sensor": [
    "STRICT_UNSEEN_SYNONYM_UNSOLVED"
  ],
  "reference:oracle_flags_mapper": [
    "ORACLE_FLAGS_MAPPER_PASS",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED"
  ],
  "reference:structured_rule_sensor_teacher": [
    "TEACHER_REFERENCE_PASS",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED"
  ],
  "smoke:direct_evidence_char_ngram_mlp": [
    "DIRECT_EVIDENCE_STILL_WEAK",
    "WEAK_AMBIGUOUS_BOTTLENECK",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED",
    "DISTILL_RULE_TEACHER_ONLY",
    "LEARNED_SCOPE_SENSOR_BOTTLENECK"
  ],
  "smoke:direct_evidence_word_ngram_mlp": [
    "DIRECT_EVIDENCE_STILL_WEAK",
    "WEAK_AMBIGUOUS_BOTTLENECK",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED",
    "DISTILL_RULE_TEACHER_ONLY",
    "LEARNED_SCOPE_SENSOR_BOTTLENECK"
  ],
  "smoke:scope_stack_char_ngram_mlp": [
    "SCOPE_STACK_NEAR_PASS_WEAK_AMBIGUOUS_BOTTLENECK",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED",
    "DISTILL_RULE_TEACHER_ONLY",
    "LEARNED_SCOPE_SENSOR_BOTTLENECK"
  ],
  "smoke:scope_stack_word_ngram_mlp": [
    "SCOPE_STACK_NEAR_PASS_WEAK_AMBIGUOUS_BOTTLENECK",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED",
    "DISTILL_RULE_TEACHER_ONLY",
    "LEARNED_SCOPE_SENSOR_BOTTLENECK"
  ]
}
```

## Failure Examples

- `reference:oracle_flags_mapper` seed `-1` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: increment by 3 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `reference:oracle_flags_mapper` seed `-1` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: raise the value by 3 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `reference:oracle_flags_mapper` seed `-1` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: product with 3 -> expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `reference:oracle_flags_mapper` seed `-1` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: halve it -> expected `REJECT_UNKNOWN`, got `HOLD_ASK_RESEARCH` (over_hold).
- `reference:oracle_flags_mapper` seed `-1` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: exponentiate by 3 -> expected `REJECT_UNKNOWN`, got `HOLD_ASK_RESEARCH` (over_hold).
- `smoke:direct_evidence_word_ngram_mlp` seed `0` `heldout_weak_ambiguous/weak`: probably multiply by 3 -> expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL` (false_commit).
- `smoke:direct_evidence_word_ngram_mlp` seed `0` `heldout_weak_ambiguous/weak`: it could be add -> expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD` (false_commit).
- `smoke:direct_evidence_word_ngram_mlp` seed `0` `heldout_weak_ambiguous/ambiguous`: add, multiply, or divide by 3 -> expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL` (false_commit).
- `smoke:direct_evidence_word_ngram_mlp` seed `0` `heldout_weak_ambiguous/ambiguous`: maybe plus, maybe times 3 -> expected `HOLD_ASK_RESEARCH`, got `HOLD_ASK_RESEARCH` (evidence_mapping_error).
- `smoke:direct_evidence_word_ngram_mlp` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: increment by 3 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `smoke:direct_evidence_word_ngram_mlp` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: raise the value by 3 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `smoke:direct_evidence_word_ngram_mlp` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: product with 3 -> expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `smoke:direct_evidence_word_ngram_mlp` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: halve it -> expected `REJECT_UNKNOWN`, got `HOLD_ASK_RESEARCH` (over_hold).
- `smoke:direct_evidence_word_ngram_mlp` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: exponentiate by 3 -> expected `REJECT_UNKNOWN`, got `HOLD_ASK_RESEARCH` (over_hold).
- `smoke:direct_evidence_char_ngram_mlp` seed `0` `heldout_weak_ambiguous/weak`: probably multiply by 3 -> expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL` (false_commit).
- `smoke:direct_evidence_char_ngram_mlp` seed `0` `heldout_weak_ambiguous/weak`: it could be add -> expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD` (false_commit).
- `smoke:direct_evidence_char_ngram_mlp` seed `0` `heldout_weak_ambiguous/ambiguous`: add, multiply, or divide by 3 -> expected `HOLD_ASK_RESEARCH`, got `HOLD_ASK_RESEARCH` (evidence_mapping_error).
- `smoke:direct_evidence_char_ngram_mlp` seed `0` `heldout_weak_ambiguous/ambiguous`: maybe plus, maybe times 3 -> expected `HOLD_ASK_RESEARCH`, got `HOLD_ASK_RESEARCH` (evidence_mapping_error).
- `smoke:direct_evidence_char_ngram_mlp` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: increment by 3 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `smoke:direct_evidence_char_ngram_mlp` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: raise the value by 3 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `smoke:direct_evidence_char_ngram_mlp` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: product with 3 -> expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `smoke:direct_evidence_char_ngram_mlp` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: halve it -> expected `REJECT_UNKNOWN`, got `HOLD_ASK_RESEARCH` (over_hold).
- `smoke:direct_evidence_char_ngram_mlp` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: exponentiate by 3 -> expected `REJECT_UNKNOWN`, got `REJECT_UNKNOWN` (evidence_mapping_error).
- `smoke:scope_stack_word_ngram_mlp` seed `0` `heldout_correction/correction`: add 3. wait, actually multiply by 3 -> expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH` (missed_execute).
- `smoke:scope_stack_word_ngram_mlp` seed `0` `heldout_weak_ambiguous/weak`: probably multiply by 3 -> expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL` (false_commit).
- `smoke:scope_stack_word_ngram_mlp` seed `0` `heldout_weak_ambiguous/weak`: it could be add -> expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD` (false_commit).
- `smoke:scope_stack_word_ngram_mlp` seed `0` `heldout_weak_ambiguous/ambiguous`: add, multiply, or divide by 3 -> expected `HOLD_ASK_RESEARCH`, got `HOLD_ASK_RESEARCH` (scope_error).
- `smoke:scope_stack_word_ngram_mlp` seed `0` `heldout_weak_ambiguous/ambiguous`: maybe plus, maybe times 3 -> expected `HOLD_ASK_RESEARCH`, got `HOLD_ASK_RESEARCH` (scope_error).
- `smoke:scope_stack_word_ngram_mlp` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: increment by 3 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `smoke:scope_stack_word_ngram_mlp` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: raise the value by 3 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `smoke:scope_stack_word_ngram_mlp` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: product with 3 -> expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `smoke:scope_stack_word_ngram_mlp` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: halve it -> expected `REJECT_UNKNOWN`, got `HOLD_ASK_RESEARCH` (over_hold).
- `smoke:scope_stack_word_ngram_mlp` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: exponentiate by 3 -> expected `REJECT_UNKNOWN`, got `HOLD_ASK_RESEARCH` (over_hold).
- `smoke:scope_stack_char_ngram_mlp` seed `0` `heldout_weak_ambiguous/weak`: probably multiply by 3 -> expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL` (false_commit).
- `smoke:scope_stack_char_ngram_mlp` seed `0` `heldout_weak_ambiguous/ambiguous`: add, multiply, or divide by 3 -> expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL` (false_commit).
- `smoke:scope_stack_char_ngram_mlp` seed `0` `heldout_weak_ambiguous/ambiguous`: maybe plus, maybe times 3 -> expected `HOLD_ASK_RESEARCH`, got `HOLD_ASK_RESEARCH` (scope_error).
- `smoke:scope_stack_char_ngram_mlp` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: increment by 3 -> expected `EXEC_ADD`, got `REJECT_UNKNOWN` (synonym_gap).
- `smoke:scope_stack_char_ngram_mlp` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: raise the value by 3 -> expected `EXEC_ADD`, got `REJECT_UNKNOWN` (synonym_gap).
- `smoke:scope_stack_char_ngram_mlp` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: product with 3 -> expected `EXEC_MUL`, got `EXEC_ADD` (scope_error).
- `smoke:scope_stack_char_ngram_mlp` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: halve it -> expected `REJECT_UNKNOWN`, got `HOLD_ASK_RESEARCH` (over_hold).
- ... 276 more in `failure_examples.jsonl`.

## Interpretation

A scope-stack positive means decomposition into learned flags plus deterministic evidence mapping is sufficient in this toy command sensor setting.

If teacher and oracle mapper pass but learned arms fail, the bottleneck is learned scope/event extraction rather than the guard or evidence mapper.

Strict unseen synonyms are diagnostic only because the setup has no pretrained semantics.

## Claim Boundary

No general NLU, full PilotPulse integration, production VRAXION/INSTNCT, or consciousness claim.

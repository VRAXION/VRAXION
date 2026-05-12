# PILOT_SENSOR_STRUCTURED_FEATURES_001 Result

## Goal

Test whether explicit normalized scope/event features solve the factor-heldout failures that raw n-gram learned sensors missed.

## Setup

- Uses the same factor-heldout train/eval split as `PILOT_SENSOR_FACTOR_HELDOUT_001`.
- Replaces raw text n-grams with explicit scope/event flags: cue flags, weak/ambiguity markers, mention-only, negation, correction targets, and multi-step unsupported.
- A linear student maps these features to evidence bands; action still comes from the fixed guard.

## Aggregate Metrics

| Model | Seeds | Score | Action | Weak/Amb | False Commit | Mention | Neg | Corr | Scope Flags | Strict Syn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `keyword_sensor` | `1` | `-5.088` | `0.362` | `0.250` | `0.383` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` |
| `oracle_flags_mapper` | `1` | `1.000` | `1.000` | `1.000` | `0.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` |
| `structured_feature_linear_student` | `10` | `0.898` | `0.966` | `0.900` | `0.034` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` |
| `structured_rule_sensor_teacher` | `1` | `1.000` | `1.000` | `1.000` | `0.000` | `1.000` | `1.000` | `1.000` | `0.000` | `0.000` |

## Verdict

```json
{
  "global": [
    "EXPLICIT_SCOPE_FEATURES_NOT_SUFFICIENT"
  ],
  "keyword_sensor": [
    "KEYWORD_BASELINE_FAILS",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED"
  ],
  "oracle_flags_mapper": [
    "ORACLE_FLAGS_MAPPER_PASS",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED"
  ],
  "structured_feature_linear_student": [
    "STRUCTURED_FEATURES_WEAK",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED"
  ],
  "structured_rule_sensor_teacher": [
    "TEACHER_REFERENCE_PASS",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED"
  ]
}
```

## Failure Examples

- `oracle_flags_mapper` seed `-1` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: increment by 9 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `oracle_flags_mapper` seed `-1` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: product with 9 -> expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `structured_feature_linear_student` seed `0` `heldout_factor/ambiguous`: divide, add, or multiply by 7 -> expected `HOLD_ASK_RESEARCH`, got `HOLD_ASK_RESEARCH` (evidence_mapping_error).
- `structured_feature_linear_student` seed `0` `heldout_factor/ambiguous`: divide, add, or multiply by 8 -> expected `HOLD_ASK_RESEARCH`, got `HOLD_ASK_RESEARCH` (evidence_mapping_error).
- `structured_feature_linear_student` seed `0` `heldout_factor/ambiguous`: divide, add, or multiply by 9 -> expected `HOLD_ASK_RESEARCH`, got `HOLD_ASK_RESEARCH` (evidence_mapping_error).
- `structured_feature_linear_student` seed `0` `heldout_factor/ambiguous`: divide, add, or multiply by 11 -> expected `HOLD_ASK_RESEARCH`, got `HOLD_ASK_RESEARCH` (evidence_mapping_error).
- `structured_feature_linear_student` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: increment by 9 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `structured_feature_linear_student` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: product with 9 -> expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `structured_feature_linear_student` seed `1` `heldout_factor/ambiguous`: divide, add, or multiply by 7 -> expected `HOLD_ASK_RESEARCH`, got `HOLD_ASK_RESEARCH` (evidence_mapping_error).
- `structured_feature_linear_student` seed `1` `heldout_factor/ambiguous`: divide, add, or multiply by 8 -> expected `HOLD_ASK_RESEARCH`, got `HOLD_ASK_RESEARCH` (evidence_mapping_error).
- `structured_feature_linear_student` seed `1` `heldout_factor/ambiguous`: divide, add, or multiply by 9 -> expected `HOLD_ASK_RESEARCH`, got `HOLD_ASK_RESEARCH` (evidence_mapping_error).
- `structured_feature_linear_student` seed `1` `heldout_factor/ambiguous`: divide, add, or multiply by 11 -> expected `HOLD_ASK_RESEARCH`, got `HOLD_ASK_RESEARCH` (evidence_mapping_error).
- `structured_feature_linear_student` seed `1` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: increment by 9 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `structured_feature_linear_student` seed `1` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: product with 9 -> expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `structured_feature_linear_student` seed `2` `heldout_factor/ambiguous`: divide, add, or multiply by 7 -> expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL` (false_commit).
- `structured_feature_linear_student` seed `2` `heldout_factor/ambiguous`: divide, add, or multiply by 8 -> expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL` (false_commit).
- `structured_feature_linear_student` seed `2` `heldout_factor/ambiguous`: divide, add, or multiply by 9 -> expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL` (false_commit).
- `structured_feature_linear_student` seed `2` `heldout_factor/ambiguous`: divide, add, or multiply by 11 -> expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL` (false_commit).
- `structured_feature_linear_student` seed `2` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: increment by 9 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `structured_feature_linear_student` seed `2` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: product with 9 -> expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `structured_feature_linear_student` seed `3` `heldout_factor/ambiguous`: divide, add, or multiply by 7 -> expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD` (false_commit).
- `structured_feature_linear_student` seed `3` `heldout_factor/ambiguous`: divide, add, or multiply by 8 -> expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD` (false_commit).
- `structured_feature_linear_student` seed `3` `heldout_factor/ambiguous`: divide, add, or multiply by 9 -> expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD` (false_commit).
- `structured_feature_linear_student` seed `3` `heldout_factor/ambiguous`: divide, add, or multiply by 11 -> expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD` (false_commit).
- `structured_feature_linear_student` seed `3` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: increment by 9 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `structured_feature_linear_student` seed `3` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: product with 9 -> expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `structured_feature_linear_student` seed `4` `heldout_factor/ambiguous`: divide, add, or multiply by 7 -> expected `HOLD_ASK_RESEARCH`, got `HOLD_ASK_RESEARCH` (evidence_mapping_error).
- `structured_feature_linear_student` seed `4` `heldout_factor/ambiguous`: divide, add, or multiply by 8 -> expected `HOLD_ASK_RESEARCH`, got `HOLD_ASK_RESEARCH` (evidence_mapping_error).
- `structured_feature_linear_student` seed `4` `heldout_factor/ambiguous`: divide, add, or multiply by 9 -> expected `HOLD_ASK_RESEARCH`, got `HOLD_ASK_RESEARCH` (evidence_mapping_error).
- `structured_feature_linear_student` seed `4` `heldout_factor/ambiguous`: divide, add, or multiply by 11 -> expected `HOLD_ASK_RESEARCH`, got `HOLD_ASK_RESEARCH` (evidence_mapping_error).
- ... 32 more in `failure_examples.jsonl`.

## Interpretation

A positive result means the factor-heldout bottleneck is the raw text-to-scope feature extractor, not the guard or evidence interface.
It does not mean the parser is learned from raw text; this is a parser-assisted feature interface.

Strict unseen synonym remains diagnostic only.

## Claim Boundary

No general NLU, full PilotPulse integration, production VRAXION/INSTNCT, or consciousness claim.

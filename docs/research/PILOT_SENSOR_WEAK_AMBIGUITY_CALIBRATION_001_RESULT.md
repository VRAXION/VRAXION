# PILOT_SENSOR_WEAK_AMBIGUITY_CALIBRATION_001 Result

## Goal

Test whether targeted weak/ambiguous training coverage reduces learned sensor false commits without changing the fixed guard.

## Setup

- Reuses `PILOT_SENSOR_SCOPE_STACK_NIGHTLY_001` direct and scope-stack MLP arms.
- Compares original training cases against augmented weak/ambiguous templates.
- Action always comes from predicted evidence through the fixed guard.

## Aggregate Metrics

| Model | Seeds | Score | Action | Weak/Amb | False Commit | Scope | Neg | Corr | Evidence Band | Strict Syn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `direct_evidence_char_ngram_mlp_augmented` | `10` | `1.000` | `1.000` | `1.000` | `0.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.200` |
| `direct_evidence_char_ngram_mlp_original` | `10` | `0.811` | `0.937` | `0.450` | `0.063` | `1.000` | `1.000` | `1.000` | `0.897` | `0.200` |
| `direct_evidence_word_ngram_mlp_augmented` | `10` | `1.000` | `1.000` | `1.000` | `0.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` |
| `keyword_sensor` | `1` | `-3.819` | `0.486` | `0.500` | `0.286` | `0.200` | `0.333` | `0.000` | `0.400` | `0.000` |
| `oracle_flags_mapper` | `1` | `1.000` | `1.000` | `1.000` | `0.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` |
| `scope_stack_char_ngram_mlp_augmented` | `10` | `0.871` | `0.971` | `1.000` | `0.000` | `1.000` | `1.000` | `0.750` | `0.971` | `0.200` |
| `scope_stack_char_ngram_mlp_original` | `10` | `0.814` | `0.934` | `0.475` | `0.060` | `1.000` | `1.000` | `0.950` | `0.909` | `0.200` |
| `scope_stack_word_ngram_mlp_augmented` | `10` | `0.871` | `0.971` | `1.000` | `0.000` | `1.000` | `1.000` | `0.750` | `0.971` | `0.000` |
| `structured_rule_sensor_teacher` | `1` | `1.000` | `1.000` | `1.000` | `0.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` |

## Verdict

```json
{
  "direct_evidence_char_ngram_mlp_augmented": [
    "WEAK_AMBIGUITY_CALIBRATION_POSITIVE",
    "AUGMENTATION_HELPS",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED"
  ],
  "direct_evidence_char_ngram_mlp_original": [
    "WEAK_AMBIGUITY_STILL_BOTTLENECK",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED"
  ],
  "direct_evidence_word_ngram_mlp_augmented": [
    "WEAK_AMBIGUITY_CALIBRATION_POSITIVE",
    "AUGMENTATION_HELPS",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED"
  ],
  "global": [
    "AUGMENTATION_HELPS_WEAK_AMBIGUITY"
  ],
  "keyword_sensor": [
    "KEYWORD_BASELINE_FAILS"
  ],
  "oracle_flags_mapper": [
    "ORACLE_FLAGS_MAPPER_PASS"
  ],
  "scope_stack_char_ngram_mlp_augmented": [
    "WEAK_AMBIGUITY_CALIBRATION_PARTIAL",
    "AUGMENTATION_HELPS",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED"
  ],
  "scope_stack_char_ngram_mlp_original": [
    "WEAK_AMBIGUITY_STILL_BOTTLENECK",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED"
  ],
  "scope_stack_word_ngram_mlp_augmented": [
    "WEAK_AMBIGUITY_CALIBRATION_PARTIAL",
    "AUGMENTATION_HELPS",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED"
  ],
  "structured_rule_sensor_teacher": [
    "TEACHER_REFERENCE_PASS"
  ]
}
```

## Failure Examples

- `oracle_flags_mapper` seed `-1` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: increment by 3 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `oracle_flags_mapper` seed `-1` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: raise the value by 3 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `oracle_flags_mapper` seed `-1` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: product with 3 -> expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `oracle_flags_mapper` seed `-1` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: halve it -> expected `REJECT_UNKNOWN`, got `HOLD_ASK_RESEARCH` (over_hold).
- `oracle_flags_mapper` seed `-1` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: exponentiate by 3 -> expected `REJECT_UNKNOWN`, got `HOLD_ASK_RESEARCH` (over_hold).
- `direct_evidence_char_ngram_mlp_original` seed `0` `heldout_weak_ambiguous/weak`: probably multiply by 3 -> expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL` (false_commit).
- `direct_evidence_char_ngram_mlp_original` seed `0` `heldout_weak_ambiguous/weak`: it could be add -> expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD` (false_commit).
- `direct_evidence_char_ngram_mlp_original` seed `0` `heldout_weak_ambiguous/ambiguous`: add, multiply, or divide by 3 -> expected `HOLD_ASK_RESEARCH`, got `HOLD_ASK_RESEARCH` (evidence_mapping_error).
- `direct_evidence_char_ngram_mlp_original` seed `0` `heldout_weak_ambiguous/ambiguous`: maybe plus, maybe times 3 -> expected `HOLD_ASK_RESEARCH`, got `HOLD_ASK_RESEARCH` (evidence_mapping_error).
- `direct_evidence_char_ngram_mlp_original` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: increment by 3 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `direct_evidence_char_ngram_mlp_original` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: raise the value by 3 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `direct_evidence_char_ngram_mlp_original` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: product with 3 -> expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `direct_evidence_char_ngram_mlp_original` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: halve it -> expected `REJECT_UNKNOWN`, got `HOLD_ASK_RESEARCH` (over_hold).
- `direct_evidence_char_ngram_mlp_original` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: exponentiate by 3 -> expected `REJECT_UNKNOWN`, got `REJECT_UNKNOWN` (evidence_mapping_error).
- `scope_stack_char_ngram_mlp_original` seed `0` `heldout_weak_ambiguous/weak`: probably multiply by 3 -> expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL` (false_commit).
- `scope_stack_char_ngram_mlp_original` seed `0` `heldout_weak_ambiguous/ambiguous`: add, multiply, or divide by 3 -> expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL` (false_commit).
- `scope_stack_char_ngram_mlp_original` seed `0` `heldout_weak_ambiguous/ambiguous`: maybe plus, maybe times 3 -> expected `HOLD_ASK_RESEARCH`, got `HOLD_ASK_RESEARCH` (scope_error).
- `scope_stack_char_ngram_mlp_original` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: increment by 3 -> expected `EXEC_ADD`, got `REJECT_UNKNOWN` (synonym_gap).
- `scope_stack_char_ngram_mlp_original` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: raise the value by 3 -> expected `EXEC_ADD`, got `REJECT_UNKNOWN` (synonym_gap).
- `scope_stack_char_ngram_mlp_original` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: product with 3 -> expected `EXEC_MUL`, got `EXEC_ADD` (scope_error).
- `scope_stack_char_ngram_mlp_original` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: halve it -> expected `REJECT_UNKNOWN`, got `HOLD_ASK_RESEARCH` (over_hold).
- `scope_stack_char_ngram_mlp_original` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: exponentiate by 3 -> expected `REJECT_UNKNOWN`, got `REJECT_UNKNOWN` (scope_error).
- `direct_evidence_char_ngram_mlp_augmented` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: increment by 3 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `direct_evidence_char_ngram_mlp_augmented` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: raise the value by 3 -> expected `EXEC_ADD`, got `REJECT_UNKNOWN` (synonym_gap).
- `direct_evidence_char_ngram_mlp_augmented` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: product with 3 -> expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `direct_evidence_char_ngram_mlp_augmented` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: halve it -> expected `REJECT_UNKNOWN`, got `HOLD_ASK_RESEARCH` (over_hold).
- `direct_evidence_char_ngram_mlp_augmented` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: exponentiate by 3 -> expected `REJECT_UNKNOWN`, got `REJECT_UNKNOWN` (evidence_mapping_error).
- `scope_stack_char_ngram_mlp_augmented` seed `0` `heldout_correction/correction`: add 3. wait, actually multiply by 3 -> expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH` (missed_execute).
- `scope_stack_char_ngram_mlp_augmented` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: increment by 3 -> expected `EXEC_ADD`, got `REJECT_UNKNOWN` (synonym_gap).
- `scope_stack_char_ngram_mlp_augmented` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: raise the value by 3 -> expected `EXEC_ADD`, got `REJECT_UNKNOWN` (synonym_gap).
- ... 363 more in `failure_examples.jsonl`.

## Interpretation

If augmentation passes, the previous bottleneck was mainly train coverage for weak/ambiguous forms. If it fails, the sensor needs stronger abstention/calibration, not just more templates.

Strict unseen synonym remains diagnostic only.

## Claim Boundary

No general NLU, full PilotPulse integration, production VRAXION/INSTNCT, or consciousness claim.

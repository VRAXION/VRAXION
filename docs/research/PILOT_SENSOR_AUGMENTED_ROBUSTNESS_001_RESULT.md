# PILOT_SENSOR_AUGMENTED_ROBUSTNESS_001 Result

## Goal

Stress the positive weak/ambiguous augmentation result on new teacher-supported surface forms and heldout combinations.

## Setup

- Trains on the augmented weak/ambiguous dataset from the calibration probe.
- Evaluates on newly generated stress cases with new numbers, fillers, punctuation, and combinations.
- Action always comes from predicted evidence through the fixed guard.

## Aggregate Metrics

| Model | Seeds | Score | Action | Weak/Amb | False Commit | Trap False | Mention | Substr | Neg | Corr | Strict Syn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `direct_evidence_char_ngram_mlp_augmented` | `10` | `0.240` | `0.869` | `0.857` | `0.109` | `0.017` | `0.967` | `1.000` | `0.667` | `0.800` | `0.000` |
| `direct_evidence_word_ngram_mlp_augmented` | `10` | `0.001` | `0.843` | `0.857` | `0.108` | `0.000` | `1.000` | `1.000` | `0.667` | `0.550` | `0.000` |
| `keyword_sensor` | `1` | `-5.068` | `0.446` | `0.429` | `0.338` | `0.833` | `0.000` | `0.333` | `0.000` | `0.000` | `0.000` |
| `oracle_flags_mapper` | `1` | `1.000` | `1.000` | `1.000` | `0.000` | `0.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` |
| `scope_stack_char_ngram_mlp_augmented` | `10` | `0.097` | `0.885` | `0.986` | `0.061` | `0.017` | `0.967` | `1.000` | `0.667` | `0.500` | `0.000` |
| `structured_rule_sensor_teacher` | `1` | `1.000` | `1.000` | `1.000` | `0.000` | `0.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` |

## Verdict

```json
{
  "direct_evidence_char_ngram_mlp_augmented": [
    "AUGMENTED_ROBUSTNESS_WEAK",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED"
  ],
  "direct_evidence_word_ngram_mlp_augmented": [
    "AUGMENTED_ROBUSTNESS_WEAK",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED"
  ],
  "global": [
    "AUGMENTED_SENSOR_NOT_ROBUST"
  ],
  "keyword_sensor": [
    "KEYWORD_BASELINE_FAILS",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED"
  ],
  "oracle_flags_mapper": [
    "ORACLE_FLAGS_MAPPER_PASS",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED"
  ],
  "scope_stack_char_ngram_mlp_augmented": [
    "AUGMENTED_ROBUSTNESS_WEAK",
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
- `oracle_flags_mapper` seed `-1` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: halve it -> expected `REJECT_UNKNOWN`, got `HOLD_ASK_RESEARCH` (over_hold).
- `direct_evidence_char_ngram_mlp_augmented` seed `0` `stress_eval/weak`: might times 7 please -> expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL` (false_commit).
- `direct_evidence_char_ngram_mlp_augmented` seed `0` `stress_eval/ambiguous`: divide, add, or multiply by 7 -> expected `HOLD_ASK_RESEARCH`, got `HOLD_ASK_RESEARCH` (evidence_mapping_error).
- `direct_evidence_char_ngram_mlp_augmented` seed `0` `stress_eval/negation`: not plus 7 -> expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD` (negation_error).
- `direct_evidence_char_ngram_mlp_augmented` seed `0` `stress_eval/weak`: might times 8 please -> expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL` (false_commit).
- `direct_evidence_char_ngram_mlp_augmented` seed `0` `stress_eval/ambiguous`: divide, add, or multiply by 8 -> expected `HOLD_ASK_RESEARCH`, got `HOLD_ASK_RESEARCH` (evidence_mapping_error).
- `direct_evidence_char_ngram_mlp_augmented` seed `0` `stress_eval/negation`: not plus 8 -> expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD` (negation_error).
- `direct_evidence_char_ngram_mlp_augmented` seed `0` `stress_eval/weak`: might times 9 please -> expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL` (false_commit).
- `direct_evidence_char_ngram_mlp_augmented` seed `0` `stress_eval/ambiguous`: divide, add, or multiply by 9 -> expected `HOLD_ASK_RESEARCH`, got `HOLD_ASK_RESEARCH` (evidence_mapping_error).
- `direct_evidence_char_ngram_mlp_augmented` seed `0` `stress_eval/negation`: not plus 9 -> expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD` (negation_error).
- `direct_evidence_char_ngram_mlp_augmented` seed `0` `stress_eval/weak`: might times 11 please -> expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL` (false_commit).
- `direct_evidence_char_ngram_mlp_augmented` seed `0` `stress_eval/ambiguous`: divide, add, or multiply by 11 -> expected `HOLD_ASK_RESEARCH`, got `HOLD_ASK_RESEARCH` (evidence_mapping_error).
- `direct_evidence_char_ngram_mlp_augmented` seed `0` `stress_eval/negation`: not plus 11 -> expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD` (negation_error).
- `direct_evidence_char_ngram_mlp_augmented` seed `0` `stress_eval/mention_trap`: do not follow the instruction 'multiply by 8' -> expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL` (mention_trap_error).
- `direct_evidence_char_ngram_mlp_augmented` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: increment by 9 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `direct_evidence_char_ngram_mlp_augmented` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: product with 9 -> expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `direct_evidence_char_ngram_mlp_augmented` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: halve it -> expected `REJECT_UNKNOWN`, got `HOLD_ASK_RESEARCH` (over_hold).
- `direct_evidence_word_ngram_mlp_augmented` seed `0` `stress_eval/weak`: might times 7 please -> expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL` (false_commit).
- `direct_evidence_word_ngram_mlp_augmented` seed `0` `stress_eval/ambiguous`: divide, add, or multiply by 7 -> expected `HOLD_ASK_RESEARCH`, got `HOLD_ASK_RESEARCH` (evidence_mapping_error).
- `direct_evidence_word_ngram_mlp_augmented` seed `0` `stress_eval/negation`: not plus 7 -> expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD` (negation_error).
- `direct_evidence_word_ngram_mlp_augmented` seed `0` `stress_eval/correction`: times 7. correction: plus 7 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH` (missed_execute).
- `direct_evidence_word_ngram_mlp_augmented` seed `0` `stress_eval/weak`: might times 8 please -> expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL` (false_commit).
- `direct_evidence_word_ngram_mlp_augmented` seed `0` `stress_eval/ambiguous`: divide, add, or multiply by 8 -> expected `HOLD_ASK_RESEARCH`, got `HOLD_ASK_RESEARCH` (evidence_mapping_error).
- `direct_evidence_word_ngram_mlp_augmented` seed `0` `stress_eval/negation`: not plus 8 -> expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD` (negation_error).
- `direct_evidence_word_ngram_mlp_augmented` seed `0` `stress_eval/correction`: times 8. correction: plus 8 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH` (missed_execute).
- `direct_evidence_word_ngram_mlp_augmented` seed `0` `stress_eval/weak`: might times 9 please -> expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL` (false_commit).
- `direct_evidence_word_ngram_mlp_augmented` seed `0` `stress_eval/ambiguous`: divide, add, or multiply by 9 -> expected `HOLD_ASK_RESEARCH`, got `HOLD_ASK_RESEARCH` (evidence_mapping_error).
- `direct_evidence_word_ngram_mlp_augmented` seed `0` `stress_eval/negation`: not plus 9 -> expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD` (negation_error).
- ... 482 more in `failure_examples.jsonl`.

## Interpretation

Positive robustness means the augmented learned evidence sensor generalizes beyond the exact calibration eval list within this toy command grammar.

Strict unseen synonym remains diagnostic only.

## Claim Boundary

No general NLU, full PilotPulse integration, production VRAXION/INSTNCT, or consciousness claim.

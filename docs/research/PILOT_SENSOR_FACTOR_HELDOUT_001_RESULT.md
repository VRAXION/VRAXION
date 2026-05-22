# PILOT_SENSOR_FACTOR_HELDOUT_001 Result

## Goal

Test whether learned sensors compose seen weak, scope, correction, and operation factors into heldout combinations.

## Setup

- Training includes every relevant atom, but with selected factor combinations held out.
- Evaluation uses heldout combinations such as `might times`, `not plus`, `times. correction: plus`, and quoted multiply instructions.
- Action always comes from predicted evidence through the fixed guard.

## Aggregate Metrics

| Model | Seeds | Score | Action | Weak/Amb | False Commit | Trap False | Mention | Neg | Corr | Strict Syn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `direct_evidence_char_ngram_mlp_factor` | `10` | `-1.102` | `0.651` | `0.475` | `0.264` | `0.000` | `1.000` | `0.500` | `0.500` | `0.000` |
| `direct_evidence_word_ngram_mlp_factor` | `10` | `-1.556` | `0.574` | `0.250` | `0.340` | `0.000` | `1.000` | `0.500` | `0.500` | `0.000` |
| `keyword_sensor` | `1` | `-5.088` | `0.362` | `0.250` | `0.383` | `0.667` | `0.000` | `0.000` | `0.000` | `0.000` |
| `oracle_flags_mapper` | `1` | `1.000` | `1.000` | `1.000` | `0.000` | `0.000` | `1.000` | `1.000` | `1.000` | `0.000` |
| `scope_stack_char_ngram_mlp_factor` | `10` | `-0.546` | `0.745` | `0.750` | `0.170` | `0.000` | `1.000` | `0.500` | `0.500` | `0.000` |
| `structured_rule_sensor_teacher` | `1` | `1.000` | `1.000` | `1.000` | `0.000` | `0.000` | `1.000` | `1.000` | `1.000` | `0.000` |

## Verdict

```json
{
  "direct_evidence_char_ngram_mlp_factor": [
    "FACTOR_HELDOUT_WEAK",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED"
  ],
  "direct_evidence_word_ngram_mlp_factor": [
    "FACTOR_HELDOUT_WEAK",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED"
  ],
  "global": [
    "FACTOR_HELDOUT_REMAINS_BOTTLENECK"
  ],
  "keyword_sensor": [
    "KEYWORD_BASELINE_FAILS",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED"
  ],
  "oracle_flags_mapper": [
    "ORACLE_FLAGS_MAPPER_PASS",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED"
  ],
  "scope_stack_char_ngram_mlp_factor": [
    "FACTOR_HELDOUT_WEAK",
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
- `direct_evidence_char_ngram_mlp_factor` seed `0` `heldout_factor/weak`: might times 7 please -> expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL` (false_commit).
- `direct_evidence_char_ngram_mlp_factor` seed `0` `heldout_factor/weak`: probably plus 7 now -> expected `HOLD_ASK_RESEARCH`, got `HOLD_ASK_RESEARCH` (evidence_mapping_error).
- `direct_evidence_char_ngram_mlp_factor` seed `0` `heldout_factor/weak`: unsure multiply by 7 -> expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL` (false_commit).
- `direct_evidence_char_ngram_mlp_factor` seed `0` `heldout_factor/ambiguous`: divide, add, or multiply by 7 -> expected `HOLD_ASK_RESEARCH`, got `HOLD_ASK_RESEARCH` (evidence_mapping_error).
- `direct_evidence_char_ngram_mlp_factor` seed `0` `heldout_factor/negation`: not plus 7 -> expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD` (negation_error).
- `direct_evidence_char_ngram_mlp_factor` seed `0` `heldout_factor/correction`: times 7. correction: plus 7 -> expected `EXEC_ADD`, got `EXEC_MUL` (correction_error).
- `direct_evidence_char_ngram_mlp_factor` seed `0` `heldout_factor/weak`: might times 8 please -> expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL` (false_commit).
- `direct_evidence_char_ngram_mlp_factor` seed `0` `heldout_factor/weak`: probably plus 8 now -> expected `HOLD_ASK_RESEARCH`, got `HOLD_ASK_RESEARCH` (evidence_mapping_error).
- `direct_evidence_char_ngram_mlp_factor` seed `0` `heldout_factor/weak`: unsure multiply by 8 -> expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL` (false_commit).
- `direct_evidence_char_ngram_mlp_factor` seed `0` `heldout_factor/ambiguous`: divide, add, or multiply by 8 -> expected `HOLD_ASK_RESEARCH`, got `HOLD_ASK_RESEARCH` (evidence_mapping_error).
- `direct_evidence_char_ngram_mlp_factor` seed `0` `heldout_factor/negation`: not plus 8 -> expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD` (negation_error).
- `direct_evidence_char_ngram_mlp_factor` seed `0` `heldout_factor/correction`: times 8. correction: plus 8 -> expected `EXEC_ADD`, got `EXEC_MUL` (correction_error).
- `direct_evidence_char_ngram_mlp_factor` seed `0` `heldout_factor/weak`: might times 9 please -> expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL` (false_commit).
- `direct_evidence_char_ngram_mlp_factor` seed `0` `heldout_factor/weak`: probably plus 9 now -> expected `HOLD_ASK_RESEARCH`, got `HOLD_ASK_RESEARCH` (evidence_mapping_error).
- `direct_evidence_char_ngram_mlp_factor` seed `0` `heldout_factor/weak`: unsure multiply by 9 -> expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL` (false_commit).
- `direct_evidence_char_ngram_mlp_factor` seed `0` `heldout_factor/ambiguous`: divide, add, or multiply by 9 -> expected `HOLD_ASK_RESEARCH`, got `HOLD_ASK_RESEARCH` (evidence_mapping_error).
- `direct_evidence_char_ngram_mlp_factor` seed `0` `heldout_factor/negation`: not plus 9 -> expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD` (negation_error).
- `direct_evidence_char_ngram_mlp_factor` seed `0` `heldout_factor/correction`: times 9. correction: plus 9 -> expected `EXEC_ADD`, got `EXEC_MUL` (correction_error).
- `direct_evidence_char_ngram_mlp_factor` seed `0` `heldout_factor/weak`: might times 11 please -> expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL` (false_commit).
- `direct_evidence_char_ngram_mlp_factor` seed `0` `heldout_factor/weak`: probably plus 11 now -> expected `HOLD_ASK_RESEARCH`, got `HOLD_ASK_RESEARCH` (evidence_mapping_error).
- `direct_evidence_char_ngram_mlp_factor` seed `0` `heldout_factor/weak`: unsure multiply by 11 -> expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL` (false_commit).
- `direct_evidence_char_ngram_mlp_factor` seed `0` `heldout_factor/ambiguous`: divide, add, or multiply by 11 -> expected `HOLD_ASK_RESEARCH`, got `HOLD_ASK_RESEARCH` (evidence_mapping_error).
- `direct_evidence_char_ngram_mlp_factor` seed `0` `heldout_factor/negation`: not plus 11 -> expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD` (negation_error).
- `direct_evidence_char_ngram_mlp_factor` seed `0` `heldout_factor/correction`: times 11. correction: plus 11 -> expected `EXEC_ADD`, got `EXEC_MUL` (correction_error).
- `direct_evidence_char_ngram_mlp_factor` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: increment by 9 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `direct_evidence_char_ngram_mlp_factor` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: product with 9 -> expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `direct_evidence_word_ngram_mlp_factor` seed `0` `heldout_factor/weak`: might times 7 please -> expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL` (false_commit).
- `direct_evidence_word_ngram_mlp_factor` seed `0` `heldout_factor/weak`: probably plus 7 now -> expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD` (false_commit).
- ... 712 more in `failure_examples.jsonl`.

## Interpretation

A positive result means systematic template coverage can generalize across some heldout factor combinations within this toy grammar.
A negative result means the previous systematic pass was closer to template coverage than compositional scope handling.

Strict unseen synonym remains diagnostic only.

## Claim Boundary

No general NLU, full PilotPulse integration, production VRAXION/INSTNCT, or consciousness claim.

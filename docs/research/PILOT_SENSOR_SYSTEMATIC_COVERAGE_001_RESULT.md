# PILOT_SENSOR_SYSTEMATIC_COVERAGE_001 Result

## Goal

Test whether the augmented robustness failure is mainly systematic template coverage or a harder parser/encoder limit.

## Setup

- Trains learned sensors on the augmented calibration data plus generated weak, ambiguity, negation, correction, mention, and multi-step variants.
- Evaluates on the robustness stress suite with heldout numbers and surface forms.
- Action always comes from predicted evidence through the fixed guard.

## Aggregate Metrics

| Model | Seeds | Score | Action | Weak/Amb | False Commit | Trap False | Mention | Substr | Neg | Corr | Strict Syn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `direct_evidence_char_ngram_mlp_systematic` | `10` | `1.000` | `1.000` | `1.000` | `0.000` | `0.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` |
| `direct_evidence_word_ngram_mlp_systematic` | `10` | `1.000` | `1.000` | `1.000` | `0.000` | `0.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` |
| `keyword_sensor` | `1` | `-5.068` | `0.446` | `0.429` | `0.338` | `0.833` | `0.000` | `0.333` | `0.000` | `0.000` | `0.000` |
| `oracle_flags_mapper` | `1` | `1.000` | `1.000` | `1.000` | `0.000` | `0.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` |
| `scope_stack_char_ngram_mlp_systematic` | `10` | `1.000` | `1.000` | `1.000` | `0.000` | `0.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` |
| `structured_rule_sensor_teacher` | `1` | `1.000` | `1.000` | `1.000` | `0.000` | `0.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` |

## Verdict

```json
{
  "direct_evidence_char_ngram_mlp_systematic": [
    "SYSTEMATIC_COVERAGE_POSITIVE",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED"
  ],
  "direct_evidence_word_ngram_mlp_systematic": [
    "SYSTEMATIC_COVERAGE_POSITIVE",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED"
  ],
  "global": [
    "TEMPLATE_COVERAGE_SOLVES_STRESS",
    "direct_evidence_char_ngram_mlp_systematic",
    "direct_evidence_word_ngram_mlp_systematic",
    "scope_stack_char_ngram_mlp_systematic"
  ],
  "keyword_sensor": [
    "KEYWORD_BASELINE_FAILS",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED"
  ],
  "oracle_flags_mapper": [
    "ORACLE_FLAGS_MAPPER_PASS",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED"
  ],
  "scope_stack_char_ngram_mlp_systematic": [
    "SYSTEMATIC_COVERAGE_POSITIVE",
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
- `direct_evidence_char_ngram_mlp_systematic` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: increment by 9 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `direct_evidence_char_ngram_mlp_systematic` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: product with 9 -> expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `direct_evidence_char_ngram_mlp_systematic` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: halve it -> expected `REJECT_UNKNOWN`, got `HOLD_ASK_RESEARCH` (over_hold).
- `direct_evidence_word_ngram_mlp_systematic` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: increment by 9 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `direct_evidence_word_ngram_mlp_systematic` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: product with 9 -> expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `direct_evidence_word_ngram_mlp_systematic` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: halve it -> expected `REJECT_UNKNOWN`, got `HOLD_ASK_RESEARCH` (over_hold).
- `scope_stack_char_ngram_mlp_systematic` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: increment by 9 -> expected `EXEC_ADD`, got `REJECT_UNKNOWN` (synonym_gap).
- `scope_stack_char_ngram_mlp_systematic` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: product with 9 -> expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `scope_stack_char_ngram_mlp_systematic` seed `0` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: halve it -> expected `REJECT_UNKNOWN`, got `HOLD_ASK_RESEARCH` (over_hold).
- `direct_evidence_char_ngram_mlp_systematic` seed `1` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: increment by 9 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `direct_evidence_char_ngram_mlp_systematic` seed `1` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: product with 9 -> expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `direct_evidence_char_ngram_mlp_systematic` seed `1` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: halve it -> expected `REJECT_UNKNOWN`, got `HOLD_ASK_RESEARCH` (over_hold).
- `direct_evidence_word_ngram_mlp_systematic` seed `1` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: increment by 9 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `direct_evidence_word_ngram_mlp_systematic` seed `1` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: product with 9 -> expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `direct_evidence_word_ngram_mlp_systematic` seed `1` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: halve it -> expected `REJECT_UNKNOWN`, got `HOLD_ASK_RESEARCH` (over_hold).
- `scope_stack_char_ngram_mlp_systematic` seed `1` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: increment by 9 -> expected `EXEC_ADD`, got `REJECT_UNKNOWN` (synonym_gap).
- `scope_stack_char_ngram_mlp_systematic` seed `1` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: product with 9 -> expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `scope_stack_char_ngram_mlp_systematic` seed `1` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: halve it -> expected `REJECT_UNKNOWN`, got `HOLD_ASK_RESEARCH` (over_hold).
- `direct_evidence_char_ngram_mlp_systematic` seed `2` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: increment by 9 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `direct_evidence_char_ngram_mlp_systematic` seed `2` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: product with 9 -> expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `direct_evidence_char_ngram_mlp_systematic` seed `2` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: halve it -> expected `REJECT_UNKNOWN`, got `HOLD_ASK_RESEARCH` (over_hold).
- `direct_evidence_word_ngram_mlp_systematic` seed `2` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: increment by 9 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `direct_evidence_word_ngram_mlp_systematic` seed `2` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: product with 9 -> expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `direct_evidence_word_ngram_mlp_systematic` seed `2` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: halve it -> expected `REJECT_UNKNOWN`, got `HOLD_ASK_RESEARCH` (over_hold).
- `scope_stack_char_ngram_mlp_systematic` seed `2` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: increment by 9 -> expected `EXEC_ADD`, got `REJECT_UNKNOWN` (synonym_gap).
- `scope_stack_char_ngram_mlp_systematic` seed `2` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: product with 9 -> expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `scope_stack_char_ngram_mlp_systematic` seed `2` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: halve it -> expected `REJECT_UNKNOWN`, got `HOLD_ASK_RESEARCH` (over_hold).
- ... 63 more in `failure_examples.jsonl`.

## Interpretation

If systematic coverage passes, the previous failure was primarily data/template coverage within this toy command grammar.
If it does not pass while the teacher/oracle do, the next blocker is a more explicit scope/event parser or stronger pretrained text encoder.

Strict unseen synonym remains diagnostic only.

## Claim Boundary

No general NLU, full PilotPulse integration, production VRAXION/INSTNCT, or consciousness claim.

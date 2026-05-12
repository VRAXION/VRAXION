# PILOT_SENSOR_DISTILL_001 Result

## Goal

Distill the hand-auditable `structured_rule_sensor` into learned text-to-evidence sensors while preserving the evidence bottleneck.

This is not a natural-language-understanding claim. The downstream action is always produced by the fixed guard from predicted evidence.

## Setup

- Teacher: `structured_rule_sensor` from `PILOT_SENSOR_001`.
- Baselines: `keyword_sensor`, `structured_rule_sensor_teacher`.
- Students: `word_ngram_linear_student`, `char_ngram_linear_student`.
- Targets: per-channel evidence bands `NONE`, `WEAK`, `STRONG`; no direct action head is used for verdict.
- Guards: `evidence_strength_margin_guard`, `topK2_guard`; thresholds are unchanged from `PILOT_TOPK_GUARD_001`.

## Context

This probe uses an auditable evidence bottleneck: text is translated into intermediate evidence concepts, then a deterministic guard produces the action.

Related reference points: concept bottleneck models, faithful translation-to-solver reasoning, and classification with reject/abstain options.

- Concept Bottleneck Models: https://www.microsoft.com/en-us/research/publication/concept-bottleneck-models/
- Faithful Chain-of-Thought Reasoning: https://huggingface.co/papers/2301.13379
- Classification with reject option: https://www.sciencedirect.com/science/article/pii/S2666827025000477

## Aggregate Metrics

| Model+Guard | Train | Main Action | Evidence Band | Disagree | Surface | Scope | Negation | Correction | False Commit | Trap False Commit | Strict Synonym |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `char_ngram_linear_student+evidence_strength_margin_guard` | `1.000` | `0.864` | `0.864` | `0.136` | `1.000` | `0.800` | `1.000` | `0.500` | `0.091` | `0.200` | `0.000` |
| `char_ngram_linear_student+topK2_guard` | `1.000` | `0.864` | `0.864` | `0.136` | `1.000` | `0.800` | `1.000` | `0.500` | `0.091` | `0.200` | `0.000` |
| `keyword_sensor+evidence_strength_margin_guard` | `0.467` | `0.500` | `0.409` | `0.500` | `1.000` | `0.200` | `0.333` | `0.250` | `0.273` | `0.800` | `0.000` |
| `keyword_sensor+topK2_guard` | `0.467` | `0.500` | `0.409` | `0.500` | `1.000` | `0.200` | `0.333` | `0.250` | `0.273` | `0.800` | `0.000` |
| `structured_rule_sensor_teacher+evidence_strength_margin_guard` | `1.000` | `1.000` | `1.000` | `0.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` | `0.000` | `0.000` |
| `structured_rule_sensor_teacher+topK2_guard` | `1.000` | `1.000` | `1.000` | `0.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` | `0.000` | `0.000` |
| `word_ngram_linear_student+evidence_strength_margin_guard` | `1.000` | `0.909` | `0.909` | `0.091` | `1.000` | `0.800` | `1.000` | `0.750` | `0.045` | `0.200` | `0.000` |
| `word_ngram_linear_student+topK2_guard` | `1.000` | `0.909` | `0.909` | `0.091` | `1.000` | `0.800` | `1.000` | `0.750` | `0.045` | `0.200` | `0.000` |

## Phenomenon-Tagged Metrics

| Model+Guard | Phenomenon | Action | False Commit | Evidence Band | Disagree |
|---|---|---:|---:|---:|---:|
| `char_ngram_linear_student+evidence_strength_margin_guard` | `correction` | `0.667` | `0.000` | `0.667` | `0.333` |
| `char_ngram_linear_student+evidence_strength_margin_guard` | `mention_trap` | `0.667` | `0.333` | `0.667` | `0.333` |
| `char_ngram_linear_student+evidence_strength_margin_guard` | `multi_step_unsupported` | `0.000` | `1.000` | `0.000` | `1.000` |
| `char_ngram_linear_student+evidence_strength_margin_guard` | `negation` | `1.000` | `0.000` | `1.000` | `0.000` |
| `char_ngram_linear_student+evidence_strength_margin_guard` | `strict_unseen_synonym` | `0.000` | `0.200` | `0.800` | `0.200` |
| `char_ngram_linear_student+evidence_strength_margin_guard` | `substring_trap` | `1.000` | `0.000` | `1.000` | `0.000` |
| `char_ngram_linear_student+evidence_strength_margin_guard` | `surface_variation` | `1.000` | `0.000` | `1.000` | `0.000` |
| `char_ngram_linear_student+topK2_guard` | `correction` | `0.667` | `0.000` | `0.667` | `0.333` |
| `char_ngram_linear_student+topK2_guard` | `mention_trap` | `0.667` | `0.333` | `0.667` | `0.333` |
| `char_ngram_linear_student+topK2_guard` | `multi_step_unsupported` | `0.000` | `1.000` | `0.000` | `1.000` |
| `char_ngram_linear_student+topK2_guard` | `negation` | `1.000` | `0.000` | `1.000` | `0.000` |
| `char_ngram_linear_student+topK2_guard` | `strict_unseen_synonym` | `0.000` | `0.200` | `0.800` | `0.200` |
| `char_ngram_linear_student+topK2_guard` | `substring_trap` | `1.000` | `0.000` | `1.000` | `0.000` |
| `char_ngram_linear_student+topK2_guard` | `surface_variation` | `1.000` | `0.000` | `1.000` | `0.000` |
| `keyword_sensor+evidence_strength_margin_guard` | `correction` | `0.000` | `0.000` | `0.000` | `1.000` |
| `keyword_sensor+evidence_strength_margin_guard` | `mention_trap` | `0.000` | `1.000` | `0.000` | `1.000` |
| `keyword_sensor+evidence_strength_margin_guard` | `multi_step_unsupported` | `1.000` | `0.000` | `1.000` | `0.000` |
| `keyword_sensor+evidence_strength_margin_guard` | `negation` | `0.333` | `0.333` | `0.000` | `0.667` |
| `keyword_sensor+evidence_strength_margin_guard` | `strict_unseen_synonym` | `0.000` | `0.000` | `1.000` | `0.000` |
| `keyword_sensor+evidence_strength_margin_guard` | `substring_trap` | `0.500` | `0.500` | `0.500` | `0.500` |
| `keyword_sensor+evidence_strength_margin_guard` | `surface_variation` | `1.000` | `0.000` | `1.000` | `0.000` |
| `keyword_sensor+topK2_guard` | `correction` | `0.000` | `0.000` | `0.000` | `1.000` |
| `keyword_sensor+topK2_guard` | `mention_trap` | `0.000` | `1.000` | `0.000` | `1.000` |
| `keyword_sensor+topK2_guard` | `multi_step_unsupported` | `1.000` | `0.000` | `1.000` | `0.000` |
| `keyword_sensor+topK2_guard` | `negation` | `0.333` | `0.333` | `0.000` | `0.667` |
| `keyword_sensor+topK2_guard` | `strict_unseen_synonym` | `0.000` | `0.000` | `1.000` | `0.000` |
| `keyword_sensor+topK2_guard` | `substring_trap` | `0.500` | `0.500` | `0.500` | `0.500` |
| `keyword_sensor+topK2_guard` | `surface_variation` | `1.000` | `0.000` | `1.000` | `0.000` |
| `structured_rule_sensor_teacher+evidence_strength_margin_guard` | `correction` | `1.000` | `0.000` | `1.000` | `0.000` |
| `structured_rule_sensor_teacher+evidence_strength_margin_guard` | `mention_trap` | `1.000` | `0.000` | `1.000` | `0.000` |
| `structured_rule_sensor_teacher+evidence_strength_margin_guard` | `multi_step_unsupported` | `1.000` | `0.000` | `1.000` | `0.000` |
| `structured_rule_sensor_teacher+evidence_strength_margin_guard` | `negation` | `1.000` | `0.000` | `1.000` | `0.000` |
| `structured_rule_sensor_teacher+evidence_strength_margin_guard` | `strict_unseen_synonym` | `0.000` | `0.000` | `1.000` | `0.000` |
| `structured_rule_sensor_teacher+evidence_strength_margin_guard` | `substring_trap` | `1.000` | `0.000` | `1.000` | `0.000` |
| `structured_rule_sensor_teacher+evidence_strength_margin_guard` | `surface_variation` | `1.000` | `0.000` | `1.000` | `0.000` |
| `structured_rule_sensor_teacher+topK2_guard` | `correction` | `1.000` | `0.000` | `1.000` | `0.000` |
| `structured_rule_sensor_teacher+topK2_guard` | `mention_trap` | `1.000` | `0.000` | `1.000` | `0.000` |
| `structured_rule_sensor_teacher+topK2_guard` | `multi_step_unsupported` | `1.000` | `0.000` | `1.000` | `0.000` |
| `structured_rule_sensor_teacher+topK2_guard` | `negation` | `1.000` | `0.000` | `1.000` | `0.000` |
| `structured_rule_sensor_teacher+topK2_guard` | `strict_unseen_synonym` | `0.000` | `0.000` | `1.000` | `0.000` |
| `structured_rule_sensor_teacher+topK2_guard` | `substring_trap` | `1.000` | `0.000` | `1.000` | `0.000` |
| `structured_rule_sensor_teacher+topK2_guard` | `surface_variation` | `1.000` | `0.000` | `1.000` | `0.000` |
| `word_ngram_linear_student+evidence_strength_margin_guard` | `correction` | `0.667` | `0.000` | `0.667` | `0.333` |
| `word_ngram_linear_student+evidence_strength_margin_guard` | `mention_trap` | `1.000` | `0.000` | `1.000` | `0.000` |
| `word_ngram_linear_student+evidence_strength_margin_guard` | `multi_step_unsupported` | `1.000` | `0.000` | `1.000` | `0.000` |
| `word_ngram_linear_student+evidence_strength_margin_guard` | `negation` | `1.000` | `0.000` | `1.000` | `0.000` |
| `word_ngram_linear_student+evidence_strength_margin_guard` | `strict_unseen_synonym` | `0.000` | `0.200` | `0.600` | `0.400` |
| `word_ngram_linear_student+evidence_strength_margin_guard` | `substring_trap` | `0.500` | `0.500` | `0.500` | `0.500` |
| `word_ngram_linear_student+evidence_strength_margin_guard` | `surface_variation` | `1.000` | `0.000` | `1.000` | `0.000` |
| `word_ngram_linear_student+topK2_guard` | `correction` | `0.667` | `0.000` | `0.667` | `0.333` |
| `word_ngram_linear_student+topK2_guard` | `mention_trap` | `1.000` | `0.000` | `1.000` | `0.000` |
| `word_ngram_linear_student+topK2_guard` | `multi_step_unsupported` | `1.000` | `0.000` | `1.000` | `0.000` |
| `word_ngram_linear_student+topK2_guard` | `negation` | `1.000` | `0.000` | `1.000` | `0.000` |
| `word_ngram_linear_student+topK2_guard` | `strict_unseen_synonym` | `0.000` | `0.200` | `0.600` | `0.400` |
| `word_ngram_linear_student+topK2_guard` | `substring_trap` | `0.500` | `0.500` | `0.500` | `0.500` |
| `word_ngram_linear_student+topK2_guard` | `surface_variation` | `1.000` | `0.000` | `1.000` | `0.000` |

## Verdict

```json
{
  "char_ngram_linear_student+evidence_strength_margin_guard": [
    "DISTILL_SCOPE_WEAK",
    "DISTILL_CORRECTION_WEAK",
    "DISTILL_FALSE_COMMIT_HIGH",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED",
    "DISTILL_RULE_TEACHER_ONLY"
  ],
  "char_ngram_linear_student+topK2_guard": [
    "DISTILL_SCOPE_WEAK",
    "DISTILL_CORRECTION_WEAK",
    "DISTILL_FALSE_COMMIT_HIGH",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED",
    "DISTILL_RULE_TEACHER_ONLY"
  ],
  "keyword_sensor+evidence_strength_margin_guard": [
    "DISTILL_SCOPE_WEAK",
    "DISTILL_FALSE_COMMIT_HIGH",
    "DISTILL_NEGATION_WEAK",
    "DISTILL_CORRECTION_WEAK"
  ],
  "keyword_sensor+topK2_guard": [
    "DISTILL_SCOPE_WEAK",
    "DISTILL_FALSE_COMMIT_HIGH",
    "DISTILL_NEGATION_WEAK",
    "DISTILL_CORRECTION_WEAK"
  ],
  "structured_rule_sensor_teacher+evidence_strength_margin_guard": [
    "TEACHER_REFERENCE_PASS"
  ],
  "structured_rule_sensor_teacher+topK2_guard": [
    "TEACHER_REFERENCE_PASS"
  ],
  "word_ngram_linear_student+evidence_strength_margin_guard": [
    "DISTILL_SCOPE_WEAK",
    "DISTILL_CORRECTION_WEAK",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED",
    "DISTILL_RULE_TEACHER_ONLY"
  ],
  "word_ngram_linear_student+topK2_guard": [
    "DISTILL_SCOPE_WEAK",
    "DISTILL_CORRECTION_WEAK",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED",
    "DISTILL_RULE_TEACHER_ONLY"
  ]
}
```

## Minimal Pairs

| Probe | Model+Guard | Teacher Evidence | Student Evidence | Student Action |
|---|---|---|---|---|
| `add 3` | `structured_rule_sensor_teacher+evidence_strength_margin_guard` | `[0.9, 0.0, 0.0]` | `[0.9, 0.0, 0.0]` | `EXEC_ADD` |
| `add 3` | `keyword_sensor+evidence_strength_margin_guard` | `[0.9, 0.0, 0.0]` | `[0.9, 0.0, 0.0]` | `EXEC_ADD` |
| `do not add 3` | `structured_rule_sensor_teacher+evidence_strength_margin_guard` | `[0.0, 0.0, 0.0]` | `[0.0, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` |
| `do not add 3` | `keyword_sensor+evidence_strength_margin_guard` | `[0.0, 0.0, 0.0]` | `[0.9, 0.0, 0.0]` | `EXEC_ADD` |
| `do not add 3` | `word_ngram_linear_student+evidence_strength_margin_guard` | `[0.0, 0.0, 0.0]` | `[0.0, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` |
| `do not add 3` | `char_ngram_linear_student+evidence_strength_margin_guard` | `[0.0, 0.0, 0.0]` | `[0.0, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` |
| `do not add 3, multiply by 3 instead` | `structured_rule_sensor_teacher+evidence_strength_margin_guard` | `[0.0, 0.9, 0.0]` | `[0.0, 0.9, 0.0]` | `EXEC_MUL` |
| `do not add 3, multiply by 3 instead` | `keyword_sensor+evidence_strength_margin_guard` | `[0.0, 0.9, 0.0]` | `[0.9, 0.9, 0.0]` | `HOLD_ASK_RESEARCH` |
| `do not add 3, multiply by 3 instead` | `word_ngram_linear_student+evidence_strength_margin_guard` | `[0.0, 0.9, 0.0]` | `[0.0, 0.9, 0.0]` | `EXEC_MUL` |
| `do not add 3, multiply by 3 instead` | `char_ngram_linear_student+evidence_strength_margin_guard` | `[0.0, 0.9, 0.0]` | `[0.0, 0.9, 0.0]` | `EXEC_MUL` |
| `the word add appears in the note` | `structured_rule_sensor_teacher+evidence_strength_margin_guard` | `[0.0, 0.0, 0.0]` | `[0.0, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` |
| `the word add appears in the note` | `keyword_sensor+evidence_strength_margin_guard` | `[0.0, 0.0, 0.0]` | `[0.9, 0.0, 0.0]` | `EXEC_ADD` |
| `the word add appears in the note` | `word_ngram_linear_student+evidence_strength_margin_guard` | `[0.0, 0.0, 0.0]` | `[0.0, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` |
| `the word add appears in the note` | `char_ngram_linear_student+evidence_strength_margin_guard` | `[0.0, 0.0, 0.0]` | `[0.0, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` |
| `add 3. wait, actually multiply by 3` | `structured_rule_sensor_teacher+evidence_strength_margin_guard` | `[0.0, 0.9, 0.0]` | `[0.0, 0.9, 0.0]` | `EXEC_MUL` |
| `add 3. wait, actually multiply by 3` | `keyword_sensor+evidence_strength_margin_guard` | `[0.0, 0.9, 0.0]` | `[0.9, 0.9, 0.0]` | `HOLD_ASK_RESEARCH` |
| `add 3. wait, actually multiply by 3` | `word_ngram_linear_student+evidence_strength_margin_guard` | `[0.0, 0.9, 0.0]` | `[0.0, 0.9, 0.0]` | `EXEC_MUL` |
| `add 3. wait, actually multiply by 3` | `char_ngram_linear_student+evidence_strength_margin_guard` | `[0.0, 0.9, 0.0]` | `[0.0, 0.9, 0.0]` | `EXEC_MUL` |

## Failure Examples

- `word_ngram_linear_student+evidence_strength_margin_guard` `heldout_scope/substring_trap`: plus sign appears on the page -> expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD` (mention_trap_error).
- `word_ngram_linear_student+evidence_strength_margin_guard` `heldout_correction/correction`: use add. actually no, use multiply -> expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH` (missed_execute).
- `word_ngram_linear_student+evidence_strength_margin_guard` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: increment by 3 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `word_ngram_linear_student+evidence_strength_margin_guard` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: raise the value by 3 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `word_ngram_linear_student+evidence_strength_margin_guard` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: product with 3 -> expected `EXEC_MUL`, got `EXEC_ADD` (evidence_band_error).
- `word_ngram_linear_student+evidence_strength_margin_guard` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: halve it -> expected `REJECT_UNKNOWN`, got `EXEC_ADD` (false_commit).
- `word_ngram_linear_student+evidence_strength_margin_guard` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: exponentiate by 3 -> expected `REJECT_UNKNOWN`, got `HOLD_ASK_RESEARCH` (unknown_missed).
- `word_ngram_linear_student+topK2_guard` `heldout_scope/substring_trap`: plus sign appears on the page -> expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD` (mention_trap_error).
- `word_ngram_linear_student+topK2_guard` `heldout_correction/correction`: use add. actually no, use multiply -> expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH` (missed_execute).
- `word_ngram_linear_student+topK2_guard` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: increment by 3 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `word_ngram_linear_student+topK2_guard` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: raise the value by 3 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `word_ngram_linear_student+topK2_guard` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: product with 3 -> expected `EXEC_MUL`, got `EXEC_ADD` (evidence_band_error).
- `word_ngram_linear_student+topK2_guard` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: halve it -> expected `REJECT_UNKNOWN`, got `EXEC_ADD` (false_commit).
- `word_ngram_linear_student+topK2_guard` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: exponentiate by 3 -> expected `REJECT_UNKNOWN`, got `HOLD_ASK_RESEARCH` (unknown_missed).
- `char_ngram_linear_student+evidence_strength_margin_guard` `heldout_scope/mention_trap`: do not follow the instruction 'add 3' -> expected `HOLD_ASK_RESEARCH|REFRAME`, got `EXEC_ADD` (mention_trap_error).
- `char_ngram_linear_student+evidence_strength_margin_guard` `heldout_correction/multi_step_unsupported`: first add 3, then multiply by 3 -> expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL` (false_commit).
- `char_ngram_linear_student+evidence_strength_margin_guard` `heldout_correction/correction`: use add. actually no, use multiply -> expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH` (missed_execute).
- `char_ngram_linear_student+evidence_strength_margin_guard` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: increment by 3 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `char_ngram_linear_student+evidence_strength_margin_guard` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: raise the value by 3 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `char_ngram_linear_student+evidence_strength_margin_guard` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: product with 3 -> expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `char_ngram_linear_student+evidence_strength_margin_guard` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: halve it -> expected `REJECT_UNKNOWN`, got `EXEC_ADD` (false_commit).
- `char_ngram_linear_student+evidence_strength_margin_guard` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: exponentiate by 3 -> expected `REJECT_UNKNOWN`, got `HOLD_ASK_RESEARCH` (unknown_missed).
- `char_ngram_linear_student+topK2_guard` `heldout_scope/mention_trap`: do not follow the instruction 'add 3' -> expected `HOLD_ASK_RESEARCH|REFRAME`, got `EXEC_ADD` (mention_trap_error).
- `char_ngram_linear_student+topK2_guard` `heldout_correction/multi_step_unsupported`: first add 3, then multiply by 3 -> expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL` (false_commit).
- `char_ngram_linear_student+topK2_guard` `heldout_correction/correction`: use add. actually no, use multiply -> expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH` (missed_execute).
- `char_ngram_linear_student+topK2_guard` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: increment by 3 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `char_ngram_linear_student+topK2_guard` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: raise the value by 3 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `char_ngram_linear_student+topK2_guard` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: product with 3 -> expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH` (synonym_gap).
- `char_ngram_linear_student+topK2_guard` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: halve it -> expected `REJECT_UNKNOWN`, got `EXEC_ADD` (false_commit).
- `char_ngram_linear_student+topK2_guard` `strict_unseen_synonym_diagnostic/strict_unseen_synonym`: exponentiate by 3 -> expected `REJECT_UNKNOWN`, got `HOLD_ASK_RESEARCH` (unknown_missed).

## Interpretation

A positive learned-student result means the rule teacher's scope-aware evidence state is learnable on this controlled toy command language.

If the teacher passes but learned students fail, the bottleneck remains learned scope-aware text-to-evidence extraction.

The strict unseen synonym split is diagnostic only because this setup has no pretrained semantic knowledge.

## Claim Boundary

No general NLU, full PilotPulse learning, full VRAXION/INSTNCT proof, production architecture, or consciousness claim.

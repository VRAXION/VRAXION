# PILOT_SENSOR_001 Result

## Goal

Test controlled and adversarial raw command text -> evidence vector -> fixed guard -> pilot action.

This is a hand-auditable sensor baseline, not a learning or natural-language-understanding claim.

## Setup

Evidence vector: `[ADD, MUL, UNKNOWN]`. Guard thresholds are reused unchanged from `PILOT_TOPK_GUARD_001`.

## Sensors And Guards

- `keyword_sensor`: direct keyword count baseline.
- `structured_rule_sensor`: rule sensor with weak markers, unknown overrides, negation scope, correction handling, and quote/mention traps.
- guards: `evidence_strength_margin_guard`, `topK2_guard`.

## Aggregate Metrics

| Sensor+Guard | Controlled | Adversarial | Evidence Band | False Commit | Trap False Commit | Negation | Correction | Unknown | Weak | Ambiguous |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `keyword_sensor+evidence_strength_margin_guard` | `0.778` | `0.333` | `0.564` | `0.308` | `0.714` | `0.286` | `0.000` | `0.833` | `0.000` | `1.000` |
| `keyword_sensor+topK2_guard` | `0.778` | `0.333` | `0.564` | `0.308` | `0.714` | `0.286` | `0.000` | `0.833` | `0.000` | `1.000` |
| `structured_rule_sensor+evidence_strength_margin_guard` | `1.000` | `1.000` | `1.000` | `0.000` | `0.000` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` |
| `structured_rule_sensor+topK2_guard` | `1.000` | `1.000` | `1.000` | `0.000` | `0.000` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` |

## Verdict

```json
{
  "keyword_sensor+evidence_strength_margin_guard": [
    "SENSOR_SCOPE_WEAK",
    "SENSOR_KEYWORD_OVERFIRES",
    "SENSOR_UNKNOWN_WEAK",
    "SENSOR_FALSE_COMMIT_HIGH"
  ],
  "keyword_sensor+topK2_guard": [
    "SENSOR_SCOPE_WEAK",
    "SENSOR_KEYWORD_OVERFIRES",
    "SENSOR_UNKNOWN_WEAK",
    "SENSOR_FALSE_COMMIT_HIGH"
  ],
  "structured_rule_sensor+evidence_strength_margin_guard": [
    "SENSOR_POSITIVE"
  ],
  "structured_rule_sensor+topK2_guard": [
    "SENSOR_POSITIVE"
  ]
}
```

## Per-Case Metrics

| Sensor | Guard | Split | Case | Expected | Evidence | Action | Correct | Band OK |
|---|---|---|---|---|---|---|---:|---:|
| `keyword_sensor` | `evidence_strength_margin_guard` | `controlled_eval` | `known_add_1` | `EXEC_ADD` | `[0.9, 0.0, 0.0]` | `EXEC_ADD` | `1` | `1` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `controlled_eval` | `known_add_2` | `EXEC_ADD` | `[0.9, 0.0, 0.0]` | `EXEC_ADD` | `1` | `1` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `controlled_eval` | `known_add_3` | `EXEC_ADD` | `[0.9, 0.0, 0.0]` | `EXEC_ADD` | `1` | `1` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `controlled_eval` | `known_mul_1` | `EXEC_MUL` | `[0.0, 0.9, 0.0]` | `EXEC_MUL` | `1` | `1` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `controlled_eval` | `known_mul_2` | `EXEC_MUL` | `[0.0, 0.9, 0.0]` | `EXEC_MUL` | `1` | `1` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `controlled_eval` | `known_mul_3` | `EXEC_MUL` | `[0.0, 0.9, 0.0]` | `EXEC_MUL` | `1` | `1` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `controlled_eval` | `unknown_div_1` | `REJECT_UNKNOWN` | `[0.0, 0.0, 0.9]` | `REJECT_UNKNOWN` | `1` | `1` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `controlled_eval` | `unknown_div_2` | `REJECT_UNKNOWN` | `[0.0, 0.0, 0.9]` | `REJECT_UNKNOWN` | `1` | `1` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `controlled_eval` | `unknown_div_3` | `REJECT_UNKNOWN` | `[0.0, 0.0, 0.9]` | `REJECT_UNKNOWN` | `1` | `1` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `controlled_eval` | `weak_add_1` | `HOLD_ASK_RESEARCH` | `[0.9, 0.0, 0.0]` | `EXEC_ADD` | `0` | `0` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `controlled_eval` | `weak_mul_1` | `HOLD_ASK_RESEARCH` | `[0.0, 0.9, 0.0]` | `EXEC_MUL` | `0` | `0` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `controlled_eval` | `ambiguous_1` | `HOLD_ASK_RESEARCH` | `[0.9, 0.9, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `controlled_eval` | `no_evidence_1` | `HOLD_ASK_RESEARCH` | `[0.0, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `controlled_eval` | `conflict_1` | `HOLD_ASK_RESEARCH` | `[0.9, 0.9, 0.9]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `controlled_eval` | `negated_add_1` | `HOLD_ASK_RESEARCH|REFRAME` | `[0.9, 0.0, 0.0]` | `EXEC_ADD` | `0` | `0` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `controlled_eval` | `negated_add_then_mul_1` | `EXEC_MUL` | `[0.9, 0.9, 0.0]` | `HOLD_ASK_RESEARCH` | `0` | `0` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `controlled_eval` | `delayed_1_step1` | `HOLD_ASK_RESEARCH` | `[0.9, 0.9, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `controlled_eval` | `delayed_1_step2` | `EXEC_MUL` | `[0.0, 0.9, 0.0]` | `EXEC_MUL` | `1` | `1` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `quote_trap_1` | `HOLD_ASK_RESEARCH` | `[0.9, 0.0, 0.0]` | `EXEC_ADD` | `0` | `0` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `quote_trap_2` | `HOLD_ASK_RESEARCH|REFRAME` | `[0.9, 0.0, 0.0]` | `EXEC_ADD` | `0` | `0` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `quote_trap_3` | `HOLD_ASK_RESEARCH` | `[0.0, 0.9, 0.0]` | `EXEC_MUL` | `0` | `0` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `scope_1` | `HOLD_ASK_RESEARCH|REFRAME` | `[0.9, 0.0, 0.0]` | `EXEC_ADD` | `0` | `0` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `scope_2` | `EXEC_MUL` | `[0.9, 0.9, 0.0]` | `HOLD_ASK_RESEARCH` | `0` | `0` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `scope_3` | `EXEC_ADD` | `[0.9, 0.9, 0.0]` | `HOLD_ASK_RESEARCH` | `0` | `0` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `scope_4` | `HOLD_ASK_RESEARCH` | `[0.9, 0.9, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `scope_5` | `HOLD_ASK_RESEARCH` | `[0.9, 0.9, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `correction_1` | `EXEC_MUL` | `[0.9, 0.9, 0.0]` | `HOLD_ASK_RESEARCH` | `0` | `0` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `correction_2` | `EXEC_ADD` | `[0.9, 0.9, 0.0]` | `HOLD_ASK_RESEARCH` | `0` | `0` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `sequence_1` | `HOLD_ASK_RESEARCH` | `[0.9, 0.9, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `morph_trap_1` | `HOLD_ASK_RESEARCH` | `[0.0, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `morph_trap_2` | `HOLD_ASK_RESEARCH` | `[0.0, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `morph_trap_3` | `HOLD_ASK_RESEARCH` | `[0.0, 0.9, 0.0]` | `EXEC_MUL` | `0` | `0` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `morph_trap_4` | `HOLD_ASK_RESEARCH` | `[0.9, 0.0, 0.0]` | `EXEC_ADD` | `0` | `0` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `weak_lexical_1` | `HOLD_ASK_RESEARCH` | `[0.9, 0.0, 0.0]` | `EXEC_ADD` | `0` | `0` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `weak_lexical_2` | `HOLD_ASK_RESEARCH` | `[0.0, 0.9, 0.0]` | `EXEC_MUL` | `0` | `0` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `weak_lexical_3` | `HOLD_ASK_RESEARCH` | `[0.9, 0.0, 0.0]` | `EXEC_ADD` | `0` | `0` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `unknown_override_1` | `REJECT_UNKNOWN|HOLD_ASK_RESEARCH` | `[0.9, 0.0, 0.9]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `unknown_override_2` | `REJECT_UNKNOWN` | `[0.0, 0.9, 0.9]` | `HOLD_ASK_RESEARCH` | `0` | `1` |
| `keyword_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `unknown_override_3` | `REJECT_UNKNOWN|HOLD_ASK_RESEARCH` | `[0.9, 0.0, 0.9]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `keyword_sensor` | `topK2_guard` | `controlled_eval` | `known_add_1` | `EXEC_ADD` | `[0.9, 0.0, 0.0]` | `EXEC_ADD` | `1` | `1` |
| `keyword_sensor` | `topK2_guard` | `controlled_eval` | `known_add_2` | `EXEC_ADD` | `[0.9, 0.0, 0.0]` | `EXEC_ADD` | `1` | `1` |
| `keyword_sensor` | `topK2_guard` | `controlled_eval` | `known_add_3` | `EXEC_ADD` | `[0.9, 0.0, 0.0]` | `EXEC_ADD` | `1` | `1` |
| `keyword_sensor` | `topK2_guard` | `controlled_eval` | `known_mul_1` | `EXEC_MUL` | `[0.0, 0.9, 0.0]` | `EXEC_MUL` | `1` | `1` |
| `keyword_sensor` | `topK2_guard` | `controlled_eval` | `known_mul_2` | `EXEC_MUL` | `[0.0, 0.9, 0.0]` | `EXEC_MUL` | `1` | `1` |
| `keyword_sensor` | `topK2_guard` | `controlled_eval` | `known_mul_3` | `EXEC_MUL` | `[0.0, 0.9, 0.0]` | `EXEC_MUL` | `1` | `1` |
| `keyword_sensor` | `topK2_guard` | `controlled_eval` | `unknown_div_1` | `REJECT_UNKNOWN` | `[0.0, 0.0, 0.9]` | `REJECT_UNKNOWN` | `1` | `1` |
| `keyword_sensor` | `topK2_guard` | `controlled_eval` | `unknown_div_2` | `REJECT_UNKNOWN` | `[0.0, 0.0, 0.9]` | `REJECT_UNKNOWN` | `1` | `1` |
| `keyword_sensor` | `topK2_guard` | `controlled_eval` | `unknown_div_3` | `REJECT_UNKNOWN` | `[0.0, 0.0, 0.9]` | `REJECT_UNKNOWN` | `1` | `1` |
| `keyword_sensor` | `topK2_guard` | `controlled_eval` | `weak_add_1` | `HOLD_ASK_RESEARCH` | `[0.9, 0.0, 0.0]` | `EXEC_ADD` | `0` | `0` |
| `keyword_sensor` | `topK2_guard` | `controlled_eval` | `weak_mul_1` | `HOLD_ASK_RESEARCH` | `[0.0, 0.9, 0.0]` | `EXEC_MUL` | `0` | `0` |
| `keyword_sensor` | `topK2_guard` | `controlled_eval` | `ambiguous_1` | `HOLD_ASK_RESEARCH` | `[0.9, 0.9, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `keyword_sensor` | `topK2_guard` | `controlled_eval` | `no_evidence_1` | `HOLD_ASK_RESEARCH` | `[0.0, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `keyword_sensor` | `topK2_guard` | `controlled_eval` | `conflict_1` | `HOLD_ASK_RESEARCH` | `[0.9, 0.9, 0.9]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `keyword_sensor` | `topK2_guard` | `controlled_eval` | `negated_add_1` | `HOLD_ASK_RESEARCH|REFRAME` | `[0.9, 0.0, 0.0]` | `EXEC_ADD` | `0` | `0` |
| `keyword_sensor` | `topK2_guard` | `controlled_eval` | `negated_add_then_mul_1` | `EXEC_MUL` | `[0.9, 0.9, 0.0]` | `HOLD_ASK_RESEARCH` | `0` | `0` |
| `keyword_sensor` | `topK2_guard` | `controlled_eval` | `delayed_1_step1` | `HOLD_ASK_RESEARCH` | `[0.9, 0.9, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `keyword_sensor` | `topK2_guard` | `controlled_eval` | `delayed_1_step2` | `EXEC_MUL` | `[0.0, 0.9, 0.0]` | `EXEC_MUL` | `1` | `1` |
| `keyword_sensor` | `topK2_guard` | `adversarial_eval` | `quote_trap_1` | `HOLD_ASK_RESEARCH` | `[0.9, 0.0, 0.0]` | `EXEC_ADD` | `0` | `0` |
| `keyword_sensor` | `topK2_guard` | `adversarial_eval` | `quote_trap_2` | `HOLD_ASK_RESEARCH|REFRAME` | `[0.9, 0.0, 0.0]` | `EXEC_ADD` | `0` | `0` |
| `keyword_sensor` | `topK2_guard` | `adversarial_eval` | `quote_trap_3` | `HOLD_ASK_RESEARCH` | `[0.0, 0.9, 0.0]` | `EXEC_MUL` | `0` | `0` |
| `keyword_sensor` | `topK2_guard` | `adversarial_eval` | `scope_1` | `HOLD_ASK_RESEARCH|REFRAME` | `[0.9, 0.0, 0.0]` | `EXEC_ADD` | `0` | `0` |
| `keyword_sensor` | `topK2_guard` | `adversarial_eval` | `scope_2` | `EXEC_MUL` | `[0.9, 0.9, 0.0]` | `HOLD_ASK_RESEARCH` | `0` | `0` |
| `keyword_sensor` | `topK2_guard` | `adversarial_eval` | `scope_3` | `EXEC_ADD` | `[0.9, 0.9, 0.0]` | `HOLD_ASK_RESEARCH` | `0` | `0` |
| `keyword_sensor` | `topK2_guard` | `adversarial_eval` | `scope_4` | `HOLD_ASK_RESEARCH` | `[0.9, 0.9, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `keyword_sensor` | `topK2_guard` | `adversarial_eval` | `scope_5` | `HOLD_ASK_RESEARCH` | `[0.9, 0.9, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `keyword_sensor` | `topK2_guard` | `adversarial_eval` | `correction_1` | `EXEC_MUL` | `[0.9, 0.9, 0.0]` | `HOLD_ASK_RESEARCH` | `0` | `0` |
| `keyword_sensor` | `topK2_guard` | `adversarial_eval` | `correction_2` | `EXEC_ADD` | `[0.9, 0.9, 0.0]` | `HOLD_ASK_RESEARCH` | `0` | `0` |
| `keyword_sensor` | `topK2_guard` | `adversarial_eval` | `sequence_1` | `HOLD_ASK_RESEARCH` | `[0.9, 0.9, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `keyword_sensor` | `topK2_guard` | `adversarial_eval` | `morph_trap_1` | `HOLD_ASK_RESEARCH` | `[0.0, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `keyword_sensor` | `topK2_guard` | `adversarial_eval` | `morph_trap_2` | `HOLD_ASK_RESEARCH` | `[0.0, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `keyword_sensor` | `topK2_guard` | `adversarial_eval` | `morph_trap_3` | `HOLD_ASK_RESEARCH` | `[0.0, 0.9, 0.0]` | `EXEC_MUL` | `0` | `0` |
| `keyword_sensor` | `topK2_guard` | `adversarial_eval` | `morph_trap_4` | `HOLD_ASK_RESEARCH` | `[0.9, 0.0, 0.0]` | `EXEC_ADD` | `0` | `0` |
| `keyword_sensor` | `topK2_guard` | `adversarial_eval` | `weak_lexical_1` | `HOLD_ASK_RESEARCH` | `[0.9, 0.0, 0.0]` | `EXEC_ADD` | `0` | `0` |
| `keyword_sensor` | `topK2_guard` | `adversarial_eval` | `weak_lexical_2` | `HOLD_ASK_RESEARCH` | `[0.0, 0.9, 0.0]` | `EXEC_MUL` | `0` | `0` |
| `keyword_sensor` | `topK2_guard` | `adversarial_eval` | `weak_lexical_3` | `HOLD_ASK_RESEARCH` | `[0.9, 0.0, 0.0]` | `EXEC_ADD` | `0` | `0` |
| `keyword_sensor` | `topK2_guard` | `adversarial_eval` | `unknown_override_1` | `REJECT_UNKNOWN|HOLD_ASK_RESEARCH` | `[0.9, 0.0, 0.9]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `keyword_sensor` | `topK2_guard` | `adversarial_eval` | `unknown_override_2` | `REJECT_UNKNOWN` | `[0.0, 0.9, 0.9]` | `HOLD_ASK_RESEARCH` | `0` | `1` |
| `keyword_sensor` | `topK2_guard` | `adversarial_eval` | `unknown_override_3` | `REJECT_UNKNOWN|HOLD_ASK_RESEARCH` | `[0.9, 0.0, 0.9]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `controlled_eval` | `known_add_1` | `EXEC_ADD` | `[0.9, 0.0, 0.0]` | `EXEC_ADD` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `controlled_eval` | `known_add_2` | `EXEC_ADD` | `[0.9, 0.0, 0.0]` | `EXEC_ADD` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `controlled_eval` | `known_add_3` | `EXEC_ADD` | `[0.9, 0.0, 0.0]` | `EXEC_ADD` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `controlled_eval` | `known_mul_1` | `EXEC_MUL` | `[0.0, 0.9, 0.0]` | `EXEC_MUL` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `controlled_eval` | `known_mul_2` | `EXEC_MUL` | `[0.0, 0.9, 0.0]` | `EXEC_MUL` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `controlled_eval` | `known_mul_3` | `EXEC_MUL` | `[0.0, 0.9, 0.0]` | `EXEC_MUL` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `controlled_eval` | `unknown_div_1` | `REJECT_UNKNOWN` | `[0.0, 0.0, 0.9]` | `REJECT_UNKNOWN` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `controlled_eval` | `unknown_div_2` | `REJECT_UNKNOWN` | `[0.0, 0.0, 0.9]` | `REJECT_UNKNOWN` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `controlled_eval` | `unknown_div_3` | `REJECT_UNKNOWN` | `[0.0, 0.0, 0.9]` | `REJECT_UNKNOWN` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `controlled_eval` | `weak_add_1` | `HOLD_ASK_RESEARCH` | `[0.45, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `controlled_eval` | `weak_mul_1` | `HOLD_ASK_RESEARCH` | `[0.0, 0.45, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `controlled_eval` | `ambiguous_1` | `HOLD_ASK_RESEARCH` | `[0.9, 0.9, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `controlled_eval` | `no_evidence_1` | `HOLD_ASK_RESEARCH` | `[0.0, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `controlled_eval` | `conflict_1` | `HOLD_ASK_RESEARCH` | `[0.8, 0.8, 0.8]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `controlled_eval` | `negated_add_1` | `HOLD_ASK_RESEARCH|REFRAME` | `[0.0, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `controlled_eval` | `negated_add_then_mul_1` | `EXEC_MUL` | `[0.0, 0.9, 0.0]` | `EXEC_MUL` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `controlled_eval` | `delayed_1_step1` | `HOLD_ASK_RESEARCH` | `[0.9, 0.9, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `controlled_eval` | `delayed_1_step2` | `EXEC_MUL` | `[0.0, 0.9, 0.0]` | `EXEC_MUL` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `quote_trap_1` | `HOLD_ASK_RESEARCH` | `[0.0, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `quote_trap_2` | `HOLD_ASK_RESEARCH|REFRAME` | `[0.0, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `quote_trap_3` | `HOLD_ASK_RESEARCH` | `[0.0, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `scope_1` | `HOLD_ASK_RESEARCH|REFRAME` | `[0.0, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `scope_2` | `EXEC_MUL` | `[0.0, 0.9, 0.0]` | `EXEC_MUL` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `scope_3` | `EXEC_ADD` | `[0.9, 0.0, 0.0]` | `EXEC_ADD` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `scope_4` | `HOLD_ASK_RESEARCH` | `[0.0, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `scope_5` | `HOLD_ASK_RESEARCH` | `[0.5, 0.5, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `correction_1` | `EXEC_MUL` | `[0.0, 0.9, 0.0]` | `EXEC_MUL` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `correction_2` | `EXEC_ADD` | `[0.9, 0.0, 0.0]` | `EXEC_ADD` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `sequence_1` | `HOLD_ASK_RESEARCH` | `[0.8, 0.8, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `morph_trap_1` | `HOLD_ASK_RESEARCH` | `[0.0, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `morph_trap_2` | `HOLD_ASK_RESEARCH` | `[0.0, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `morph_trap_3` | `HOLD_ASK_RESEARCH` | `[0.0, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `morph_trap_4` | `HOLD_ASK_RESEARCH` | `[0.0, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `weak_lexical_1` | `HOLD_ASK_RESEARCH` | `[0.45, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `weak_lexical_2` | `HOLD_ASK_RESEARCH` | `[0.0, 0.45, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `weak_lexical_3` | `HOLD_ASK_RESEARCH` | `[0.45, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `unknown_override_1` | `REJECT_UNKNOWN|HOLD_ASK_RESEARCH` | `[0.0, 0.0, 0.9]` | `REJECT_UNKNOWN` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `unknown_override_2` | `REJECT_UNKNOWN` | `[0.0, 0.0, 0.9]` | `REJECT_UNKNOWN` | `1` | `1` |
| `structured_rule_sensor` | `evidence_strength_margin_guard` | `adversarial_eval` | `unknown_override_3` | `REJECT_UNKNOWN|HOLD_ASK_RESEARCH` | `[0.25, 0.0, 0.9]` | `REJECT_UNKNOWN` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `controlled_eval` | `known_add_1` | `EXEC_ADD` | `[0.9, 0.0, 0.0]` | `EXEC_ADD` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `controlled_eval` | `known_add_2` | `EXEC_ADD` | `[0.9, 0.0, 0.0]` | `EXEC_ADD` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `controlled_eval` | `known_add_3` | `EXEC_ADD` | `[0.9, 0.0, 0.0]` | `EXEC_ADD` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `controlled_eval` | `known_mul_1` | `EXEC_MUL` | `[0.0, 0.9, 0.0]` | `EXEC_MUL` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `controlled_eval` | `known_mul_2` | `EXEC_MUL` | `[0.0, 0.9, 0.0]` | `EXEC_MUL` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `controlled_eval` | `known_mul_3` | `EXEC_MUL` | `[0.0, 0.9, 0.0]` | `EXEC_MUL` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `controlled_eval` | `unknown_div_1` | `REJECT_UNKNOWN` | `[0.0, 0.0, 0.9]` | `REJECT_UNKNOWN` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `controlled_eval` | `unknown_div_2` | `REJECT_UNKNOWN` | `[0.0, 0.0, 0.9]` | `REJECT_UNKNOWN` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `controlled_eval` | `unknown_div_3` | `REJECT_UNKNOWN` | `[0.0, 0.0, 0.9]` | `REJECT_UNKNOWN` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `controlled_eval` | `weak_add_1` | `HOLD_ASK_RESEARCH` | `[0.45, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `controlled_eval` | `weak_mul_1` | `HOLD_ASK_RESEARCH` | `[0.0, 0.45, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `controlled_eval` | `ambiguous_1` | `HOLD_ASK_RESEARCH` | `[0.9, 0.9, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `controlled_eval` | `no_evidence_1` | `HOLD_ASK_RESEARCH` | `[0.0, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `controlled_eval` | `conflict_1` | `HOLD_ASK_RESEARCH` | `[0.8, 0.8, 0.8]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `controlled_eval` | `negated_add_1` | `HOLD_ASK_RESEARCH|REFRAME` | `[0.0, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `controlled_eval` | `negated_add_then_mul_1` | `EXEC_MUL` | `[0.0, 0.9, 0.0]` | `EXEC_MUL` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `controlled_eval` | `delayed_1_step1` | `HOLD_ASK_RESEARCH` | `[0.9, 0.9, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `controlled_eval` | `delayed_1_step2` | `EXEC_MUL` | `[0.0, 0.9, 0.0]` | `EXEC_MUL` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `adversarial_eval` | `quote_trap_1` | `HOLD_ASK_RESEARCH` | `[0.0, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `adversarial_eval` | `quote_trap_2` | `HOLD_ASK_RESEARCH|REFRAME` | `[0.0, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `adversarial_eval` | `quote_trap_3` | `HOLD_ASK_RESEARCH` | `[0.0, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `adversarial_eval` | `scope_1` | `HOLD_ASK_RESEARCH|REFRAME` | `[0.0, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `adversarial_eval` | `scope_2` | `EXEC_MUL` | `[0.0, 0.9, 0.0]` | `EXEC_MUL` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `adversarial_eval` | `scope_3` | `EXEC_ADD` | `[0.9, 0.0, 0.0]` | `EXEC_ADD` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `adversarial_eval` | `scope_4` | `HOLD_ASK_RESEARCH` | `[0.0, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `adversarial_eval` | `scope_5` | `HOLD_ASK_RESEARCH` | `[0.5, 0.5, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `adversarial_eval` | `correction_1` | `EXEC_MUL` | `[0.0, 0.9, 0.0]` | `EXEC_MUL` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `adversarial_eval` | `correction_2` | `EXEC_ADD` | `[0.9, 0.0, 0.0]` | `EXEC_ADD` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `adversarial_eval` | `sequence_1` | `HOLD_ASK_RESEARCH` | `[0.8, 0.8, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `adversarial_eval` | `morph_trap_1` | `HOLD_ASK_RESEARCH` | `[0.0, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `adversarial_eval` | `morph_trap_2` | `HOLD_ASK_RESEARCH` | `[0.0, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `adversarial_eval` | `morph_trap_3` | `HOLD_ASK_RESEARCH` | `[0.0, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `adversarial_eval` | `morph_trap_4` | `HOLD_ASK_RESEARCH` | `[0.0, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `adversarial_eval` | `weak_lexical_1` | `HOLD_ASK_RESEARCH` | `[0.45, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `adversarial_eval` | `weak_lexical_2` | `HOLD_ASK_RESEARCH` | `[0.0, 0.45, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `adversarial_eval` | `weak_lexical_3` | `HOLD_ASK_RESEARCH` | `[0.45, 0.0, 0.0]` | `HOLD_ASK_RESEARCH` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `adversarial_eval` | `unknown_override_1` | `REJECT_UNKNOWN|HOLD_ASK_RESEARCH` | `[0.0, 0.0, 0.9]` | `REJECT_UNKNOWN` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `adversarial_eval` | `unknown_override_2` | `REJECT_UNKNOWN` | `[0.0, 0.0, 0.9]` | `REJECT_UNKNOWN` | `1` | `1` |
| `structured_rule_sensor` | `topK2_guard` | `adversarial_eval` | `unknown_override_3` | `REJECT_UNKNOWN|HOLD_ASK_RESEARCH` | `[0.25, 0.0, 0.9]` | `REJECT_UNKNOWN` | `1` | `1` |

## Failure Cases

- `keyword_sensor+evidence_strength_margin_guard` on `weak_add_1` (maybe add 3): expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD`, evidence `[0.9, 0.0, 0.0]`.
- `keyword_sensor+evidence_strength_margin_guard` on `weak_mul_1` (maybe multiply): expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL`, evidence `[0.0, 0.9, 0.0]`.
- `keyword_sensor+evidence_strength_margin_guard` on `negated_add_1` (do not add 3): expected `HOLD_ASK_RESEARCH|REFRAME`, got `EXEC_ADD`, evidence `[0.9, 0.0, 0.0]`.
- `keyword_sensor+evidence_strength_margin_guard` on `negated_add_then_mul_1` (do not add 3, multiply by 3): expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH`, evidence `[0.9, 0.9, 0.0]`.
- `keyword_sensor+evidence_strength_margin_guard` on `quote_trap_1` (the word add appears in the note): expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD`, evidence `[0.9, 0.0, 0.0]`.
- `keyword_sensor+evidence_strength_margin_guard` on `quote_trap_2` (do not follow the instruction 'add 3'): expected `HOLD_ASK_RESEARCH|REFRAME`, got `EXEC_ADD`, evidence `[0.9, 0.0, 0.0]`.
- `keyword_sensor+evidence_strength_margin_guard` on `quote_trap_3` (someone said multiply, but no operation is requested): expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL`, evidence `[0.0, 0.9, 0.0]`.
- `keyword_sensor+evidence_strength_margin_guard` on `scope_1` (do not add 3): expected `HOLD_ASK_RESEARCH|REFRAME`, got `EXEC_ADD`, evidence `[0.9, 0.0, 0.0]`.
- `keyword_sensor+evidence_strength_margin_guard` on `scope_2` (do not add 3, multiply by 3 instead): expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH`, evidence `[0.9, 0.9, 0.0]`.
- `keyword_sensor+evidence_strength_margin_guard` on `scope_3` (do not multiply by 3, add 3 instead): expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH`, evidence `[0.9, 0.9, 0.0]`.
- `keyword_sensor+evidence_strength_margin_guard` on `correction_1` (add 3. wait, actually multiply by 3): expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH`, evidence `[0.9, 0.9, 0.0]`.
- `keyword_sensor+evidence_strength_margin_guard` on `correction_2` (multiply by 3. correction: add 3): expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH`, evidence `[0.9, 0.9, 0.0]`.
- `keyword_sensor+evidence_strength_margin_guard` on `morph_trap_3` (multiply-styled text, but no operation): expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL`, evidence `[0.0, 0.9, 0.0]`.
- `keyword_sensor+evidence_strength_margin_guard` on `morph_trap_4` (plus sign appears on the page): expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD`, evidence `[0.9, 0.0, 0.0]`.
- `keyword_sensor+evidence_strength_margin_guard` on `weak_lexical_1` (maybe add 3): expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD`, evidence `[0.9, 0.0, 0.0]`.
- `keyword_sensor+evidence_strength_margin_guard` on `weak_lexical_2` (probably multiply by 3): expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL`, evidence `[0.0, 0.9, 0.0]`.
- `keyword_sensor+evidence_strength_margin_guard` on `weak_lexical_3` (it could be add): expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD`, evidence `[0.9, 0.0, 0.0]`.
- `keyword_sensor+evidence_strength_margin_guard` on `unknown_override_2` (use sqrt; do not multiply): expected `REJECT_UNKNOWN`, got `HOLD_ASK_RESEARCH`, evidence `[0.0, 0.9, 0.9]`.
- `keyword_sensor+topK2_guard` on `weak_add_1` (maybe add 3): expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD`, evidence `[0.9, 0.0, 0.0]`.
- `keyword_sensor+topK2_guard` on `weak_mul_1` (maybe multiply): expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL`, evidence `[0.0, 0.9, 0.0]`.
- `keyword_sensor+topK2_guard` on `negated_add_1` (do not add 3): expected `HOLD_ASK_RESEARCH|REFRAME`, got `EXEC_ADD`, evidence `[0.9, 0.0, 0.0]`.
- `keyword_sensor+topK2_guard` on `negated_add_then_mul_1` (do not add 3, multiply by 3): expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH`, evidence `[0.9, 0.9, 0.0]`.
- `keyword_sensor+topK2_guard` on `quote_trap_1` (the word add appears in the note): expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD`, evidence `[0.9, 0.0, 0.0]`.
- `keyword_sensor+topK2_guard` on `quote_trap_2` (do not follow the instruction 'add 3'): expected `HOLD_ASK_RESEARCH|REFRAME`, got `EXEC_ADD`, evidence `[0.9, 0.0, 0.0]`.
- `keyword_sensor+topK2_guard` on `quote_trap_3` (someone said multiply, but no operation is requested): expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL`, evidence `[0.0, 0.9, 0.0]`.
- `keyword_sensor+topK2_guard` on `scope_1` (do not add 3): expected `HOLD_ASK_RESEARCH|REFRAME`, got `EXEC_ADD`, evidence `[0.9, 0.0, 0.0]`.
- `keyword_sensor+topK2_guard` on `scope_2` (do not add 3, multiply by 3 instead): expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH`, evidence `[0.9, 0.9, 0.0]`.
- `keyword_sensor+topK2_guard` on `scope_3` (do not multiply by 3, add 3 instead): expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH`, evidence `[0.9, 0.9, 0.0]`.
- `keyword_sensor+topK2_guard` on `correction_1` (add 3. wait, actually multiply by 3): expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH`, evidence `[0.9, 0.9, 0.0]`.
- `keyword_sensor+topK2_guard` on `correction_2` (multiply by 3. correction: add 3): expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH`, evidence `[0.9, 0.9, 0.0]`.
- `keyword_sensor+topK2_guard` on `morph_trap_3` (multiply-styled text, but no operation): expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL`, evidence `[0.0, 0.9, 0.0]`.
- `keyword_sensor+topK2_guard` on `morph_trap_4` (plus sign appears on the page): expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD`, evidence `[0.9, 0.0, 0.0]`.
- `keyword_sensor+topK2_guard` on `weak_lexical_1` (maybe add 3): expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD`, evidence `[0.9, 0.0, 0.0]`.
- `keyword_sensor+topK2_guard` on `weak_lexical_2` (probably multiply by 3): expected `HOLD_ASK_RESEARCH`, got `EXEC_MUL`, evidence `[0.0, 0.9, 0.0]`.
- `keyword_sensor+topK2_guard` on `weak_lexical_3` (it could be add): expected `HOLD_ASK_RESEARCH`, got `EXEC_ADD`, evidence `[0.9, 0.0, 0.0]`.
- `keyword_sensor+topK2_guard` on `unknown_override_2` (use sqrt; do not multiply): expected `REJECT_UNKNOWN`, got `HOLD_ASK_RESEARCH`, evidence `[0.0, 0.9, 0.9]`.

## Interpretation

A positive result means controlled command text can be mapped to guard-compatible evidence under adversarial keyword and scope traps.

If controlled eval passes but adversarial eval fails, the sensor is only a rule sanity check, not a robust evidence extractor.

## Claim Boundary

No general natural-language understanding, full Pilot Pulse learning, full VRAXION/INSTNCT proof, production architecture, or consciousness claim.

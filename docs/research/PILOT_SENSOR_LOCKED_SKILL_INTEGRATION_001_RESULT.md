# PILOT_SENSOR_LOCKED_SKILL_INTEGRATION_001 Result

## Goal

Test the full toy path from text sensor to fixed guard to frozen ADD/MUL skill execution.

## Setup

- Frozen skill modules are deterministic `add(value, operand)` and `mul(value, operand)` functions.
- The pilot action decides execute, reject, or hold. HOLD/REJECT must not execute a skill.
- Evaluation combines robustness stress cases and factor-heldout cases.

## Aggregate Metrics

| Model | Seeds | Action | Result | Exec Result | False Exec | Missed Exec | Weak | Amb | Neg | Corr | Drift | Strict Syn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `keyword_sensor` | `1` | `0.413` | `0.455` | `0.364` | `0.355` | `0.231` | `0.000` | `1.000` | `0.000` | `0.000` | `0.000` | `0.000` |
| `learned_systematic_char_sensor` | `10` | `1.000` | `1.000` | `1.000` | `0.000` | `0.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` | `0.000` |
| `structured_rule_sensor` | `1` | `1.000` | `1.000` | `1.000` | `0.000` | `0.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` | `0.000` |

## Verdict

```json
{
  "global": [
    "LOCKED_SKILL_PIPELINE_PASS_WITH_STRUCTURED_SENSOR"
  ],
  "keyword_sensor": [
    "KEYWORD_EXECUTION_BASELINE_FAILS",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED"
  ],
  "learned_systematic_char_sensor": [
    "LEARNED_SENSOR_LOCKED_SKILL_POSITIVE",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED"
  ],
  "structured_rule_sensor": [
    "STRUCTURED_SENSOR_LOCKED_SKILL_POSITIVE",
    "STRICT_UNSEEN_SYNONYM_UNSOLVED"
  ]
}
```

## Failure Examples

- `keyword_sensor` seed `-1` `stress_eval/weak`: probably plus 7 now value `14` -> expected `HOLD_ASK_RESEARCH`/`None`, got `EXEC_ADD`/`21` (false_execution).
- `keyword_sensor` seed `-1` `stress_eval/weak`: might times 7 please value `10` -> expected `HOLD_ASK_RESEARCH`/`None`, got `EXEC_MUL`/`70` (false_execution).
- `keyword_sensor` seed `-1` `stress_eval/weak`: it could be add 7 value `11` -> expected `HOLD_ASK_RESEARCH`/`None`, got `EXEC_ADD`/`18` (false_execution).
- `keyword_sensor` seed `-1` `stress_eval/weak`: unsure multiply by 7 value `12` -> expected `HOLD_ASK_RESEARCH`/`None`, got `EXEC_MUL`/`84` (false_execution).
- `keyword_sensor` seed `-1` `stress_eval/negation`: not plus 7 value `11` -> expected `HOLD_ASK_RESEARCH`/`None`, got `EXEC_ADD`/`18` (false_execution).
- `keyword_sensor` seed `-1` `stress_eval/negation`: do not multiply by 7, add 7 instead value `12` -> expected `EXEC_ADD`/`19`, got `HOLD_ASK_RESEARCH`/`None` (missed_execution).
- `keyword_sensor` seed `-1` `stress_eval/negation`: do not add 7, multiply by 7 instead value `13` -> expected `EXEC_MUL`/`91`, got `HOLD_ASK_RESEARCH`/`None` (missed_execution).
- `keyword_sensor` seed `-1` `stress_eval/correction`: please add 7; actually multiply by 7 value `14` -> expected `EXEC_MUL`/`98`, got `HOLD_ASK_RESEARCH`/`None` (missed_execution).
- `keyword_sensor` seed `-1` `stress_eval/correction`: times 7. correction: plus 7 value `10` -> expected `EXEC_ADD`/`17`, got `HOLD_ASK_RESEARCH`/`None` (missed_execution).
- `keyword_sensor` seed `-1` `stress_eval/weak`: probably plus 8 now value `11` -> expected `HOLD_ASK_RESEARCH`/`None`, got `EXEC_ADD`/`19` (false_execution).
- `keyword_sensor` seed `-1` `stress_eval/weak`: might times 8 please value `12` -> expected `HOLD_ASK_RESEARCH`/`None`, got `EXEC_MUL`/`96` (false_execution).
- `keyword_sensor` seed `-1` `stress_eval/weak`: it could be add 8 value `13` -> expected `HOLD_ASK_RESEARCH`/`None`, got `EXEC_ADD`/`21` (false_execution).
- `keyword_sensor` seed `-1` `stress_eval/weak`: unsure multiply by 8 value `14` -> expected `HOLD_ASK_RESEARCH`/`None`, got `EXEC_MUL`/`112` (false_execution).
- `keyword_sensor` seed `-1` `stress_eval/negation`: not plus 8 value `13` -> expected `HOLD_ASK_RESEARCH`/`None`, got `EXEC_ADD`/`21` (false_execution).
- `keyword_sensor` seed `-1` `stress_eval/negation`: do not multiply by 8, add 8 instead value `14` -> expected `EXEC_ADD`/`22`, got `HOLD_ASK_RESEARCH`/`None` (missed_execution).
- `keyword_sensor` seed `-1` `stress_eval/negation`: do not add 8, multiply by 8 instead value `10` -> expected `EXEC_MUL`/`80`, got `HOLD_ASK_RESEARCH`/`None` (missed_execution).
- `keyword_sensor` seed `-1` `stress_eval/correction`: please add 8; actually multiply by 8 value `11` -> expected `EXEC_MUL`/`88`, got `HOLD_ASK_RESEARCH`/`None` (missed_execution).
- `keyword_sensor` seed `-1` `stress_eval/correction`: times 8. correction: plus 8 value `12` -> expected `EXEC_ADD`/`20`, got `HOLD_ASK_RESEARCH`/`None` (missed_execution).
- `keyword_sensor` seed `-1` `stress_eval/weak`: probably plus 9 now value `13` -> expected `HOLD_ASK_RESEARCH`/`None`, got `EXEC_ADD`/`22` (false_execution).
- `keyword_sensor` seed `-1` `stress_eval/weak`: might times 9 please value `14` -> expected `HOLD_ASK_RESEARCH`/`None`, got `EXEC_MUL`/`126` (false_execution).
- `keyword_sensor` seed `-1` `stress_eval/weak`: it could be add 9 value `10` -> expected `HOLD_ASK_RESEARCH`/`None`, got `EXEC_ADD`/`19` (false_execution).
- `keyword_sensor` seed `-1` `stress_eval/weak`: unsure multiply by 9 value `11` -> expected `HOLD_ASK_RESEARCH`/`None`, got `EXEC_MUL`/`99` (false_execution).
- `keyword_sensor` seed `-1` `stress_eval/negation`: not plus 9 value `10` -> expected `HOLD_ASK_RESEARCH`/`None`, got `EXEC_ADD`/`19` (false_execution).
- `keyword_sensor` seed `-1` `stress_eval/negation`: do not multiply by 9, add 9 instead value `11` -> expected `EXEC_ADD`/`20`, got `HOLD_ASK_RESEARCH`/`None` (missed_execution).
- `keyword_sensor` seed `-1` `stress_eval/negation`: do not add 9, multiply by 9 instead value `12` -> expected `EXEC_MUL`/`108`, got `HOLD_ASK_RESEARCH`/`None` (missed_execution).
- `keyword_sensor` seed `-1` `stress_eval/correction`: please add 9; actually multiply by 9 value `13` -> expected `EXEC_MUL`/`117`, got `HOLD_ASK_RESEARCH`/`None` (missed_execution).
- `keyword_sensor` seed `-1` `stress_eval/correction`: times 9. correction: plus 9 value `14` -> expected `EXEC_ADD`/`23`, got `HOLD_ASK_RESEARCH`/`None` (missed_execution).
- `keyword_sensor` seed `-1` `stress_eval/weak`: probably plus 11 now value `10` -> expected `HOLD_ASK_RESEARCH`/`None`, got `EXEC_ADD`/`21` (false_execution).
- `keyword_sensor` seed `-1` `stress_eval/weak`: might times 11 please value `11` -> expected `HOLD_ASK_RESEARCH`/`None`, got `EXEC_MUL`/`121` (false_execution).
- `keyword_sensor` seed `-1` `stress_eval/weak`: it could be add 11 value `12` -> expected `HOLD_ASK_RESEARCH`/`None`, got `EXEC_ADD`/`23` (false_execution).
- ... 101 more in `failure_examples.jsonl`.

## Interpretation

A positive structured-sensor result means the hand-auditable sensor, fixed guard, and frozen skills compose into a working toy execution path.
Learned sensor weakness here is expected from the factor-heldout result and marks the raw text-to-scope stage as the blocker.

## Claim Boundary

No general NLU, full PilotPulse integration, production VRAXION/INSTNCT, or consciousness claim.

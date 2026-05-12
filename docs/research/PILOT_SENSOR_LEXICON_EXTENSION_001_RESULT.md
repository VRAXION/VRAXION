# PILOT_SENSOR_LEXICON_EXTENSION_001 Result

## Goal

Test whether the strict synonym failures are lexical coverage failures rather than guard or execution-policy failures.

## Aggregate Metrics

| Sensor | Action | Strict Syn | Scope Alias | Regression | False Commit | Missed Execute |
|---|---:|---:|---:|---:|---:|---:|
| `alias_extended_sensor` | `1.000` | `1.000` | `1.000` | `1.000` | `0.000` | `0.000` |
| `base_structured_sensor` | `0.571` | `0.000` | `0.833` | `1.000` | `0.000` | `0.286` |

## Verdict

```json
{
  "alias_extended_sensor": [
    "LEXICON_EXTENSION_POSITIVE"
  ],
  "base_structured_sensor": [
    "BASE_STRICT_SYNONYM_WEAK"
  ],
  "global": [
    "STRICT_SYNONYM_IS_LEXICON_COVERAGE"
  ]
}
```

## Failure Examples

- `base_structured_sensor` `strict_alias/strict_synonym`: increment by 9 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH`.
- `base_structured_sensor` `strict_alias/strict_synonym`: raise the value by 9 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH`.
- `base_structured_sensor` `strict_alias/strict_synonym`: product with 9 -> expected `EXEC_MUL`, got `HOLD_ASK_RESEARCH`.
- `base_structured_sensor` `strict_alias/strict_synonym`: halve it -> expected `REJECT_UNKNOWN`, got `HOLD_ASK_RESEARCH`.
- `base_structured_sensor` `strict_alias/strict_synonym`: exponentiate by 3 -> expected `REJECT_UNKNOWN`, got `HOLD_ASK_RESEARCH`.
- `base_structured_sensor` `scope_alias/correction`: product with 3. correction: increment by 3 -> expected `EXEC_ADD`, got `HOLD_ASK_RESEARCH`.

## Interpretation

A positive alias sensor result means the previous strict synonym failures are solved by explicit lexicon coverage in this toy command grammar.
This is not semantic generalization; it is a bounded lexical normalizer in front of the existing scope-aware sensor.

## Claim Boundary

No general NLU, pretrained semantics, full PilotPulse integration, production VRAXION/INSTNCT, or consciousness claim.
